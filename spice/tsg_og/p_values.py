import logging
from tqdm.auto import tqdm
from copy import deepcopy

import numpy as np

from spice import data_loaders
from spice.utils import get_logger
from spice.segmentation import get_events_at_position_all_ls
from spice.tsg_og.simulation import resimulate_events_multiple, copy_list_of_selection_points
from spice.tsg_og.detection import (
    convolution_simulation_per_ls, SelectionPoints, within_ci_fitness_filter,
    _optimize_selection_points)
from spice.length_scales import DEFAULT_SEGMENT_SIZE_DICT, LS_I_DICT

logger = get_logger('tsg_og_p_values')

CENTROMERES_OBSERVED = data_loaders.load_centromeres(extended=False, observed=True)
CHROM_LENS = data_loaders.load_chrom_lengths()

def p_value_using_resim(
        cur_chrom,
        cur_up_down,
        N_test,
        data_per_length_scale,
        n_iterations_optim=1_000,
        blocked_distance_th=2e5,
        within_ci_filtering=True,
        log_progress=False,
        skip_tqdm=False,
        save_all=False,
        save_outliers=None,
        segment_size_dict=DEFAULT_SEGMENT_SIZE_DICT):
    """Calculate p-values using resimulation with random position selection.
    
    Args:
        cur_chrom: Chromosome to analyze
        cur_up_down: Either 'up' (gains) or 'down' (losses)
        N_test: Number of random simulations to perform
        n_iterations_optim: Number of optimization iterations (default: 1000)
    """
    assert cur_up_down in ['up', 'down'], "cur_up_down must be either 'up' or 'down'"
    
    logging.getLogger('tsg_og_detection').setLevel(logging.WARNING)

    results = []
    for iteration in tqdm(range(N_test), disable=skip_tqdm, desc="P-value iterations"):
        if log_progress:
            logger.info(f'Starting iteration {iteration+1} / {N_test}')
        resim = resimulate_events_multiple(
            cur_chrom, data_per_length_scale, None,
            N_sims=1, segment_size_dict=segment_size_dict, n_cores=1,
            normalize_from_signal=True)
        cur_resim = [x[0] for x in resim]

        # Determine cur_pos using random logic
        p_arm_length = CENTROMERES_OBSERVED.loc[cur_chrom, 'small']['centro_start']
        q_arm_length = (CHROM_LENS.loc[cur_chrom] - 
                        CENTROMERES_OBSERVED.loc[cur_chrom, 'small']['centro_end'])
        is_p_arm = np.random.choice(
            [True, False],
            p=[p_arm_length/(p_arm_length+q_arm_length), q_arm_length/(p_arm_length+q_arm_length)])
        if is_p_arm:
            cur_pos = np.random.randint(1e6, CENTROMERES_OBSERVED.loc[cur_chrom, 'small']['centro_start']-1e6)
        else:
            cur_pos = np.random.randint(
                CENTROMERES_OBSERVED.loc[cur_chrom, 'small']['centro_end']+1e6,
                CHROM_LENS.loc[cur_chrom]-1e6)

        data_per_length_scale_ = deepcopy(data_per_length_scale)
        for key, i in LS_I_DICT.items():
            if key[0] == 'combined':
                continue
            data_per_length_scale_[key]['signals'] = cur_resim[i]
            signal_std = (data_per_length_scale_[key]['signal_bounds'][1] - 
                        data_per_length_scale_[key]['signal_bounds'][0])

            data_per_length_scale_[key]['signal_bounds'] = (
                cur_resim[i] - signal_std/2,
                cur_resim[i] + signal_std/2
            )
        base_selection_points = 8*[[SelectionPoints(loci=[(cur_pos, 0)])]]
        up_down_order = np.array([cur_up_down=='up'])
        optimized_selection_points_per_cluster, _, _ = _optimize_selection_points(
            n_iterations_optim, 
            list(zip(*base_selection_points)), 
            data_per_length_scale_, 
            cur_chrom,
            best_loss=np.inf, 
            show_progress=False, 
            N_iterations_base=0, 
            segment_size_dict=segment_size_dict,
            allow_pos_change=False,
            up_down_order=up_down_order,
            blocked_distance_th=blocked_distance_th
        )
        optimized_selection_points = list(zip(*optimized_selection_points_per_cluster))
        optimized_selection_points_raw = copy_list_of_selection_points(optimized_selection_points)

        if within_ci_filtering:
            optimized_selection_points = p_values_within_ci_filter(
                cur_chrom,
                optimized_selection_points,
                cur_resim,
                data_per_length_scale
            )

        all_events_at_pos = get_events_at_position_all_ls(data_per_length_scale_, cur_chrom, cur_pos)
        loci_fitness = np.maximum(0, np.array([x[0][0].fitness for x in optimized_selection_points]))
        added_events_ = (all_events_at_pos * loci_fitness) / (loci_fitness + 1)
        added_events = np.sum(added_events_)

        if save_all or (save_outliers is not None and added_events >= save_outliers):
            results.append({
                'added_events': added_events,
                'optimized_selection_points': optimized_selection_points,
                'optimized_selection_points_raw': optimized_selection_points_raw,
                'cur_resim': cur_resim
            })
        else:
            results.append({'added_events': added_events})

    return results


def p_values_within_ci_filter(cur_chrom, optimized_selection_points, cur_resim, data_per_length_scale):

    data_per_length_scale_ = deepcopy(data_per_length_scale)
    for key, i in LS_I_DICT.items():
        if key[0] == 'combined':
            continue
        data_per_length_scale_[key]['signals'] = cur_resim[i]
        signal_std = (data_per_length_scale_[key]['signal_bounds'][1] - 
                    data_per_length_scale_[key]['signal_bounds'][0])

        data_per_length_scale_[key]['signal_bounds'] = (
            cur_resim[i] - signal_std/2,
            cur_resim[i] + signal_std/2
        )

    filtered_selection_points = within_ci_fitness_filter(
            cur_chrom=cur_chrom,
            ranked_selection_points=optimized_selection_points,
            data_per_length_scale=data_per_length_scale_,
            remove_empty_loci=False)
    return filtered_selection_points


def get_actual_p_values_from_results(cur_loci, results, N_random):
    return ((
        np.sum(cur_loci["added_events"].values[:, None] <
               np.array([x["added_events"] for x in results]), axis=1) + 1) / (N_random+1)
)