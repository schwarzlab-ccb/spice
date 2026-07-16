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
    _optimize_selection_points, TELOMERES_OBSERVED)
from spice.length_scales import LENGTH_SCALE_NAMES, DEFAULT_SEGMENT_SIZE_DICT, LS_I_DICT

logger = get_logger('tsg_og_p_values')

CENTROMERES_OBSERVED = data_loaders.load_centromeres(extended=False, observed=True)
CHROM_LENS = data_loaders.load_chrom_lengths()

# Length-scale slot indices per direction in the optimized-fitness 8-vector
# (order: small_gain, small_loss, mid1_gain, mid1_loss, mid2_gain, mid2_loss, large_gain, large_loss).
# Used by the 'fitness' test statistic (see get_actual_p_values_from_results).
_DIR_SLOTS = {'up': [0, 2, 4, 6], 'down': [1, 3, 5, 7]}

def p_value_using_resim(
        cur_chrom,
        cur_up_down,
        N_test,
        data_per_length_scale,
        mode='random',
        n_iterations_optim=None,
        blocked_distance_th=2e5,
        within_ci_filtering=True,
        log_progress=False,
        skip_tqdm=False,
        save_all=False,
        save_outliers=None,
        segment_size_dict=DEFAULT_SEGMENT_SIZE_DICT):
    """Calculate p-values using resimulation.

    Args:
        cur_chrom: Chromosome to analyze
        cur_up_down: Either 'up' (gains) or 'down' (losses)
        N_test: Number of simulations to perform
        mode: Either 'random' (uniformly random loci, the SPICE default) or 'top' (loci at the
            position of the largest residual per resimulation, mirroring locus detection itself)
        n_iterations_optim: Number of optimization iterations (defaults: 1000 for 'random',
            5000 for 'top')
    """
    assert cur_up_down in ['up', 'down'], "cur_up_down must be either 'up' or 'down'"
    assert mode in ['top', 'random'], f"mode must be 'top' or 'random', got {mode!r}"

    if n_iterations_optim is None:
        n_iterations_optim = 5_000 if mode == 'top' else 1_000

    logging.getLogger('tsg_og_detection').setLevel(logging.WARNING)

    # Length scales matching cur_up_down's direction in the 8-slot gain/loss ordering (see
    # _DIR_SLOTS), and the telomere/centromere blocking regions, used by 'top' mode to pick the
    # position of the largest same-direction residual -- mirrors locus detection itself.
    if mode == 'top':
        length_scales_for_residuals = np.arange(0, 8, 2).astype(int) + (0 if cur_up_down == 'up' else 1)
        tel_cen_distance_th = max(blocked_distance_th, segment_size_dict['large'])
        telomere_block_start = int((TELOMERES_OBSERVED.loc[cur_chrom, 'small']['chrom_start'] + tel_cen_distance_th) / segment_size_dict['small'])
        telomere_block_end = int((TELOMERES_OBSERVED.loc[cur_chrom, 'small']['chrom_end'] - tel_cen_distance_th) / segment_size_dict['small'])
        centromere_block_start = int((CENTROMERES_OBSERVED.loc[cur_chrom, 'small']['centro_start'] - tel_cen_distance_th) / segment_size_dict['small'])
        centromere_block_end = int((CENTROMERES_OBSERVED.loc[cur_chrom, 'small']['centro_end'] + tel_cen_distance_th) / segment_size_dict['small'])

    results = []
    for iteration in tqdm(range(N_test), disable=skip_tqdm, desc="P-value iterations"):
        if log_progress:
            logger.info(f'Starting iteration {iteration+1} / {N_test}')
        resim = resimulate_events_multiple(
            cur_chrom, data_per_length_scale, None,
            N_sims=1, segment_size_dict=segment_size_dict, n_cores=1,
            normalize_from_signal=True)
        cur_resim = [x[0] for x in resim]

        if mode == 'top':
            # Position of the largest same-direction residual in this resimulation, blocking
            # telomeres, the centromere, and regions already within the bootstrap CI -- same
            # construction as the residual-driven locus search in detect_tsgs_ogs_for_all_length_scales.
            conv_sim = convolution_simulation_per_ls(cur_chrom, data_per_length_scale, None,
                                    segment_size_dict=segment_size_dict)
            cur_residuals = [(rs - generated_signal) / data['cur_loss_norm']
                        for rs, data, generated_signal in
                        zip(cur_resim, data_per_length_scale.values(), conv_sim)]
            cur_residuals_upsampled = [np.repeat(cur_res, data['signal_upsampling']) for cur_res, data in zip(cur_residuals, data_per_length_scale.values())]
            cur_pad_width = [(len(cur_residuals_upsampled[0])-len(cur_res)) for cur_res in cur_residuals_upsampled]
            cur_residuals_upsampled = [np.pad(cur_res, (pad // 2 + pad % 2, pad // 2)) for cur_res, pad in zip(cur_residuals_upsampled, cur_pad_width)]
            cur_residuals_abs_sum = np.sum(np.stack([np.abs(x) for ls_i, x in enumerate(cur_residuals_upsampled) if ls_i in length_scales_for_residuals]), axis=0)

            # Block telomeres and centromere
            cur_residuals_abs_sum[:telomere_block_start] = 0
            cur_residuals_abs_sum[telomere_block_end:] = 0
            cur_residuals_abs_sum[centromere_block_start:centromere_block_end] = 0

            # Block regions that are within CI
            within_ci = [np.logical_and(conv < data['signal_bounds'][1], conv > data['signal_bounds'][0])
                        for data, conv in zip(data_per_length_scale.values(), conv_sim)]
            ci_up = [np.repeat(c, data['signal_upsampling'])
                    for c, data in zip(within_ci, data_per_length_scale.values())]
            pad = [(len(ci_up[0]) - len(c)) for c in ci_up]
            ci_up = [np.pad(c, (p // 2 + p % 2, p // 2)) for c, p in zip(ci_up, pad)]
            cur_residuals_abs_sum[np.all(np.stack(ci_up), axis=0).astype(bool)] = 0

            cur_pos = np.argmax(cur_residuals_abs_sum) * segment_size_dict['small']
        else:  # mode == 'random'
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
            allow_pos_change=mode == 'top',
            up_down_order=up_down_order,
            blocked_distance_th=blocked_distance_th
        )
        optimized_selection_points = list(zip(*optimized_selection_points_per_cluster))
        optimized_selection_points_raw = copy_list_of_selection_points(optimized_selection_points)
        cur_pos = optimized_selection_points[0][0][0].pos

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
        # 'fitness' test statistic: mean optimized fitness over the four same-direction length
        # scales. Unlike added_events (which saturates via fitness/(fitness+1) and tracks event
        # density), this is monotone in fitness, so the resulting p-value tracks selection strength.
        fitness_stat = float(np.mean(loci_fitness[_DIR_SLOTS[cur_up_down]]))

        cur_results = {'added_events': added_events, 'fitness_stat': fitness_stat}
        cur_results = {**cur_results,
                       **{f'fit_{ls}': loci_fitness[_DIR_SLOTS[cur_up_down][i]]
                          for i, ls in enumerate(LENGTH_SCALE_NAMES)}
        }
            
        if save_all or (save_outliers is not None and added_events >= save_outliers):
            cur_results = {
                **cur_results,
                **{
                    'optimized_selection_points': optimized_selection_points,
                    'optimized_selection_points_raw': optimized_selection_points_raw,
                    'cur_resim': cur_resim
            }}
        results.append(cur_results)

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


def _observed_statistic(cur_loci, statistic):
    """Observed per-locus value of the chosen test statistic, matching the null construction."""
    if statistic == 'added_events':
        return cur_loci["added_events"].values
    if statistic == 'fitness':
        # same direction-matched mean-fitness as recorded in the null (clip <0 to 0)
        return _observed_fitness_per_ls(cur_loci).mean(axis=1)
    raise ValueError(f"unknown statistic {statistic!r} (expected 'added_events' or 'fitness')")


def get_actual_p_values_from_results(cur_loci, results, N_random, statistic='added_events'):
    """Empirical upper-tail p per locus: fraction of null resims whose statistic exceeds the
    observed value. `statistic` selects the quantity tested -- 'added_events' (SPICE default) or
    'fitness' (monotone in selection strength; see p_value_using_resim)."""
    key = 'fitness_stat' if statistic == 'fitness' else 'added_events'
    obs = _observed_statistic(cur_loci, statistic)
    null = np.array([x[key] for x in results])
    return (np.sum(obs[:, None] < null[None, :], axis=1) + 1) / (N_random + 1)


def get_actual_p_values_per_ls_from_results(cur_peaks, results, N_random):
    """Empirical upper-tail p per locus, per length scale, for the 'fitness' statistic: fraction of
    null resims whose per-length-scale fitness (`fit_<ls>` in `results`, from `p_value_using_resim`)
    exceeds the observed per-length-scale fitness. Returns an (n_loci, len(LENGTH_SCALE_NAMES)) array,
    columns ordered as `LENGTH_SCALE_NAMES`."""
    obs = _observed_fitness_per_ls(cur_peaks)  # (n_loci, n_ls)
    null = np.array([[x[f'fit_{ls}'] for ls in LENGTH_SCALE_NAMES] for x in results])  # (N_random, n_ls)
    return (np.sum(obs[:, None, :] < null[None, :, :], axis=1) + 1) / (N_random + 1)


def _observed_fitness_per_ls(cur_loci):
    """Observed per-locus, per-length-scale fitness (clip <0 to 0), matching the null construction
    (see `fit_<ls>` in `p_value_using_resim`), direction-matched via `up_down`."""
    ls_is = np.array([0, 2, 4, 6]) if cur_loci['up_down'].iloc[0] == 'up' else np.array([1, 3, 5, 7])
    cols = [f'fit_{s}' for s in ls_is]
    assert cur_loci[cols].min().min() >= 0
    return np.maximum(0.0, cur_loci[cols].to_numpy(float))  # (n_loci, len(LENGTH_SCALE_NAMES))
