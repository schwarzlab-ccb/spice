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


def _gpd_upper_tail_p(obs, null, N_random, tail_q=0.90, min_exc=15):
    """Sub-floor upper-tail p: empirical in the body, a Generalized-Pareto peaks-over-threshold fit
    above the `tail_q` quantile (Knijnenburg 2009). Extrapolates past the 1/(N+1) empirical floor
    without a normality assumption, so strong loci beyond the null's range get distinct p-values."""
    from scipy import stats
    null = np.asarray(null, float); obs = np.asarray(obs, float)
    p = (np.sum(obs[:, None] < null[None, :], axis=1) + 1) / (N_random + 1)   # empirical body/fallback
    u = np.quantile(null, tail_q)
    exc = null[null > u] - u
    if len(exc) >= min_exc:
        c, _, scale = stats.genpareto.fit(exc, floc=0)
        frac_above = np.mean(null > u)
        hi = obs > u
        p[hi] = np.clip(frac_above * stats.genpareto.sf(obs[hi] - u, c, loc=0, scale=scale),
                        np.finfo(float).tiny, 1.0)
    return p


def get_actual_p_values_from_results(cur_loci, results, N_random, statistic='added_events',
                                     method='empirical'):
    """Upper-tail p per locus vs the resim null. `statistic` selects the tested quantity --
    'added_events' (SPICE default) or 'fitness' (monotone in selection strength). `method` selects
    the tail: 'empirical' (count; floors at 1/(N+1)) or 'gpd' (peaks-over-threshold; sub-floor)."""
    key = 'fitness_stat' if statistic == 'fitness' else 'added_events'
    obs = _observed_statistic(cur_loci, statistic)
    null = np.array([x[key] for x in results])
    if method == 'gpd':
        return _gpd_upper_tail_p(obs, null, N_random)
    if method == 'empirical':
        return (np.sum(obs[:, None] < null[None, :], axis=1) + 1) / (N_random + 1)
    raise ValueError(f"unknown method {method!r} (expected 'empirical' or 'gpd')")


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
