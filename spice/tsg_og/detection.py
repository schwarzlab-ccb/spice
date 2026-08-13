import os
import itertools
import logging
from copy import copy, deepcopy
from functools import reduce
from joblib import Parallel, delayed
import numpy as np

from spice import data_loaders, directories, config
from spice.utils import open_pickle, CALC_NEW
from spice.logging import log_debug, get_logger
from spice.random_state import derive_seed, np_rng, seed_task
from spice.tsg_og.simulation import (
    SelectionPoints, create_convolution_kernel, create_centromere_values, Locus,
    convolution_simulation, combine_selection_points, copy_list_of_selection_points,
    convolution_simulation_per_ls, create_height_multiplier)
from spice.length_scales import DEFAULT_SEGMENT_SIZE_DICT, DEFAULT_LENGTH_SCALE_BOUNDARIES
from spice.segmentation import create_events_in_segmentation
from spice.tsg_og.signal_bootstrap import get_signal_bootstrap_bounds
from spice.tsg_og.event_rate_per_loci import calc_total_events_per_loci


logger = get_logger('tsg_og_detection')
DATA_LOADERS_DIR = os.path.join(directories['results_dir'], 'data_loaders')
CHROMS = ['chr' + str(x) for x in range(1, 23)] + ['chrX', 'chrY']
CENTROMERES = data_loaders.load_centromeres()
CENTROMERES_OBSERVED = data_loaders.load_centromeres(extended=False, observed=True)
TELOMERES_OBSERVED = data_loaders.load_telomeres_observed()
PLATEAU_WIDTH = 10e5
CHROM_LENS = data_loaders.load_chrom_lengths()


def calc_mse_loss(data_per_length_scale, cur_conv_simulated):
    return sum([np.mean((data['signals'][data['non_centromere_index']] - generated_signal[data['non_centromere_index']]) ** 2) / data['cur_loss_norm']
                for data, generated_signal in zip(data_per_length_scale.values(), cur_conv_simulated)])


def calc_within_ci_bootstrap(data_per_length_scale, simulated_conv, exclude_zero_signal=False):
    if exclude_zero_signal:       
        cur_within_ci = [np.mean(
            np.logical_or(
                data['signals'][data['non_centromere_index']] == 0,
                np.logical_and(cur_conv[data['non_centromere_index']] < data['signal_bounds'][1][data['non_centromere_index']],
                               cur_conv[data['non_centromere_index']] > data['signal_bounds'][0][data['non_centromere_index']])))
            for data, cur_conv in zip(data_per_length_scale.values(), simulated_conv)]
    else:
        cur_within_ci = [np.mean(
            np.logical_and(cur_conv[data['non_centromere_index']] < data['signal_bounds'][1][data['non_centromere_index']],
                           cur_conv[data['non_centromere_index']] > data['signal_bounds'][0][data['non_centromere_index']]))
            for data, cur_conv in zip(data_per_length_scale.values(), simulated_conv)]
    return cur_within_ci


def get_cur_widths(final_events_df, cur_chrom, cur_length_scale=None, cur_type="gain",
                   length_scale_boundaries=DEFAULT_LENGTH_SCALE_BOUNDARIES):

    cur_length_scale_border = length_scale_boundaries[cur_length_scale]
    cur_widths = final_events_df.query('pos == "internal" and type == @cur_type and chrom == @cur_chrom and width > @cur_length_scale_border[0] and width <= @cur_length_scale_border[1]')['width'].values

    return cur_widths


@CALC_NEW()
def collect_data_per_length_scale(
        final_events_df,
        cur_chrom,
        segment_size_dict=DEFAULT_SEGMENT_SIZE_DICT,
        length_scale_boundaries=DEFAULT_LENGTH_SCALE_BOUNDARIES,
        filter_plateaus=True,
        loci_results_dir=None,
        assert_non_empty=True,
        N_bootstrap=1_000,
        N_kernel=100_000
        ):
    log_debug(logger, f'Collecting data for all length scales for {cur_chrom}') 

    plateau_events = final_events_df.query('plateau != "neither_left_nor_right"').copy().reset_index(drop=True)
    if filter_plateaus:
        final_events_df = final_events_df.query('plateau == "neither_left_nor_right"').copy().reset_index(drop=True)

    signal_bootstrap_bounds = get_signal_bootstrap_bounds(
        cur_chrom, loci_results_dir, N_bootstrap=N_bootstrap)

    data_per_length_scale = {}
    for ls_i, (cur_length_scale, cur_type) in enumerate(itertools.product(['small', 'mid1', 'mid2', 'large'], ['gain', 'loss'])):
        log_debug(logger, f'Processing length scale {cur_length_scale} and type {cur_type} for chromosome {cur_chrom}')
        cur_widths = get_cur_widths(
            final_events_df, cur_chrom, cur_length_scale=cur_length_scale, cur_type=cur_type)
        if len(cur_widths) == 0:
            if assert_non_empty:
                raise ValueError(f'No events found for {cur_chrom}, {cur_length_scale}, {cur_type}')
            else:
                logger.warning(f'No events found for {cur_chrom}, {cur_length_scale}, {cur_type}, skipping')
                data_per_length_scale[(cur_length_scale, cur_type)] = None
                continue

        cur_length_scale_border = length_scale_boundaries[cur_length_scale]
        cur_events = final_events_df.query('pos == "internal" and type == @cur_type and chrom == @cur_chrom and width > @cur_length_scale_border[0] and width <= @cur_length_scale_border[1]').copy()
    
        log_debug(logger, 'Create events in segmentation')
        loci_width = int(np.round(np.mean(cur_widths) / segment_size_dict[cur_length_scale], 0))
        signals = (create_events_in_segmentation(
            cur_events, bin_df=segment_size_dict[cur_length_scale], skip_tqdm=True)
            .loc[cur_chrom].sum(axis=1).values)
        
        log_debug(logger, 'Create convolution kernel and height_multiplier')
        kernel = create_convolution_kernel(
            cur_widths, segment_size_dict[cur_length_scale], N_kernel, which='locus')
        height_multiplier = create_height_multiplier(
            cur_widths, cur_chrom, cur_length_scale, cur_type, loci_width, segment_size_dict=segment_size_dict,
            n_widths=N_kernel, n_sims=100)
        centromere_values = create_centromere_values(
            cur_chrom, cur_length_scale, cur_widths, segment_size_dict[cur_length_scale])
        
        # Define centromere region and non-centromere index
        centro_start, centro_end = CENTROMERES_OBSERVED.loc[cur_chrom, cur_length_scale].values
        centro_start_i, centro_end_i = int(centro_start // segment_size_dict[cur_length_scale]), int(centro_end // segment_size_dict[cur_length_scale])
        non_centromere_index = np.setdiff1d(np.arange(len(signals)), np.arange(centro_start_i, centro_end_i))
        
        cur_loss_norm = np.mean(signals[non_centromere_index])
        
        # Use tuple (cur_length_scale, cur_type) as dictionary key
        data_per_length_scale[(cur_length_scale, cur_type)] = {
            'chrom': cur_chrom,
            'signals': signals,
            'cur_widths': cur_widths,
            'loci_width': loci_width,
            'length_scale': cur_length_scale,
            'type': cur_type,
            'length_scale_i': ls_i,
            'non_centromere_index': non_centromere_index,
            'cur_loss_norm': cur_loss_norm,
            'kernel': kernel,
            'height_multiplier': height_multiplier,
            'centromere_values': centromere_values,
            'signal_bounds': signal_bootstrap_bounds[ls_i],
        }

    for key in data_per_length_scale.keys():
        if data_per_length_scale[key] is None or data_per_length_scale[('small', 'gain')] is None:
            continue
        data_per_length_scale[key]['signal_upsampling'] = (
            len(data_per_length_scale[('small', 'gain')]['signals']) / len(data_per_length_scale[key]['signals'])
        )

    return data_per_length_scale


@CALC_NEW()
def combine_signal_and_bootstrap_across_ls(
        data_per_length_scale, cur_chrom, segment_size_dict=DEFAULT_SEGMENT_SIZE_DICT):

    all_combined_signal = dict()
    all_total_bootstrap = dict()
    for cur_type in ['gain', 'loss']:

        chrom_len = data_loaders.load_chrom_lengths().loc[cur_chrom]
        len_small_signal = len(data_per_length_scale[('small', cur_type)]['signals'])
        cur_signal = [data_per_length_scale[(cur_ls, cur_type)]['signals']
                    for cur_ls in ['small', 'mid1', 'mid2', 'large']]
        cur_signal_interp = [np.interp(np.linspace(0, chrom_len, len_small_signal),
                    np.arange(len(sig)) * segment_size_dict[cur_ls], sig)
                    for sig, cur_ls in zip(cur_signal, ['small', 'mid1', 'mid2', 'large'])]
        combined_signal = np.sum(np.stack(cur_signal_interp), axis=0)

        cur_bootstrap_low = [data_per_length_scale[(cur_ls, cur_type)]['signal_bounds'][0]
                    for cur_ls in ['small', 'mid1', 'mid2', 'large']]
        cur_bootstrap_high = [data_per_length_scale[(cur_ls, cur_type)]['signal_bounds'][1]
                    for cur_ls in ['small', 'mid1', 'mid2', 'large']]
        cur_bootstrap_low_interp = [np.interp(np.linspace(0, chrom_len, len_small_signal),
                    np.arange(len(sig)) * segment_size_dict[cur_ls], sig)
                    for sig, cur_ls in zip(cur_bootstrap_low, ['small', 'mid1', 'mid2', 'large'])]
        cur_bootstrap_high_interp = [np.interp(np.linspace(0, chrom_len, len_small_signal),
                    np.arange(len(sig)) * segment_size_dict[cur_ls], sig)
                    for sig, cur_ls in zip(cur_bootstrap_high, ['small', 'mid1', 'mid2', 'large'])]
        cur_bootstrap_low_interp_diff = [sig - bs for sig, bs in zip(cur_signal_interp, cur_bootstrap_low_interp)]
        cur_bootstrap_high_interp_diff = [bs - sig for sig, bs in zip(cur_signal_interp, cur_bootstrap_high_interp)]
        total_bootstrap_low_diff = np.sum(np.stack(cur_bootstrap_low_interp_diff), axis=0)
        total_bootstrap_high_diff = np.sum(np.stack(cur_bootstrap_high_interp_diff), axis=0)

        total_bootstrap = (combined_signal - total_bootstrap_low_diff,
                           combined_signal + total_bootstrap_high_diff)
        
        all_combined_signal[cur_type] = combined_signal
        all_total_bootstrap[cur_type] = total_bootstrap

    return all_combined_signal, all_total_bootstrap


def _optimize_selection_points(N_iterations, best_selection_points_per_cluster, data_per_length_scale,
                              cur_chrom, blocked_cluster_positions=None, blocked_distance_th=2e5, max_pos_change=3e5,
                              segment_size_dict=DEFAULT_SEGMENT_SIZE_DICT, best_loss=float('inf'), final_iteration=False,
                              max_fitness=1_000, up_down_order=None, block_centromere_pos=True, allow_pos_change=True,
                              N_iterations_base=0, loci_to_optimize=None, max_deviation=0.001, show_progress=False,
                              allowed_fitness_change=None, ls_to_optimize=None, legacy_height_multiplier=False):
    """
    Optimize selection points using simulated annealing.

    Parameters:
    - N_iterations: int, number of iterations
    - best_selection_points_per_cluster: list, best selection points per cluster
    - data_per_length_scale: dictionary, data per length scale
    - cur_chrom: str, current chromosome
    - segment_size_dict: dict, segment size
    - CHROM_LENS: DataFrame, chromosome lengths

    Returns:
    - best_selection_points_per_cluster: list, optimized selection points per cluster
    - best_loss: float, best loss value
    - all_losses: list, all loss values
    """

    cur_selection_points_per_cluster = copy_list_of_selection_points(best_selection_points_per_cluster)
    if allowed_fitness_change is None:
        allowed_fitness_change = np.ones((8, len(best_selection_points_per_cluster)))
    assert allowed_fitness_change.shape == (8, len(best_selection_points_per_cluster)), (allowed_fitness_change.shape, len(best_selection_points_per_cluster))
    if ls_to_optimize is not None:
        allowed_fitness_change[np.setdiff1d(np.arange(8), ls_to_optimize), :] = False
    if not isinstance(max_fitness, (list, np.ndarray)):
        max_fitness = np.array([max_fitness] * 8)
    else:
        max_fitness = np.array(max_fitness)
        assert len(max_fitness) == 8, len(max_fitness)
    if up_down_order is not None and len(up_down_order) > 0:
        assert len(up_down_order) == len(best_selection_points_per_cluster), (len(up_down_order), len(best_selection_points_per_cluster))

    all_losses = []

    tel_cen_distance_th = max(blocked_distance_th, segment_size_dict['large'])
    telomere_block_start = (TELOMERES_OBSERVED.loc[cur_chrom, 'small']['chrom_start'] + tel_cen_distance_th) 
    telomere_block_end = (TELOMERES_OBSERVED.loc[cur_chrom, 'small']['chrom_end'] - tel_cen_distance_th) 
    centromere_block_start = (CENTROMERES_OBSERVED.loc[cur_chrom, 'small']['centro_start'] - tel_cen_distance_th) 
    centromere_block_end = (CENTROMERES_OBSERVED.loc[cur_chrom, 'small']['centro_end'] + tel_cen_distance_th)

    generated_signals = convolution_simulation_per_ls(
        cur_chrom, data_per_length_scale, list(zip(*best_selection_points_per_cluster)),
        segment_size_dict=DEFAULT_SEGMENT_SIZE_DICT, legacy_height_multiplier=legacy_height_multiplier)
    if loci_to_optimize is None:
        loci_to_optimize = range(len(best_selection_points_per_cluster))
    assert len(loci_to_optimize) > 0, 'No loci to optimize'
    assert N_iterations == 0 or N_iterations >= N_iterations_base, f'N_iterations ({N_iterations}) should be greater than N_iterations_base ({N_iterations_base})'

    for iteration in range(N_iterations):
        # Choose a cluster index to modify (the first N_iteration_base ones only optimize last one)
        cur_cluster_i = (len(best_selection_points_per_cluster) - 1 if (iteration < N_iterations_base and not final_iteration)
                         else np_rng().choice(loci_to_optimize))
        cur_cluster_pos = best_selection_points_per_cluster[cur_cluster_i][0][0].pos
        cur_cluster_fitness = [x[0].fitness for x in best_selection_points_per_cluster[cur_cluster_i]]

        # Randomly adjust position (10% of the time) or fitness (90% of the time)
        pos_change = max_pos_change * np_rng().uniform(-1, 1) if np_rng().random_sample() < (allow_pos_change * 0.1) else 0
        new_cluster_pos = cur_cluster_pos + pos_change

        # Check proximity to blocked positions
        near_blocked = (
            blocked_cluster_positions is not None and
            len(blocked_cluster_positions) > 0 and
            np.min(np.abs(blocked_cluster_positions - new_cluster_pos)) < blocked_distance_th
        )
        # Check proximity to telomeres and centromere
        near_telomere = (new_cluster_pos < telomere_block_start or
                         new_cluster_pos > telomere_block_end)
        near_centromere = (block_centromere_pos and
                           new_cluster_pos > centromere_block_start and
                           new_cluster_pos < centromere_block_end)
        # Invalidate position change if near any blocked area
        if pos_change != 0 and (near_blocked or near_telomere or near_centromere):
            continue

        fitness_change = np.zeros(8)
        if pos_change == 0:
            if not allowed_fitness_change[:, cur_cluster_i].any():
                continue
            # 50% chance to adjust fitness based on residuals or randomly
            if generated_signals[0] is not None and np_rng().random_sample() < 0.5:
                fitness_diff = np.array([(data['signals'] - generated_signal)[int(cur_cluster_pos // segment_size_dict[data['length_scale']])] / 
                                        (data['signals'][int(cur_cluster_pos // segment_size_dict[data['length_scale']])] + 1e-10)
                                         for data, generated_signal in
                                         zip(data_per_length_scale.values(), generated_signals)])
            else:
                fitness_diff = np.ones(8)

            cur_fitness_ls_i = np_rng().choice(np.where(allowed_fitness_change[:, cur_cluster_i])[0])
            fitness_diff = np.array([x if i == cur_fitness_ls_i else 0 for i, x in enumerate(fitness_diff)])
            fitness_change = np.maximum(cur_cluster_fitness, 1) * fitness_diff * np_rng().uniform()
            new_fitness_values = np.minimum(cur_cluster_fitness + fitness_change, max_fitness)
            if up_down_order is not None and len(up_down_order) > 0:
                # up: pos gains and neg losses
                if up_down_order[cur_cluster_i]:
                    new_fitness_values[::2] = np.clip(new_fitness_values[::2], 0, np.inf)
                    new_fitness_values[1::2] = np.clip(new_fitness_values[1::2], -np.inf, 0)
                # down: pos losses and neg gains
                else:
                    new_fitness_values[::2] = np.clip(new_fitness_values[::2], -np.inf, 0)
                    new_fitness_values[1::2] = np.clip(new_fitness_values[1::2], 0, np.inf)

            if any(np.isnan(fitness_change)):
                raise ValueError('Nan in fitness change')
        else:
            new_fitness_values = cur_cluster_fitness

        # Save these in case the proposal is rejected
        old_cluster = copy(cur_selection_points_per_cluster[cur_cluster_i])
        old_generated_signals = copy(generated_signals)
        
        new_cluster = [SelectionPoints(loci=[(new_cluster_pos, cur_fit)]) for cur_fit in new_fitness_values]
        cur_selection_points_per_cluster[cur_cluster_i] = new_cluster
        cur_selection_points = list(zip(*cur_selection_points_per_cluster))

        if iteration == 0 or pos_change != 0:
            changed_ls = np.arange(8)
        else:
            changed_ls = np.where(fitness_change != 0)[0]

        for ls in changed_ls:
            data = list(data_per_length_scale.values())[ls]
            cur_sp = cur_selection_points[ls]
            generated_signals[ls] = convolution_simulation(
                cur_chrom=cur_chrom, 
                selection_points=combine_selection_points(cur_sp), 
                cur_widths=data['cur_widths'], 
                kernel=data['kernel'],
                kernel_edge=data.get('kernel_edge', None),
                cur_length_scale=data['length_scale'], 
                segment_size=segment_size_dict[data['length_scale']], 
                centromere_values=data['centromere_values'],
                cur_signal=data['signals'],
                legacy_height_multiplier=legacy_height_multiplier,
                height_multiplier=data.get('height_multiplier', None)
                )
        
        cur_loss = calc_mse_loss(data_per_length_scale, generated_signals)

        if calc_acceptance(cur_loss, best_loss, iteration,
                           N_iterations_base if (iteration < N_iterations_base and not final_iteration) else N_iterations - N_iterations_base,
                           T_schedule='min_max', max_deviation=max_deviation, min_deviation=0.00001):
            best_selection_points_per_cluster = copy(cur_selection_points_per_cluster)
            best_loss = cur_loss
            all_losses.append(cur_loss)
        else:
            cur_selection_points_per_cluster[cur_cluster_i] = old_cluster
            for ls in changed_ls:
                generated_signals[ls] = old_generated_signals[ls] 

    return best_selection_points_per_cluster, best_loss, all_losses


@CALC_NEW()
def detect_tsgs_ogs_for_all_length_scales(
    cur_chrom,
    data_per_length_scale=None,
    final_events_df=None,
    segment_size_dict=DEFAULT_SEGMENT_SIZE_DICT,
    N_iterations_base=3_000,
    max_N_iterations=20_000,
    final_N_iterations=1_000_000,
    N_loci=50,
    fixed_selection_points=None,
    max_fitness=1_000,
    length_scales_for_residuals=None,
    force_up_down=False,
    blocked_distance_th=2e5):
    """Analyzes a chromosome by performing simulations and optimizations."""
    log_debug(logger, f"Detecting TSG / OG for {cur_chrom}. ALL length scales and gains/losses combined")

    if segment_size_dict is None:
        segment_size_dict = {'small': 1e5, 'mid1': 1e5, 'mid2': 1e5, 'large': 1e5}
    assert len(segment_size_dict) == 4

    assert all([x['chrom']==cur_chrom for x in data_per_length_scale.values()]), f'Wrong data_per_length_scale for current chrom {cur_chrom}'

    blocked_distance_th_bin = int(blocked_distance_th / segment_size_dict['small'])
    tel_cen_distance_th = max(blocked_distance_th, segment_size_dict['large'])
    telomere_block_start = int((TELOMERES_OBSERVED.loc[cur_chrom, 'small']['chrom_start'] + tel_cen_distance_th) / segment_size_dict['small'])
    telomere_block_end = int((TELOMERES_OBSERVED.loc[cur_chrom, 'small']['chrom_end'] - tel_cen_distance_th) / segment_size_dict['small'])
    centromere_block_start = int((CENTROMERES_OBSERVED.loc[cur_chrom, 'small']['centro_start'] - tel_cen_distance_th) / segment_size_dict['small'])
    centromere_block_end = int((CENTROMERES_OBSERVED.loc[cur_chrom, 'small']['centro_end'] + tel_cen_distance_th) / segment_size_dict['small'])   

    if length_scales_for_residuals is None:
        length_scales_for_residuals = np.arange(8).astype(int)
    assert len(length_scales_for_residuals)%2 == 0, 'length_scales_for_residuals should be even'
    if fixed_selection_points is None:
        start_from_scratch = True
        fixed_selection_points = 8 * [[SelectionPoints()]]
        up_down_order = []
    else:
        start_from_scratch = False
        up_down_order = list((np.stack([[x[0].fitness > 0 for x in y] for y in fixed_selection_points[::2]]) > 0).any(axis=0))
        
    blocked_cluster_positions = []
    total_losses, total_selection_points = [], []

    best_loss = float('inf')
    new_pos = None
    start_gene_iteration = 0 if start_from_scratch else len(fixed_selection_points[0])
    if start_gene_iteration != 0:
        log_debug(logger, f'Starting from iteration {start_gene_iteration} of {N_loci}')
    
    for gene_iteration in range(start_gene_iteration, N_loci+1):
        if gene_iteration == 0:         
            log_debug(logger, f"Starting iteration {gene_iteration} of {N_loci}")
        else:
            log_debug(logger, f"Starting iteration {gene_iteration} of {N_loci} (best loss = {best_loss:.2e}, pos of new locus = {new_pos*segment_size_dict['small']:.8e} {'' if len(up_down_order) == 0 else 'UP' if up_down_order[-1] else 'DOWN'})")
        best_selection_points = deepcopy(fixed_selection_points)
        best_selection_points_per_cluster = list(zip(*best_selection_points))
            
        all_losses = []
        N_iterations = min(max_N_iterations, int(N_iterations_base * np.sqrt(gene_iteration + 1))) if gene_iteration > 0 else 0
        if gene_iteration == N_loci:
            N_iterations = final_N_iterations
            log_debug(logger, f'Starting final optimization round, with {N_iterations} iterations')
        # Run optimization loop
        best_selection_points_per_cluster, best_loss, all_losses = _optimize_selection_points(
            N_iterations, best_selection_points_per_cluster, data_per_length_scale, cur_chrom,
            best_loss=best_loss, max_fitness=max_fitness, segment_size_dict=segment_size_dict, N_iterations_base=N_iterations_base,
            final_iteration=(gene_iteration == N_loci), blocked_cluster_positions=blocked_cluster_positions[:-1],
            up_down_order=up_down_order if force_up_down else None, blocked_distance_th=blocked_distance_th)
        
        best_selection_points = list(zip(*best_selection_points_per_cluster))
        fixed_selection_points = copy_list_of_selection_points(best_selection_points)
        
        # Simulate with the best selection points and calculate residuals to find position of next locus
        final_simulated = convolution_simulation_per_ls(cur_chrom, data_per_length_scale, fixed_selection_points,
                                segment_size_dict=DEFAULT_SEGMENT_SIZE_DICT)
        best_loss = calc_mse_loss(data_per_length_scale, final_simulated)
        cur_residuals = [(data['signals'] - generated_signal) / data['cur_loss_norm']
                    for data, generated_signal in 
                    zip(data_per_length_scale.values(), final_simulated)]
        cur_residuals_upsampled = [np.repeat(cur_res, data['signal_upsampling']) for cur_res, data in zip(cur_residuals, data_per_length_scale.values())]
        cur_pad_width = [(len(cur_residuals_upsampled[0])-len(cur_res)) for cur_res in cur_residuals_upsampled]
        cur_residuals_upsampled = [np.pad(cur_res, (pad // 2 + pad % 2, pad // 2)) for cur_res, pad in zip(cur_residuals_upsampled, cur_pad_width)]       
        cur_residuals_abs_sum = np.sum(np.stack([np.abs(x) for ls_i, x in enumerate(cur_residuals_upsampled) if ls_i in length_scales_for_residuals]), axis=0)

        # Block telomeres and centromere
        cur_residuals_abs_sum[:telomere_block_start] = 0
        cur_residuals_abs_sum[telomere_block_end:] = 0
        cur_residuals_abs_sum[centromere_block_start:centromere_block_end] = 0        

        # Set residual sum to zero around existing loci
        if gene_iteration > 0 and blocked_distance_th_bin > 0:
            for cluster in fixed_selection_points[0]:
                cur_cluster_pos = int(np.round(cluster[0].pos / segment_size_dict['small'], 0))
                cur_residuals_abs_sum[max(0, cur_cluster_pos-blocked_distance_th_bin):min(len(cur_residuals_abs_sum), cur_cluster_pos+blocked_distance_th_bin+1)] = 0

        # If region is within 95% CI of the bootstrap signal, set residuals to zero
        within_ci_all_ls = [
            np.logical_and(cur_conv < data['signal_bounds'][1], cur_conv > data['signal_bounds'][0])
            for data, cur_conv in zip(data_per_length_scale.values(), final_simulated)
        ]    
        within_ci_all_ls_upsampled = [np.repeat(cur_res, data['signal_upsampling']) for cur_res, data in zip(within_ci_all_ls, data_per_length_scale.values())]
        cur_pad_width = [(len(within_ci_all_ls_upsampled[0])-len(cur_res)) for cur_res in within_ci_all_ls_upsampled]
        within_ci_all_ls_upsampled = [np.pad(cur_res, (pad // 2 + pad % 2, pad // 2)) for cur_res, pad in zip(within_ci_all_ls_upsampled, cur_pad_width)]
        within_ci_all_ls_final = np.all(np.stack(within_ci_all_ls_upsampled), axis=0).astype(bool)
        cur_residuals_abs_sum[within_ci_all_ls_final] = 0

        if cur_residuals_abs_sum.max() == 0:
            log_debug(logger, f'No residuals left to optimize, stopping at gene iteration {gene_iteration}')
            break
        new_pos = np.argmax(cur_residuals_abs_sum)

        # If the more gains are higher than losses in the majority of LS then the locus is up, otherwise it's down
        # don't use np.abs here so neg selection can be taken into account
        # Check if the majority of LS is up or down and only use the abs values in case of a tie (otherwise the large LS might overpower this)
        cur_gains = np.stack([x for ls_i, x in enumerate(cur_residuals_upsampled) if ls_i in length_scales_for_residuals])[:, new_pos][::2]
        cur_losses = np.stack([x for ls_i, x in enumerate(cur_residuals_upsampled) if ls_i in length_scales_for_residuals])[:, new_pos][1::2]
        is_up = bool((np.mean(cur_gains > cur_losses) > 0.5) or
                     (np.mean(cur_gains == cur_losses) and (np.sum(cur_gains) > np.sum(cur_losses))))
        up_down_order.append(is_up)

        new_selection_points = [SelectionPoints(loci=[[new_pos*segment_size_dict['small'], 0]]) for _ in range(8)]

        fixed_selection_points = [[x] for x in new_selection_points] if gene_iteration == 0 \
            else [list(x) + [y] for x, y in zip(fixed_selection_points, new_selection_points)]
        blocked_cluster_positions = np.array([x[0].pos for x in fixed_selection_points[0]])

        # If max fitness is reached, halve all fitness values in this length scale
        max_fitnes_ls = np.where([any([locus[0].fitness >= max_fitness for locus in ls_loci]) for ls_loci in fixed_selection_points])[0]
        if len(max_fitnes_ls) > 0:
            fixed_selection_points = [list(x) for x in fixed_selection_points]
            for cur_max_fitness_ls in max_fitnes_ls:
                for cluster_i in range(len(fixed_selection_points[0])):
                    fixed_selection_points[cur_max_fitness_ls][cluster_i] = SelectionPoints(loci=[[
                        fixed_selection_points[cur_max_fitness_ls][cluster_i][0].pos, fixed_selection_points[cur_max_fitness_ls][cluster_i][0].fitness / 2]])
            max_fitness_reached = np.where(np.stack([[x[0].fitness == max_fitness for x in y] for y in fixed_selection_points]))
            log_debug(logger, f'Max fitness reached at gene iteration {gene_iteration} (pos = {new_pos*segment_size_dict["small"]}), aborting optimization ' +
                                f'for length scales {max_fitness_reached[0]} and clusters {max_fitness_reached[1]} ' +
                                f'--> halfed the fitness in the length scales {max_fitnes_ls}')

        total_losses.append(all_losses)
        total_selection_points.append(best_selection_points_per_cluster)
    
    log_debug(logger, f"Detection for chromosome {cur_chrom} completed and saved.")
    
    return best_selection_points, total_losses, total_selection_points


@CALC_NEW()
def flip_up_down_assignment(
    cur_chrom,
    final_selection_points,
    data_per_length_scale=None,
    final_events_df=None,
    segment_size_dict=DEFAULT_SEGMENT_SIZE_DICT,
    n_neighbors=10,
    N_iterations=11_000,
    N_iterations_single=1_000,
    filename=None,
    max_pos_change=1e5):
    log_debug(logger, 'Flipping up/down assignment of loci')

    assert all([x['chrom']==cur_chrom for x in data_per_length_scale.values()]), f'Wrong data_per_length_scale for current chrom {cur_chrom}'

    final_selection_points = copy_list_of_selection_points(final_selection_points)
    simulated_conv = convolution_simulation_per_ls(cur_chrom, data_per_length_scale, final_selection_points)

    # STEP 1: Find suitable candidate loci
    conv_above_signal = [np.clip(cur_conv - data['signal_bounds'][1], 0, None)
                for cur_conv, data in zip(simulated_conv, data_per_length_scale.values())]
    conv_below_signal = [np.clip(data['signal_bounds'][0] - cur_conv, 0, None)
                for cur_conv, data in zip(simulated_conv, data_per_length_scale.values())]

    up_down_deviation_neighborhood = []
    for cluster_i in range(len(final_selection_points[0])):

        is_up = any([final_selection_points[j][cluster_i][0].fitness > 0 for j in range(0, 8, 2)])
        cur_pos = final_selection_points[0][cluster_i][0].pos
        cur_pos_bin = [int(cur_pos / segment_size_dict[data['length_scale']]) for data in data_per_length_scale.values()]
        cur_left_i = [max(0, p - int(data['loci_width'] / 4)) for data, p in zip(data_per_length_scale.values(), cur_pos_bin)]
        cur_right_i = [min(len(data['signals']), p + int(data['loci_width'] / 4)) for data, p in zip(data_per_length_scale.values(), cur_pos_bin)]

        max_above = [max(a[l:r]) for a, l, r in zip(conv_below_signal, cur_left_i, cur_right_i)]
        max_below = [max(b[l:r]) for b, l, r in zip(conv_above_signal, cur_left_i, cur_right_i)]
        if is_up:
            cur_up_down_deviation_neighborhood = [max_below[i] if i % 2 == 0 else max_above[i] for i in range(8)]
        else:
            cur_up_down_deviation_neighborhood = [max_above[i] if i % 2 == 0 else max_below[i] for i in range(8)]

        up_down_deviation_neighborhood.append(cur_up_down_deviation_neighborhood)
    up_down_deviation_neighborhood = np.stack(up_down_deviation_neighborhood)
    
    total_up_down_deviation_neighborhood = up_down_deviation_neighborhood[:, :4].max(axis=1)
    total_up_down_deviation_neighborhood[total_up_down_deviation_neighborhood < 5] = 0

    candidate_loci = np.argsort(total_up_down_deviation_neighborhood)[::-1][:np.sum(total_up_down_deviation_neighborhood!=0)]

    log_debug(logger, f'Found {len(candidate_loci)} candidate loci to flip up/down assignment')

    # STEP 2: Flip the candidate loci and see if this improves the loss
    simulated_conv = convolution_simulation_per_ls(cur_chrom, data_per_length_scale, final_selection_points,
                            segment_size_dict=DEFAULT_SEGMENT_SIZE_DICT)
    base_loss = calc_mse_loss(data_per_length_scale, simulated_conv)
    up_down_order_raw = [any([final_selection_points[j][cluster_j][0].fitness > 0 for j in range(0, 8, 2)])
                    for cluster_j in range(len(final_selection_points[0]))]
    log_debug(logger, f'Starting up/down flipping. Base loss: {base_loss:.4e}')

    n_flipped = 0
    for cluster_i in candidate_loci:

        cur_selection_points = copy_list_of_selection_points(final_selection_points)
        cur_selection_points = [list(x) for x in cur_selection_points]
        cur_pos = final_selection_points[0][cluster_i][0].pos
        neighborhood_clusters = np.argsort([np.abs(x[0].pos-cur_pos) for x in final_selection_points[0]])[:(n_neighbors+1)]
        assert cluster_i in neighborhood_clusters
        for i in range(8):
            cur_selection_points[i][cluster_i] = SelectionPoints(loci=[[cur_pos, 0]])
        up_down_order = deepcopy(up_down_order_raw)
        up_down_order[cluster_i] = not up_down_order[cluster_i]
        cur_selection_points_per_cluster = list(zip(*cur_selection_points))

        simulated_conv_empty = convolution_simulation_per_ls(cur_chrom, data_per_length_scale, cur_selection_points,
                            segment_size_dict=DEFAULT_SEGMENT_SIZE_DICT)
        empty_loss = calc_mse_loss(data_per_length_scale, simulated_conv_empty)

        cur_selection_points_per_cluster, optim_best_loss, _ = _optimize_selection_points(
                    N_iterations, cur_selection_points_per_cluster, data_per_length_scale, cur_chrom,
                    best_loss=empty_loss, segment_size_dict=segment_size_dict, final_iteration=False, show_progress=False,
                    loci_to_optimize=neighborhood_clusters, N_iterations_base=0, max_deviation=0.0001,
                    max_pos_change=max_pos_change, blocked_cluster_positions=np.array([x[0].pos for i, x in enumerate(final_selection_points[0]) if i != cluster_i]),
                    up_down_order=up_down_order, blocked_distance_th=2e5)

        log_debug(logger, f'Locus {cluster_i}: {"flip" if optim_best_loss < base_loss else "keep"} up/down assignment. cur loss: {optim_best_loss:.4e}')
        if optim_best_loss < base_loss:
            # optimize just the locus again
            cur_selection_points_per_cluster, optim_best_loss_, _ = _optimize_selection_points(
                        N_iterations_single, cur_selection_points_per_cluster, data_per_length_scale, cur_chrom,
                        best_loss=empty_loss, segment_size_dict=segment_size_dict, final_iteration=False, show_progress=False,
                        loci_to_optimize=[cluster_i], N_iterations_base=0, max_deviation=0.0001,
                        blocked_cluster_positions=np.array([x[0].pos for i, x in enumerate(final_selection_points[0]) if i != cluster_i]),
                        up_down_order=up_down_order, blocked_distance_th=2e5)
            if optim_best_loss_ < base_loss:
                n_flipped += 1
                up_down_order_raw[cluster_i] = not up_down_order_raw[cluster_i]
                final_selection_points = list(zip(*cur_selection_points_per_cluster))
                base_loss = optim_best_loss_

    log_debug(logger, f'Finished! Flipped {n_flipped} loci out of {len(candidate_loci)} candidate loci. Final loss {base_loss:.4e}')

    return final_selection_points


@CALC_NEW()
def rank_loci(
    cur_chrom,
    best_selection_points,
    final_events_df=None,
    force_up_down=True,
    data_per_length_scale=None,
    show_progress=False,
    log_progress=False,
    segment_size_dict=DEFAULT_SEGMENT_SIZE_DICT,
    max_n_clusters=None,
    N_iterations=1_000,
    max_fitness=1_000,
    optimized_locus_iterations=None,
    n_cores=None):

    log_debug(logger, f'Using {n_cores} cores for parallelization' if n_cores is not None else 
                'Using single core for optimization')

    assert len(best_selection_points) == 8
    if max_n_clusters is None:
        max_n_clusters = len(best_selection_points[0])
    best_selection_points = [list(x) for x in copy_list_of_selection_points(best_selection_points)]
    N_clusters = len(best_selection_points[0])

    original_up_down_order = (np.stack([[x[0].fitness > 0 for x in y] for y in best_selection_points[::2]]) > 0).any(axis=0)
    # allowed_fitness_change = np.stack([[x[0].fitness != 0 for x in y] for y in best_selection_points])
    fixed_up_down_order = []

    assert all([x['chrom']==cur_chrom for x in data_per_length_scale.values()]), f'Wrong data_per_length_scale for current chrom {cur_chrom}'

    if optimized_locus_iterations is None:
        optimized_locus_iterations = []
        fixed_cluster_i = []
        fixed_clusters = 8*[[]]
    else:
        log_debug(logger, f'Using existing optimized locus iterations of length {len(optimized_locus_iterations[-1][0])}')
        fixed_cluster_i = optimized_locus_iterations[-1][2]
        fixed_clusters = copy_list_of_selection_points(optimized_locus_iterations[-1][0])
        fixed_clusters = [list(x) for x in copy_list_of_selection_points(fixed_clusters)]

    cur_conv = convolution_simulation_per_ls(cur_chrom, data_per_length_scale, fixed_clusters)
    best_loss = calc_mse_loss(data_per_length_scale, cur_conv)

    for iteration in range(max_n_clusters):
        log_debug(logger, f'Ranking {iteration+1} of {max_n_clusters} clusters')

        # Have to define this here because of the parallelization
        def _optimize_cluster(cluster_i, iteration, fixed_clusters):
            if cluster_i in fixed_cluster_i:
                return None, None

            # Seeded per (chrom, iteration, cluster): joblib workers are separate processes with
            # their own RNG state, so without this each cluster's optimisation would be drawn from
            # OS entropy. Keyed on identity rather than call order so a cluster's result is the same
            # serial or parallel, and at any n_cores.
            seed_task(derive_seed('rank_loci', cur_chrom, iteration, cluster_i))

            cur_selection_points = [[x[cluster_i]] for x in best_selection_points]
            cur_selection_points = copy_list_of_selection_points([list(x) + list(y) for x, y in zip(fixed_clusters, cur_selection_points)])
            cur_selection_points_per_cluster = list(zip(*cur_selection_points))
            cur_up_down_order = fixed_up_down_order + [original_up_down_order[cluster_i]]

            N_iterations_ = int(N_iterations * np.sqrt(iteration + 1))
            best_selection_points_per_cluster, optim_loss, all_losses = _optimize_selection_points(
                N_iterations_, cur_selection_points_per_cluster, data_per_length_scale, cur_chrom, segment_size_dict=segment_size_dict,
                allow_pos_change=False, best_loss=np.inf, max_fitness=max_fitness, final_iteration=False, blocked_cluster_positions=None,
                up_down_order=cur_up_down_order, N_iterations_base=int(N_iterations_/2),
                # This is currently disabled so that all fitness values are possible, filtering will happen after this so it should be fine
                # allowed_fitness_change=allowed_fitness_change[:, np.array(fixed_cluster_i + [cluster_i])]
                )
            return optim_loss, copy_list_of_selection_points(best_selection_points_per_cluster)

        if n_cores is not None:
            results = Parallel(n_jobs=n_cores)(delayed(_optimize_cluster)(cluster_i, iteration, fixed_clusters) for cluster_i in range(N_clusters))
        else:
            results = [_optimize_cluster(cluster_i, iteration, fixed_clusters) for cluster_i in range(N_clusters)]

        loss_per_added_cluster, all_optimized_loci = zip(*results)

        best_cluster_i = np.argmin([x if x is not None else np.inf for x in loss_per_added_cluster])
        log_debug(logger, f'Best cluster {best_cluster_i} at pos {all_optimized_loci[best_cluster_i][-1][0][0].pos:.4e} with loss {loss_per_added_cluster[best_cluster_i]}')
        fixed_cluster_i.append(best_cluster_i)
        fixed_up_down_order.append(original_up_down_order[best_cluster_i])

        if loss_per_added_cluster[best_cluster_i] < best_loss:
            best_loss = loss_per_added_cluster[best_cluster_i]
            fixed_clusters = copy_list_of_selection_points(list(zip(*all_optimized_loci[best_cluster_i])))
        else:
            logger.warning(f'Loss did not improve, setting fitness of cluster to zero')
            # Use the selection points pre-optimization and set the latest selection point to zero fitness
            cur_selection_points = [[x[best_cluster_i]] for x in best_selection_points]
            fixed_clusters = copy_list_of_selection_points([list(x) + list(y) for x, y in zip(fixed_clusters, cur_selection_points)])
            for i in range(8):
                fixed_clusters[i][-1] = SelectionPoints(loci=[[fixed_clusters[i][-1][0].pos, 0]])

        optimized_locus_iterations.append((copy_list_of_selection_points(fixed_clusters), loss_per_added_cluster, deepcopy(fixed_cluster_i)))

    log_debug(logger, f'Finished ranking loci for {cur_chrom}.')

    return optimized_locus_iterations


@CALC_NEW()
def within_ci_fitness_filter(
        cur_chrom,
        ranked_selection_points,
        final_events_df=None,
        data_per_length_scale=None,
        segment_size_dict=DEFAULT_SEGMENT_SIZE_DICT,
        loci_to_consider=None,
        n_neighbors_optimization=10,
        N_iterations_optimization=10_000,
        mse_threshold=0.01,
        remove_empty_loci=True,
        show_progress=False,
        log_progress=True):
    log_debug(logger, f'Within CI filtering loci for {cur_chrom}')
    
    assert len(ranked_selection_points) == 8, len(ranked_selection_points)
    if len(ranked_selection_points[0]) == 0:
        log_debug(logger, 'No loci to filter, returning')
        return ranked_selection_points

    assert all([x['chrom']==cur_chrom for x in data_per_length_scale.values()]), f'Wrong data_per_length_scale for current chrom {cur_chrom}'

    pre_filter_fitness = np.stack([[ls[0].fitness for ls in locus] for locus in list(zip(*ranked_selection_points))])
    log_debug(logger, f'Pre-filtering, {np.sum(pre_filter_fitness == 0)} loci and length scales have zero fitness (out of {8* len(pre_filter_fitness)})')

    if loci_to_consider is not None:
        loci_to_consider = np.sort(loci_to_consider)[::-1]
    else:
        loci_to_consider = np.arange(len(ranked_selection_points[0]), 0, -1) - 1
    log_debug(logger, f'Considering {len(loci_to_consider)} out of {len(ranked_selection_points[0])} loci for filtering')

    base_conv = convolution_simulation_per_ls(cur_chrom, data_per_length_scale, ranked_selection_points)
    up_down_order = [any([ranked_selection_points[j][cluster_j][0].fitness > 0 for j in range(0, 8, 2)])
                     for cluster_j in range(len(ranked_selection_points[0]))]
    adjusted_selection_points = [list(x) for x in copy_list_of_selection_points(ranked_selection_points)]


    for iteration, cluster_i in enumerate(loci_to_consider):
        old_fitness = [x[cluster_i][0].fitness for x in adjusted_selection_points]
        cur_pos = adjusted_selection_points[0][cluster_i][0].pos
        if log_progress:
            log_debug(logger, f'Filtering cluster {cluster_i} at position {cur_pos} ({iteration+1} / {len(loci_to_consider)})')

        cur_zero_selection_points = [list(x) for x in copy_list_of_selection_points(adjusted_selection_points)]
        for i in range(8):
            cur_zero_selection_points[i][cluster_i] = SelectionPoints(loci=[Locus(cur_pos, 0)])

        # Optimize the neighborhood of the current cluster. If the optimized score is better, use that one.
        if len(cur_zero_selection_points[0]) > 1:
            zero_conv = convolution_simulation_per_ls(cur_chrom, data_per_length_scale, cur_zero_selection_points,
                                    segment_size_dict=segment_size_dict)
            zero_loss = calc_mse_loss(data_per_length_scale, zero_conv)
            cur_zero_selection_points_per_cluster = list(zip(*cur_zero_selection_points))
            neighborhood_clusters = np.argsort([np.abs(x[0].pos-cur_pos) for x in cur_zero_selection_points[0]])[1:n_neighbors_optimization+1]
            allowed_fitness_change = np.stack([[x[0].fitness != 0 for x in y] for y in adjusted_selection_points])
            if cluster_i in neighborhood_clusters:
                logger.warning(f'Cluster {cluster_i} in neighborhood clusters {neighborhood_clusters}. Distances of them are: {np.array([np.abs(x[0].pos-cur_pos) for x in cur_zero_selection_points[0]])[neighborhood_clusters]}')
            optimized_selection_points_per_cluster, optim_loss, _ = _optimize_selection_points(
                        N_iterations_optimization, cur_zero_selection_points_per_cluster, data_per_length_scale, cur_chrom,
                        best_loss=zero_loss, segment_size_dict=segment_size_dict, final_iteration=False, show_progress=False,
                        loci_to_optimize=neighborhood_clusters, N_iterations_base=0, max_deviation=0.0001,
                        max_pos_change=0, allow_pos_change=False, allowed_fitness_change=allowed_fitness_change,
                        up_down_order=up_down_order, blocked_distance_th=2e5)
            assert all([x[0].fitness == 0 for x in optimized_selection_points_per_cluster[cluster_i]])
            if optim_loss < zero_loss:
                cur_zero_selection_points = list(zip(*optimized_selection_points_per_cluster))

        # Simulate and check if the signal is still within the 95% confidence interval
        zero_conv = convolution_simulation_per_ls(cur_chrom, data_per_length_scale, cur_zero_selection_points)
        for i, (data, cur_zero_conv, cur_base_conv) in enumerate(
            zip(data_per_length_scale.values(), zero_conv, base_conv)):
            
            cur_pos_i = int(np.round(cur_pos / segment_size_dict[data['length_scale']], 0))
            cur_loci_width_i = int(data['loci_width'] / 4)
            left_i = max(0, cur_pos_i - cur_loci_width_i)
            right_i = min(len(data['signals']), cur_pos_i + cur_loci_width_i)
            cur_indices = np.intersect1d(np.arange(left_i, right_i), data['non_centromere_index'])

            old_within_ci = np.mean(np.logical_and(
                cur_base_conv < data['signal_bounds'][1],
                cur_base_conv > data['signal_bounds'][0])[cur_indices])
            zero_within_ci = np.mean(np.logical_and(
                cur_zero_conv < data['signal_bounds'][1],
                cur_zero_conv > data['signal_bounds'][0])[cur_indices])
            old_loss = (np.mean((data['signals'] - cur_base_conv)[cur_indices] ** 2) /
                        data['cur_loss_norm'])
            zero_loss = (np.mean((data['signals'] - cur_zero_conv)[cur_indices] ** 2) /
                         data['cur_loss_norm'])

            set_to_zero = (zero_within_ci == 1 or 
                           ((old_within_ci == 0 or zero_within_ci >= old_within_ci) and
                            ((zero_loss - old_loss) / old_loss) < mse_threshold))

            if log_progress and logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'Cluster {cluster_i}, LS {i} => {"Set to zero" if set_to_zero else "Keep"} (cur_loci_width_i: {cur_loci_width_i}, left_i: {left_i}, right_i: {right_i}): ' +
                             f'Old loss: {old_loss:.4e}, Zero loss: {zero_loss:.4e}, Ratio: {(zero_loss - old_loss) / old_loss:4f}, ' +
                             f'Old within CI: {old_within_ci:.4e}, Zero within CI: {zero_within_ci:.4e}'
                             )

            if set_to_zero:
                adjusted_selection_points[i] = list(cur_zero_selection_points[i])
                assert adjusted_selection_points[i][cluster_i][0].fitness == 0
                base_conv[i] = cur_zero_conv

        new_fitness = [x[cluster_i][0].fitness for x in adjusted_selection_points]
        if log_progress:
            log_debug(logger, f'Cluster {cluster_i}: Old fitness: {old_fitness}. New fitness: {new_fitness}' )

    post_filter_fitness = np.stack([[ls[0].fitness for ls in locus] for locus in list(zip(*adjusted_selection_points))])
    log_debug(logger, f'Filtering increased zero fitness loci/length scales by {np.sum(post_filter_fitness == 0) - np.sum(pre_filter_fitness == 0)} from {np.sum(pre_filter_fitness == 0)} to {np.sum(post_filter_fitness == 0)} (out of {8* len(pre_filter_fitness)}).')

    # Filter out loci that have zero fitness across all length scales
    if remove_empty_loci:
        old_n_loci = len(adjusted_selection_points[0])
        cur_is_nonzero = np.any(np.stack(
            [[locus[0].fitness > 0 for locus in ls_selection_points] for ls_selection_points in adjusted_selection_points]
            ), axis=0)
        adjusted_selection_points = [
            [locus for locus, cur_nonzero in zip(ls_selection_points, cur_is_nonzero) if cur_nonzero]
            for ls_selection_points in adjusted_selection_points
        ]
        log_debug(logger, f'Filtered out {old_n_loci - len(adjusted_selection_points[0])} loci with zero fitness across all length scales, resulting in {len(adjusted_selection_points[0])} non-zero loci.')


    log_debug(logger, f'Finished within ci filtering loci for {cur_chrom}.')

    return adjusted_selection_points


@CALC_NEW()
def limiting_fitness(
    cur_chrom, 
    raw_selection_points, 
    data_per_length_scale, 
    segment_size_dict=DEFAULT_SEGMENT_SIZE_DICT, 
    max_iterations=10, 
    max_deviation=0.0001, 
    blocked_distance_th=2e5, 
    show_progress=False,
    loss_threshold=0.25,
    within_ci_threshold=0.025,
    ls_i_to_check=None,
    N_iterations_optim=1_000,
    allow_all_fitness_change=True,
):
    log_debug(logger, f'Limiting fitness')

    assert all([x['chrom']==cur_chrom for x in data_per_length_scale.values()]), f'Wrong data_per_length_scale for current chrom {cur_chrom}'

    if ls_i_to_check is None:
        ls_i_to_check = (0, 1, 2, 3, 4, 5, 6, 7)

    cur_selection_points = [list(x) for x in copy_list_of_selection_points(raw_selection_points)]

    raw_conv = convolution_simulation_per_ls(
        cur_chrom, data_per_length_scale, cur_selection_points, segment_size_dict=segment_size_dict)
    raw_loss = calc_mse_loss(data_per_length_scale, raw_conv)

    up_down_order = [any([raw_selection_points[j][cluster_j][0].fitness > 0 for j in range(0, 8, 2)])
                     for cluster_j in range(len(raw_selection_points[0]))]
    if allow_all_fitness_change:
        allowed_fitness_change = None
    else:
        allowed_fitness_change = np.stack([[x[0].fitness != 0 for x in y] for y in raw_selection_points])

    best_selection_points = [list(x) for x in copy_list_of_selection_points(raw_selection_points)]
    for ls_i in ls_i_to_check:

        raw_conv = convolution_simulation_per_ls(
            cur_chrom, data_per_length_scale, best_selection_points, segment_size_dict=segment_size_dict)
        raw_loss = calc_mse_loss(data_per_length_scale, raw_conv)

        log_debug(logger, f'Limiting for length scales {ls_i}')
        cur_data_per_length_scale = {k: v for i, (k, v) in enumerate(data_per_length_scale.items()) if i == ls_i}

        raw_within_ci = np.mean([np.mean(np.logical_and(
                        cur_conv[data['non_centromere_index']] < data['signal_bounds'][1][data['non_centromere_index']],
                        cur_conv[data['non_centromere_index']] > data['signal_bounds'][0][data['non_centromere_index']]))
                    for ls, (data, cur_conv) in enumerate(zip(data_per_length_scale.values(), raw_conv)) if ls == ls_i])        

        upper_limit = 1
        lower_limit = 0

        for iteration in range(max_iterations):
            old_max_fitness = np.max([x[0].fitness for x in best_selection_points[ls_i]])
            # Adjust the the fitness values
            cur_selection_points = copy_list_of_selection_points(best_selection_points)
            for cluster_i in range(len(cur_selection_points[0])):
                cur_selection_points[ls_i][cluster_i] = SelectionPoints(loci=[[
                    cur_selection_points[ls_i][cluster_i][0].pos, cur_selection_points[ls_i][cluster_i][0].fitness
                    * (upper_limit + lower_limit) / 2]])

            new_max_fitness = np.max([x[0].fitness for x in cur_selection_points[ls_i]])
            # This makes sure length scales other than ls_i are not capped
            max_fitness = 1_000 * np.ones(8)
            max_fitness[ls_i] = (old_max_fitness + new_max_fitness) / 2
    
            # optimize the values
            pre_optim_conv = convolution_simulation_per_ls(
                cur_chrom, data_per_length_scale, cur_selection_points, segment_size_dict=segment_size_dict)          
            pre_optim_loss = calc_mse_loss(data_per_length_scale, pre_optim_conv)

            log_debug(logger, f'Max fitness for ls {ls_i} before optim: {np.max([x[0].fitness for x in cur_selection_points[ls_i]])}')

            optimized_selection_points_per_cluster, optim_loss, all_losses_optimization = _optimize_selection_points(
                N_iterations_optim, list(zip(*cur_selection_points)), data_per_length_scale, cur_chrom,
                best_loss=pre_optim_loss, segment_size_dict=segment_size_dict, final_iteration=False, show_progress=show_progress,
                loci_to_optimize=None, N_iterations_base=0, max_deviation=max_deviation,
                max_fitness=max_fitness, ls_to_optimize=[ls_i],
                max_pos_change=0, allow_pos_change=False, up_down_order=up_down_order, blocked_distance_th=blocked_distance_th,
                allowed_fitness_change=allowed_fitness_change)
            if optim_loss == pre_optim_loss:
                logger.debug('Loss did not improve during selection process')
            if optim_loss < pre_optim_loss:
                cur_selection_points = list(zip(*optimized_selection_points_per_cluster))
                cur_selection_points = [list(x) for x in cur_selection_points]

            # Caluculate new loss and within CI values
            optim_conv = convolution_simulation_per_ls(
                cur_chrom, data_per_length_scale, cur_selection_points, segment_size_dict=segment_size_dict)
            new_loss = calc_mse_loss(data_per_length_scale, optim_conv)
            new_within_ci = np.mean([np.mean(np.logical_and(
                cur_conv[data['non_centromere_index']] < data['signal_bounds'][1][data['non_centromere_index']],
                cur_conv[data['non_centromere_index']] > data['signal_bounds'][0][data['non_centromere_index']]))
                for data, cur_conv in zip(cur_data_per_length_scale.values(), optim_conv[ls_i:ls_i+1])])            

            log_debug(logger, f'{iteration}: Cur factor: {(upper_limit + lower_limit) / 2:.4g}. Max allowed fitness: {max_fitness[ls_i]:.2f}. Max fitness: {old_max_fitness:.2f} -> {new_max_fitness:.2f}. Loss: {raw_loss:.4e} -> {pre_optim_loss:.4e} (raw) -> {new_loss:.4e}. Loss ratio {(new_loss - raw_loss) / raw_loss:.4e}. Within CI: {raw_within_ci:.2f} -> {new_within_ci:.2f}.')
            
            log_debug(logger, f'Max fitness for ls {ls_i} after optim: {np.max([x[0].fitness for x in cur_selection_points[ls_i]])}')
            if (((new_loss - raw_loss) / raw_loss < loss_threshold) and ((raw_within_ci - new_within_ci) < within_ci_threshold)):
                raw_loss = new_loss
                raw_within_ci = new_within_ci
                accepted = True
                upper_limit = (upper_limit + lower_limit) / 2
                best_selection_points = copy_list_of_selection_points(cur_selection_points)
            else:
                accepted = False
                lower_limit = (upper_limit + lower_limit) / 2

            log_debug(logger, f'{"Accepted" if accepted else "Not accepted"}. New upper limit: {upper_limit:.3g}, new lower limit: {lower_limit:.3g}')


    return best_selection_points


@CALC_NEW()
def final_optimization_step(
    cur_chrom,
    final_selection_points,
    final_events_df=None,
    data_per_length_scale=None,
    segment_size_dict=DEFAULT_SEGMENT_SIZE_DICT,
    n_neighbors_optimization=10,
    N_iterations_optimization=11_000,
    max_pos_change=1e5):

    log_debug(logger, f'Final optimization step for {len(final_selection_points[0])} loci for {cur_chrom}')
    final_selection_points = copy_list_of_selection_points(final_selection_points)

    assert all([x['chrom']==cur_chrom for x in data_per_length_scale.values()]), f'Wrong data_per_length_scale for current chrom {cur_chrom}'

    N_loci = len(final_selection_points[0])
    original_conv = convolution_simulation_per_ls(cur_chrom, data_per_length_scale, final_selection_points,
                        segment_size_dict=segment_size_dict)
    original_loss = calc_mse_loss(data_per_length_scale, original_conv)
    log_debug(logger, f'Original loss: {original_loss:.4e}')

    up_down_order = [any([final_selection_points[j][cluster_j][0].fitness > 0 for j in range(0, 8, 2)])
                     for cluster_j in range(len(final_selection_points[0]))]
    allowed_fitness_change = np.stack([[x[0].fitness != 0 for x in y] for y in final_selection_points])
    base_conv = convolution_simulation_per_ls(cur_chrom, data_per_length_scale, final_selection_points)
    best_loss = calc_mse_loss(data_per_length_scale, base_conv)

    all_losses = [best_loss]
    for cluster_i in np.concatenate((np.arange(N_loci), np.arange(N_loci)[::-1][1:])):
        log_debug(logger, f'Optimizing locus {cluster_i} (out of {len(final_selection_points[0])})')

        cur_pos = final_selection_points[0][cluster_i][0].pos
        neighborhood_clusters = np.argsort([np.abs(x[0].pos-cur_pos) for x in final_selection_points[0]])[:n_neighbors_optimization+1]
        assert cluster_i in neighborhood_clusters
    
        optimized_selection_points_per_cluster, optim_loss, _ = _optimize_selection_points(
                    N_iterations_optimization, list(zip(*final_selection_points)), data_per_length_scale, cur_chrom,
                    best_loss=best_loss, segment_size_dict=segment_size_dict, final_iteration=False, show_progress=False,
                    loci_to_optimize=neighborhood_clusters, N_iterations_base=0, max_deviation=0.0001,
                    max_pos_change=max_pos_change, allow_pos_change=False, up_down_order=up_down_order,
                    blocked_distance_th=2e5, allowed_fitness_change=allowed_fitness_change)
        cur_optim_selection_points = list(zip(*optimized_selection_points_per_cluster))
        log_debug(logger, f'Optimized {cluster_i}: former best loss = {best_loss:.4e}, optim loss = {optim_loss:.4e} --> {"keep old" if optim_loss >= best_loss else "update to new"}')
        if optim_loss < best_loss:
            best_loss = optim_loss
            final_selection_points = copy_list_of_selection_points(cur_optim_selection_points)
        all_losses.append(best_loss)

    log_debug(logger, f'Done! Original loss: {original_loss:.4e}. Final optimized loss: {best_loss:.4e}')

    return final_selection_points, all_losses


@CALC_NEW()
def infer_loci_widths(
    cur_chrom,
    final_selection_points,
    loci_results_dir=None,
    data_per_length_scale=None,
    segment_size_dict=DEFAULT_SEGMENT_SIZE_DICT,
    num_bootstrap_iterations=250,
    max_pos_change=1e5,
    max_deviation=0.00001,
    max_fitness=None,
    num_optimization_iterations=1_000,
    which_loci=None,
    n_jobs=-1,
    N_bootstrap=None
    ):
    """
    Infer locus widths using bootstrap resampling and optimization.

    Parameters:
    - cur_chrom (str): Current chromosome.
    - data_per_length_scale (list): Data per length scale.
    - final_selection_points (list): Final selection points.
    - final_events_df (DataFrame): Final events data.
    - segment_size_dict (dict): Segment size dictionary.
    - data_loaders_dir (str): Directory for data loaders.
    - num_bootstrap_iterations (int): Number of bootstrap iterations (default: 100).
    - max_pos_change (float): Maximum position change during optimization (default: 1e5).
    - max_deviation (float): Maximum deviation for optimization (default: 0.00001).
    - max_fitness (float): Maximum fitness value (default: None).
    - num_optimization_iterations (int): Number of optimization iterations (default: 1_000).
    - n_jobs (int): Number of jobs for parallelization (default: -1, all available cores).

    Returns:
    - bootstrap_loci_widths (list): Inferred locus widths from bootstrap resampling.
    """
    log_debug(logger, f'Inferring locus widths for {cur_chrom}')

    assert num_bootstrap_iterations <= N_bootstrap, f'num_bootstrap_iterations ({num_bootstrap_iterations}) cannot be larger than N_bootstrap ({N_bootstrap})'

    if loci_results_dir is None:
        loci_results_dir = os.path.join(config['directories']['results_dir'], config['name'], 'loci_detection')
    if N_bootstrap is None:
        N_bootstrap = config['loci_detection']['N_bootstrap']

    signal_bootstraps = open_pickle(os.path.join(loci_results_dir, 'signal_bootstrap', f'{cur_chrom}_N_{N_bootstrap}.pickle'))

    assert all([x['chrom']==cur_chrom for x in data_per_length_scale.values()]), f'Wrong data_per_length_scale for current chrom {cur_chrom}'

    if which_loci is None:
        which_loci = np.arange(len(final_selection_points[0]))

    up_down_order = [any([final_selection_points[j][cluster_j][0].fitness > 0 for j in range(0, 8, 2)])
                     for cluster_j in range(len(final_selection_points[0]))]
    allowed_fitness_change = np.stack([[x[0].fitness != 0 for x in y] for y in final_selection_points])
    simulated_conv = convolution_simulation_per_ls(
        cur_chrom, data_per_length_scale, final_selection_points, segment_size_dict=segment_size_dict)

    def __optimize_for_bootstrap_iteration(bootstrap_iteration, cluster_i):
        # Per-task stream, keyed on the bootstrap iteration and locus this call stands for (see
        # _optimize_cluster above): reproducible across processes and independent of n_jobs.
        seed_task(derive_seed('infer_loci_widths', cur_chrom, cluster_i, bootstrap_iteration))
        mod_data_per_length_scale = deepcopy(data_per_length_scale)
        for ls_i in range(8):
            ls_key = list(mod_data_per_length_scale.keys())[ls_i]
            mod_data_per_length_scale[ls_key]['signals'] = signal_bootstraps[ls_i][bootstrap_iteration]
        mod_base_loss = calc_mse_loss(mod_data_per_length_scale, simulated_conv)

        optimized_selection_points_per_cluster, optim_loss, _ = _optimize_selection_points(
            num_optimization_iterations, list(zip(*final_selection_points)), mod_data_per_length_scale, cur_chrom,
            best_loss=mod_base_loss, segment_size_dict=segment_size_dict, final_iteration=False, show_progress=False,
            loci_to_optimize=[cluster_i], N_iterations_base=0, max_deviation=max_deviation,
            max_fitness=max_fitness if max_fitness is not None else np.max([[y[0].fitness for y in x] for x in final_selection_points]),
            max_pos_change=max_pos_change, allow_pos_change=True, up_down_order=up_down_order,
            allowed_fitness_change=allowed_fitness_change)

        return optimized_selection_points_per_cluster[cluster_i][0][0].pos

    bootstrap_loci_widths = []
    for cluster_i in which_loci:
        log_debug(logger, f'Inferring locus width for cluster {cluster_i} of {len(final_selection_points[0])}')
        if n_jobs == 1:
            all_pos = [
            __optimize_for_bootstrap_iteration(bootstrap_iteration, cluster_i)
            for bootstrap_iteration in range(num_bootstrap_iterations)
            ]
        else:
            all_pos = Parallel(n_jobs=n_jobs)(
            delayed(__optimize_for_bootstrap_iteration)(bootstrap_iteration, cluster_i)
            for bootstrap_iteration in range(num_bootstrap_iterations)
            )
        bootstrap_loci_widths.append(all_pos)

    log_debug(logger, 'Finished optimizing locus widths!')

    return bootstrap_loci_widths


@CALC_NEW()
def merge_overlapping_loci(
    cur_chrom, 
    selection_points, 
    loci_widths,
    data_per_length_scale,
    segment_size_dict=DEFAULT_SEGMENT_SIZE_DICT,
    n_iterations_optim=10_000,
    show_progress_optim=False,
    max_deviation_optim=0.00001,
    nr_stds_widths=5,
):
    """
    Identify overlapping loci for a single chromosome, remove them, and optimize the remaining ones.
    
    Parameters:
    -----------
    cur_chrom : str
        Chromosome to process
    selection_points : list
        Selection points for the chromosome
    loci_widths : list
        Locus widths for the chromosome
    data_per_length_scale : list
        Data per length scale for the chromosome
    segment_size_dict : dict
        Dictionary of segment sizes for different length scales
    n_iterations_optim : int
        Number of iterations for optimization
    show_progress_optim : bool
        Whether to show progress during optimization
    max_deviation_optim : float
        Maximum allowed deviation in optimization
    output_dir : str, optional
        Directory to save output file
    filename : str, optional
        Name of the output file
        
    Returns:
    --------
    list
        Optimized selection points for the chromosome
    list
        Convolution results for the chromosome
    pandas.DataFrame
        DataFrame containing information about removed loci
    """
    from spice.tsg_og.loci import create_loci_df
    
    log_debug(logger, f'Merging overlapping loci for {cur_chrom}')

    assert all([x['chrom']==cur_chrom for x in data_per_length_scale.values()]), f'Wrong data_per_length_scale for current chrom {cur_chrom}'
    assert len(selection_points[0]) > 0, f'No selection points for {cur_chrom}!'
    assert len(selection_points[0]) == len(loci_widths), f'Number of selection points ({len(selection_points[0])}) and locus widths ({len(loci_widths)}) do not match!'

    loci_df = create_loci_df({cur_chrom: selection_points}, {cur_chrom: loci_widths}, nr_stds_widths=nr_stds_widths).sort_values('rank_on_chrom')
    log_debug(logger, f'Found {len(loci_df)} loci for {cur_chrom}.')

    # Find overlapping loci. create_loci_df's canonical schema is start/end/type (the std-based
    # bounds pos±width/2, and OG/TSG direction); this step still referenced the retired
    # start_std/end_std/up_down names -> KeyError. It only surfaced under loci_steps='full' (which
    # runs `merging`; 'fast' skips it), so the schema drift had gone unnoticed. Map to current columns.
    loci_within_other_loci = np.logical_and(
        np.logical_and(
            loci_df['start'].values[:, None] < loci_df['pos'].values,
            loci_df['end'].values[:, None] > loci_df['pos'].values
        ),
        np.logical_and(
            loci_df['chrom'].values[:, None] == loci_df['chrom'].values,
            loci_df['type'].values[:, None] == loci_df['type'].values,
        )
    )
    np.fill_diagonal(loci_within_other_loci, False)
    
    # Iteratively remove loci until no overlap remains
    loci_to_remove = []
    inside_loci = []
    loci_within_other_loci_copy = loci_within_other_loci.copy()
    
    while loci_within_other_loci_copy.sum() > 0:
        nr_inside = loci_within_other_loci_copy.sum(axis=1)
        if nr_inside.max() == 0:
            break
            
        locus_to_remove_idx = np.argmax(nr_inside)
        loci_to_remove.append(locus_to_remove_idx)
        inside_loci.extend(list(np.where(loci_within_other_loci_copy[locus_to_remove_idx, :])[0]))
        
        loci_within_other_loci_copy[locus_to_remove_idx, :] = False
        loci_within_other_loci_copy[:, locus_to_remove_idx] = False
        
    remaining_loci = np.setdiff1d(np.arange(len(loci_df)), loci_to_remove)
    log_debug(logger, f"Out of {len(loci_df)} initial loci, removing {len(loci_to_remove)} loci  ({len(loci_df) - len(loci_to_remove)} remaining): {loci_to_remove}")

    # Calculate the base loss for the current chromosome
    base_conv = convolution_simulation_per_ls(
        cur_chrom, data_per_length_scale, selection_points)
    base_loss = calc_mse_loss(data_per_length_scale, base_conv)

    # Break if no loci to remove    
    if len(loci_to_remove) == 0:
        logger.warning(f'No loci to merge for {cur_chrom}. Returning original selection points.')
        return selection_points, base_conv, loci_df.loc[[]], loci_to_remove

    # Set fitness of removed loci to zero
    merged_selection_points = [list(x) for x in copy_list_of_selection_points(selection_points)]
    for cluster_i in loci_to_remove:
        for i in range(8):
            merged_selection_points[i][cluster_i] = SelectionPoints(
                loci=[[merged_selection_points[i][cluster_i][0].pos, 0]]
            )

    # Calculate initial loss after zeroing out removed loci
    merged_conv_raw = convolution_simulation_per_ls(
        cur_chrom, data_per_length_scale, merged_selection_points
    )
    merged_loss_raw = calc_mse_loss(data_per_length_scale, merged_conv_raw)
    
    # Setup optimization constraints
    allowed_fitness_change = np.stack([[x[0].fitness != 0 for x in y] for y in merged_selection_points])
    cur_remaining_loci = loci_df.loc[remaining_loci]['rank_on_chrom'].values
    up_down_order = [any([merged_selection_points[j][cluster_j][0].fitness > 0 for j in range(0, 8, 2)])
                     for cluster_j in range(len(merged_selection_points[0]))]
    
    # Allow fitness change for loci that are overlapping with the removed loci, this way they can "take over" the fitness of the removed loci for all LS
    overlapping_loci = np.setdiff1d(np.unique(np.concatenate([np.where(x)[0] for x in loci_within_other_loci[loci_to_remove]])),
                                    loci_to_remove)
    allowed_fitness_change[:, overlapping_loci] = True

    # Optimize the remaining loci
    optimized_selection_points_per_cluster, optim_loss, _ = _optimize_selection_points(
        n_iterations_optim, 
        list(zip(*merged_selection_points)), 
        data_per_length_scale, 
        cur_chrom,
        best_loss=merged_loss_raw, 
        show_progress=show_progress_optim, 
        N_iterations_base=0, 
        segment_size_dict=segment_size_dict,
        loci_to_optimize=cur_remaining_loci, 
        final_iteration=False, 
        allowed_fitness_change=allowed_fitness_change,
        max_deviation=max_deviation_optim, 
        allow_pos_change=False,
        up_down_order=up_down_order,
        blocked_distance_th=2e5
    )
    
    merged_selection_points_optim = list(zip(*optimized_selection_points_per_cluster))
    log_debug(logger, f'{len(merged_selection_points_optim[0])} merged_selection_points')

    merged_conv_optim = convolution_simulation_per_ls(
        cur_chrom, data_per_length_scale, merged_selection_points_optim
    )
    
    log_debug(logger, f"base loss {base_loss:.2e} - merged raw loss: {merged_loss_raw:.2e} - merged optimized loss: {optim_loss:.2e}")
    
    # Filter out loci that have zero fitness across all length scales
    cur_is_nonzero = np.any(np.stack(
        [[locus[0].fitness > 0 for locus in ls_selection_points] for ls_selection_points in merged_selection_points_optim]
        ), axis=0)
    filtered_merged_selection_points = [
        [locus for locus, cur_nonzero in zip(ls_selection_points, cur_is_nonzero) if cur_nonzero]
        for ls_selection_points in merged_selection_points_optim
    ]
    log_debug(logger, f'Filtered merged selection points: {len(filtered_merged_selection_points[0])} remaining loci after filtering')
    
    return filtered_merged_selection_points, merged_conv_optim, loci_df.loc[loci_to_remove], loci_to_remove


@CALC_NEW()
def filter_loci(
    cur_chrom,
    final_selection_points,
    loci_widths,
    data_per_length_scale,
    final_events_df,
    segment_size_dict=DEFAULT_SEGMENT_SIZE_DICT,
    max_n_iterations=25,
    th_width_to_kernel_ratio_largest=1,
    th_max_abs_fitness=0,
    th_sum_abs_fitness=0,
    th_locus_prominence=10,
    th_added_events=0,
    perform_prominence_overlap_check=False,
    n_iterations_optim=100_000,
    show_progress_optim=False,
    max_deviation_optim=0.00001,
    nr_stds_widths=5,
    limit_max_fitness=True,
    prominence_calc_on='conv',
):
    
    log_debug(logger, f'Final locus filtering for {cur_chrom}')

    if loci_widths is None:
        loci_widths = np.zeros((len(final_selection_points[0]), 200))
    assert len(loci_widths) == len(final_selection_points[0]), f'Number of locus widths ({len(loci_widths)}) does not match number of selection points ({len(final_selection_points[0])})!'

    initial_n_loci = len(final_selection_points[0])
    allowed_fitness_change_pos = np.array([], dtype=float)
    for cur_iteration in range(max_n_iterations):
        log_debug(logger, f'Iteration {cur_iteration + 1} of {max_n_iterations}')

        # Calculate the base loss for the current chromosome
        base_conv = convolution_simulation_per_ls(
            cur_chrom, data_per_length_scale, final_selection_points)
        base_loss = calc_mse_loss(data_per_length_scale, base_conv)

        cur_fitness = np.array([[x[0].fitness for x in ls] for ls in final_selection_points]).T

        # Filter based on ratio of width to kernel size (of largest nonzero LS)
        largest_nonzero_ls_i = 7 - np.argmax(cur_fitness[:, ::-1]!=0, axis=1)
        largest_nonzero_ls = [{
            0: 'small', 1: 'small', 2: 'mid1', 3: 'mid1', 4: 'mid2', 5: 'mid2', 6: 'large', 7: 'large'
            }[x] for x in largest_nonzero_ls_i]
        largest_nonzero_ls_kernel_size = [
            data_per_length_scale[(x, 'gain')]['loci_width'] * DEFAULT_SEGMENT_SIZE_DICT[x]
            for x in largest_nonzero_ls]

        loci_to_keep_bool, loci_to_keep, loci_to_remove = _identify_loci_to_filter(
            cur_chrom=cur_chrom,
            data_per_length_scale=data_per_length_scale,
            final_selection_points=final_selection_points,
            loci_widths=loci_widths,
            final_events_df=final_events_df,
            cur_iteration=cur_iteration,
            nr_stds_widths=nr_stds_widths,
            th_width_to_kernel_ratio_largest=th_width_to_kernel_ratio_largest,
            th_max_abs_fitness=th_max_abs_fitness,
            th_sum_abs_fitness=th_sum_abs_fitness,
            th_locus_prominence=th_locus_prominence,
            th_added_events=th_added_events,
            prominence_calc_on=prominence_calc_on,
            perform_prominence_overlap_check=perform_prominence_overlap_check
        )

        if len(loci_to_remove) == 0:
            log_debug(logger, f'No more loci to remove for {cur_chrom} (iteration {cur_iteration + 1} of {max_n_iterations}).')
            break

        log_debug(logger, f'Final loci to keep: {np.sum(loci_to_keep_bool)} out of {len(loci_to_keep_bool)} loci (removed {np.sum(~loci_to_keep_bool)} loci)')
        if np.sum(loci_to_keep_bool) == 0:
            logger.warning(f'All loci removed for {cur_chrom}. Stopping here and returning empty selection points.')
            final_selection_points = [[] for _ in range(8)]
            break
        
        filtered_selection_points = [[x for x, keep in zip(ls_selection_points, loci_to_keep_bool) if keep]
                                    for ls_selection_points in final_selection_points]
        
        # Optimize the remaining loci after filtering
        
        ## Calculate initial loss after removing removed loci
        filtered_conv_raw = convolution_simulation_per_ls(
            cur_chrom, data_per_length_scale, filtered_selection_points
        )
        filtered_loss_raw = calc_mse_loss(data_per_length_scale, filtered_conv_raw)

        ## Setup optimization constraints
        allowed_fitness_change = np.stack([[x[0].fitness != 0 for x in y] for y in final_selection_points])
        up_down_order = np.array([any([final_selection_points[j][cluster_j][0].fitness > 0 for j in range(0, 8, 2)])
                            for cluster_j in range(len(final_selection_points[0]))])
        
        ## Find overlapping loci (based on the kernel size of the largest nonzero LS)
        locus_pos = np.array([x[0].pos for x in final_selection_points[0]])
        locus_start = locus_pos - np.array(largest_nonzero_ls_kernel_size) / 2
        locus_end = locus_pos + np.array(largest_nonzero_ls_kernel_size) / 2
        loci_within_other_loci = reduce(np.logical_and, [
            locus_start[loci_to_remove][:, None] < locus_pos[loci_to_keep],
            locus_end[loci_to_remove][:, None] > locus_pos[loci_to_keep],
            up_down_order[loci_to_remove][:, None] == up_down_order[loci_to_keep],
        ])

        ## Allow fitness change for loci that are overlapping with the removed loci, this way they can "take over" the fitness of the removed loci for all LS
        ## Have to do this in this awkward way because the indices change the whole time
        cur_overlapping_loci = loci_to_keep[np.unique(np.concatenate([np.where(x)[0] for x in loci_within_other_loci]))]
        prev_overlapping_loci = np.array([i for i, x in enumerate(filtered_selection_points[0]) if x[0].pos in allowed_fitness_change_pos])
        allowed_fitness_change_pos = np.concatenate([
            allowed_fitness_change_pos, np.array([x[0].pos for i, x in enumerate(final_selection_points[0]) if i in cur_overlapping_loci])])
        cur_allowed_fitness_change = np.unique(np.concatenate([prev_overlapping_loci, cur_overlapping_loci])).astype(int)
        allowed_fitness_change[:, cur_allowed_fitness_change] = True

        if allowed_fitness_change.any(axis=0).sum() > 0:
            ## Run optimization
            log_debug(logger, f'Optimizing {len(filtered_selection_points[0])} ({allowed_fitness_change.any(axis=0).sum()} have allowed fitness change, {len(cur_allowed_fitness_change)} because they are close to removed loci) remaining loci for {cur_chrom}')
            optimized_selection_points_per_cluster, optim_loss, _ = _optimize_selection_points(
                n_iterations_optim, 
                list(zip(*filtered_selection_points)), 
                data_per_length_scale, 
                cur_chrom,
                best_loss=filtered_loss_raw, 
                show_progress=show_progress_optim, 
                N_iterations_base=0, 
                max_fitness=[1.1*max([y[0].fitness for y in x]) for x in filtered_selection_points] if limit_max_fitness else None,
                segment_size_dict=segment_size_dict,
                loci_to_optimize=None, # all, since the loci to remove are already filtered out 
                final_iteration=False, 
                allowed_fitness_change=allowed_fitness_change[:, loci_to_keep],
                max_deviation=max_deviation_optim, 
                allow_pos_change=False,
                up_down_order=up_down_order[loci_to_keep],
                blocked_distance_th=2e5
            )
            filtered_selection_points_optim = list(zip(*optimized_selection_points_per_cluster))

        else:
            log_debug(logger, f'No loci with allowed fitness change for {cur_chrom}, skipping optimization.')
            filtered_selection_points_optim = filtered_selection_points
            optim_loss = filtered_loss_raw
        log_debug(logger, f"base loss {base_loss:.2e} - filtered raw loss: {filtered_loss_raw:.2e} - filtered optimized loss: {optim_loss:.2e}")

        # Filter out loci that have zero fitness across all length scales
        cur_is_nonzero = np.any(np.stack(
            [[locus[0].fitness > 0 for locus in ls_selection_points] for ls_selection_points in filtered_selection_points_optim]
            ), axis=0)
        filtered_selection_points_optim_nonzero = [
            [locus for locus, cur_nonzero in zip(ls_selection_points, cur_is_nonzero) if cur_nonzero]
            for ls_selection_points in filtered_selection_points_optim
        ]
        loci_widths = [widths for widths, cur_keep in zip(loci_widths, loci_to_keep_bool) if cur_keep]
        loci_widths = [widths for widths, cur_nonzero in zip(loci_widths, cur_is_nonzero) if cur_nonzero]

        log_debug(logger, f'Filtered merged selection points: {len(filtered_selection_points_optim_nonzero[0])} remaining loci after filtering')

        final_selection_points = filtered_selection_points_optim_nonzero
    else:
        log_debug(logger, f'Hit the maximum number of iterations {max_n_iterations}. Returning final selection points.')

    log_debug(logger, f'Finished final filtering: Removed {initial_n_loci - len(final_selection_points[0])} loci in total for {cur_chrom} (from {initial_n_loci} to {len(final_selection_points[0])})')

    return final_selection_points


def _identify_loci_to_filter(
    cur_chrom,
    data_per_length_scale,
    final_selection_points,
    loci_widths,
    final_events_df,
    cur_iteration=None,
    nr_stds_widths=5,
    th_width_to_kernel_ratio_largest=1,
    th_max_abs_fitness=1,
    th_sum_abs_fitness=2,
    th_locus_prominence=10,
    th_added_events=15,
    prominence_calc_on='conv',
    perform_prominence_overlap_check=False
    ):
    from spice.tsg_og.loci import calc_prominence, prominence_overlap_check

    cur_fitness = np.array([[x[0].fitness for x in ls] for ls in final_selection_points]).T

    # Filter based on ratio of width to kernel size (of largest nonzero LS)
    largest_nonzero_ls_i = 7 - np.argmax(cur_fitness[:, ::-1]!=0, axis=1)
    largest_nonzero_ls = [{
        0: 'small', 1: 'small', 2: 'mid1', 3: 'mid1', 4: 'mid2', 5: 'mid2', 6: 'large', 7: 'large'
        }[x] for x in largest_nonzero_ls_i]
    largest_nonzero_ls_kernel_size = [
        data_per_length_scale[(x, 'gain')]['loci_width'] * DEFAULT_SEGMENT_SIZE_DICT[x]
        for x in largest_nonzero_ls]
    width_to_kernel_ratio_largest = nr_stds_widths * np.array([np.std(x) for x in loci_widths]) / largest_nonzero_ls_kernel_size
    log_debug(logger, f'Removing loci based on width to kernel ratio (largest nonzero LS): {np.sum(width_to_kernel_ratio_largest > th_width_to_kernel_ratio_largest)} out of {len(width_to_kernel_ratio_largest)} loci')

    # Filter based on max abs fitness
    max_abs_fitness = np.max(np.abs(cur_fitness), axis=1)
    log_debug(logger, f'Removing loci based on max abs fitness: {np.sum(max_abs_fitness < th_max_abs_fitness)} out of {len(max_abs_fitness)} loci{". Is skipped in first iteration." if cur_iteration is not None and cur_iteration==0 else ""}')

    # Filter based on sum of abs fitness
    sum_abs_fitness = np.sum(np.abs(cur_fitness), axis=1)
    log_debug(logger, f'Removing loci based on sum of abs fitness: {np.sum(sum_abs_fitness < th_sum_abs_fitness)} out of {len(sum_abs_fitness)} loci{". Is skipped in first iteration." if cur_iteration is not None and cur_iteration==0 else ""}')

    ## Filter based on locus prominence
    cur_loci_df = calc_prominence(
        cur_chrom, data_per_length_scale, selection_points=final_selection_points, loci_widths=loci_widths, calc_on=prominence_calc_on)
    assert not cur_loci_df['max_prominence'].isna().any(), "NaN values in max_prominence, cannot filter based on locus prominence! Check the calc_prominence function."
    locus_prominence_bool = cur_loci_df['max_prominence'].values > th_locus_prominence
    if perform_prominence_overlap_check:
        locus_prominence_has_overlap = prominence_overlap_check(cur_loci_df, data_per_length_scale, th_locus_prominence)
        locus_prominence_bool = np.logical_or(locus_prominence_bool, ~locus_prominence_has_overlap)
    log_debug(logger, f'Removing loci based on locus prominence: {np.sum(~locus_prominence_bool)} out of {len(locus_prominence_bool)} loci')

    ## Filter based on added events
    added_events = calc_total_events_per_loci(
        cur_chrom,
        final_events_df=final_events_df,
        cur_selection_points=final_selection_points
    )
    total_added_events = np.stack([x[:-1] for x in added_events.values()]).sum(axis=0)
    log_debug(logger, f'Removing loci based on added events: {np.sum(total_added_events < th_added_events)} out of {len(total_added_events)} loci')

    # Combine all filters
    loci_to_keep_bool = reduce(np.logical_and, [
        width_to_kernel_ratio_largest <= th_width_to_kernel_ratio_largest,
        max_abs_fitness >= (0 if cur_iteration is not None and cur_iteration==0 else th_max_abs_fitness),
        sum_abs_fitness >= (0 if cur_iteration is not None and cur_iteration==0 else th_sum_abs_fitness),
        total_added_events >= th_added_events,
        locus_prominence_bool
    ])
    loci_to_keep = np.where(loci_to_keep_bool)[0]
    loci_to_remove = np.where(~loci_to_keep_bool)[0]

    return loci_to_keep_bool, loci_to_keep, loci_to_remove


def add_loci_one_by_one(
    cur_chrom, 
    raw_selection_points, 
    data_per_length_scale,
    N_iterations_base=3_000,
    segment_size_dict=DEFAULT_SEGMENT_SIZE_DICT,
    show_progress=False
):
    """
    Adds loci one by one to evaluate the effect of each additional locus.
    
    Parameters:
    -----------
    cur_chrom : str
        Chromosome to process
    raw_selection_points : list
        List containing selection points for each length scale
    data_per_length_scale : list
        List containing data per length scale
    segment_size_dict : dict
        Dictionary of segment sizes for different length scales
    output_dir : str, optional
        Directory to save output file
    filename : str, optional
        Name of the output file
    max_fitness : float
        Maximum fitness value allowed during optimization
    show_progress : bool
        Whether to display progress bar

    Returns:
    --------
    dict
        Dictionary containing results for each number of loci
    """
    log_debug(logger, f'Adding loci one by one for {cur_chrom}')
    
    assert all([x['chrom']==cur_chrom for x in data_per_length_scale.values()]), f'Wrong data_per_length_scale for current chrom {cur_chrom}'
    
    up_down_order = [any([raw_selection_points[j][cluster_j][0].fitness > 0 for j in range(0, 8, 2)])
                        for cluster_j in range(len(raw_selection_points[0]))]

    loci_one_by_one = []
    best_selection_points = None
    empty_conv = convolution_simulation_per_ls(cur_chrom, data_per_length_scale, best_selection_points)
    empty_loss = calc_mse_loss(data_per_length_scale, empty_conv)
    best_loss = empty_loss
    
    for N_loci in range(len(raw_selection_points[0])+1):
        log_debug(logger, f'Adding locus nr {N_loci} (out of {len(raw_selection_points[0])}) for {cur_chrom}')
        if N_loci > 0:
            new_selection_points = [[x[N_loci-1]] for x in raw_selection_points]
            if best_selection_points is None:
                cur_selection_points = copy_list_of_selection_points(new_selection_points)
            else:
                cur_selection_points = copy_list_of_selection_points([list(x) + list(y) for x, y in zip(best_selection_points, new_selection_points)])
            for i in range(8):
                cur_selection_points[i][-1] = SelectionPoints(loci=[[cur_selection_points[i][-1][0].pos, 0]])
            cur_selection_points_per_cluster = list(zip(*cur_selection_points))
            assert len(cur_selection_points_per_cluster) == N_loci, f'Expected {N_loci} clusters, got {len(cur_selection_points_per_cluster)}'

            base_conv = convolution_simulation_per_ls( cur_chrom, data_per_length_scale, cur_selection_points)
            base_loss = calc_mse_loss(data_per_length_scale, base_conv)
            assert base_loss == best_loss, f'Current loss {base_loss:.4e} is different from best loss {best_loss:.4e}'

            N_iterations = int(N_iterations_base * np.sqrt(N_loci + 1))
            optimized_selection_points_per_cluster, optim_loss, loss_over_time = _optimize_selection_points(
                N_iterations, 
                cur_selection_points_per_cluster, 
                data_per_length_scale, 
                cur_chrom,
                best_loss=base_loss, 
                show_progress=False, 
                N_iterations_base=0, 
                segment_size_dict=segment_size_dict,
                max_pos_change=0,
                up_down_order=up_down_order[:N_loci],
                allow_pos_change=False
            )

            if optim_loss < best_loss and len(loss_over_time) > 0:
                best_selection_points = list(zip(*optimized_selection_points_per_cluster))
                best_loss = optim_loss
            else:
                logger.warning(f'Optimization did not improve the loss at added locus {N_loci} (optim_loss: {optim_loss:.4e}, best_loss: {best_loss:.4e}). Setting fitness to zero for the new locus.')
                best_selection_points = copy_list_of_selection_points(cur_selection_points)

        cur_conv = convolution_simulation_per_ls(cur_chrom, data_per_length_scale, best_selection_points)
        cur_loss = calc_mse_loss(data_per_length_scale, cur_conv)
        cur_within_ci = calc_within_ci_bootstrap(data_per_length_scale, cur_conv)

        cur_loci_one_by_one = {
            'selection_points': best_selection_points,
            'conv': cur_conv,
            'mse_loss': cur_loss,
            'within_ci': cur_within_ci,
        }
        loci_one_by_one.append(cur_loci_one_by_one)
        log_debug(logger, f'MSE loss: {cur_loss:.4e} - within CI: {np.mean(cur_within_ci):.2f}')

    log_debug(logger, f'Finished adding loci one by one for {cur_chrom}')
    return loci_one_by_one


def calc_acceptance(new_loss, current_loss, iteration, max_iter, T_schedule='min_max', T_init=None, max_deviation=None, min_deviation=None):
    if new_loss < current_loss:
        return True
    elif current_loss == np.inf and new_loss == np.inf:
        return False
    else:    
        if T_schedule == "init":
            cur_T = T_init * (1 - iteration / max_iter)
            acceptance_prob = np.exp(-((new_loss - current_loss)) / cur_T)
        elif T_schedule == "min_max":
            cur_deviation = 10**(np.log10(max_deviation) - iteration/max_iter*(np.log10(max_deviation) - np.log10(min_deviation)))
            acceptance_prob = np.exp(-((new_loss - current_loss) / current_loss / cur_deviation)) 
        else:
            raise ValueError(f"Invalid temperature schedule: {T_schedule}")
        
        return np_rng().uniform(0, 1) < acceptance_prob
