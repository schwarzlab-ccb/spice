"""Main loci detection pipeline for de-novo TSG/OG detection."""

import os
import sys
from io import StringIO
from typing import Dict, Tuple, List, Optional
import pandas as pd
import numpy as np

from spice import config, data_loaders
from spice.length_scales import DEFAULT_LENGTH_SCALE_BOUNDARIES
from spice.utils import open_pickle, save_pickle, CALC_NEW
from spice.logging import log_debug, get_logger
from spice.random_state import derive_seed, seed_task
from spice.tsg_og.detection import (
    collect_data_per_length_scale, detect_tsgs_ogs_for_all_length_scales, n_loci_from_spacing,
    rank_loci, within_ci_fitness_filter,
    flip_up_down_assignment, final_optimization_step, limiting_fitness, infer_loci_widths, merge_overlapping_loci,
    calc_mse_loss, filter_loci, _optimize_selection_points, SelectionPoints)
from spice.tsg_og.signal_bootstrap import bootstrap_sampling_of_signal
from spice.tsg_og.simulation import copy_list_of_selection_points, convolution_simulation_per_ls
from spice.loci_preprocessing import process_final_events_for_loci_routines
from spice.tsg_og.loci import (
    create_loci_df, assign_p_values, calculate_events_per_loci_df)
from spice.tsg_og.permutation import fitness_statistic

if sys.version_info >= (3, 9):
    from importlib.resources import files
else:
    try:
        from importlib_resources import files
    except ImportError:
        files = None

logger = get_logger('loci_detection_main')
CHROMS = ['chr' + str(x) for x in range(1, 23)] + ['chrX', 'chrY']

#: Stages whose persisted pickle holds the producing function's WHOLE return tuple, while the
#: in-memory RESULTS entry is only its first element (the per-length-scale selection points):
#:   detection               -> (selection_points, _, _)
#:   optimizing_intermediate -> (selection_points, all_losses)      [final_optimization_step]
#:   merging                 -> (selection_points, conv, removed, to_remove)
#:   optimizing              -> (selection_points, _)               [final_optimization_step]
#: CALC_NEW pickles the return value, so a run that RESUMES one of these from disk must unwrap it;
#: a run that reaches it in the same process never does, because the assignment already unpacked.
#: Missing the unwrap does not fail where it happens -- the next stage receives the 8-element outer
#: tuple where it expects the tracks, and dies confusingly, e.g. `spice loci_detection --chrom chr21
#: --loci-steps filter_loci_intermediate_1` reporting "Number of locus widths (16) does not match
#: number of selection points (8)", the 8 being the number of LENGTH SCALES rather than loci.
_STAGES_PICKLED_AS_TUPLE = frozenset({'detection', 'optimizing_intermediate', 'merging', 'optimizing'})


def _load_stage(output_dir, filenames, stage):
    """Load one persisted detection stage, unwrapping the stages saved as a return tuple."""
    obj = open_pickle(os.path.join(output_dir, filenames[stage]))
    return obj[0] if stage in _STAGES_PICKLED_AS_TUPLE else obj


def run_loci_detection_per_chrom(
    final_events_df,
    cur_chrom,
    which='full',
    name=None,
    N_loci=100,
    N_loci_spacing=None,
    overwrite=False,
    overwrite_preprocessing=False,
    loci_results_dir=None,
    skip_up_down=False,
    use_original_rank=False,
    length_scales_for_residuals='01234567',
    N_bootstrap=1_000,
    N_kernel=100_000,
    detection_N_iterations_base=3000,
    detection_max_N_iterations=20_000,
    detection_final_N_iterations=250_000,
    detection_blocked_distance_th=2e5,
    ranking_N_iterations=500,
    flipping_N_iterations=11_000,
    flipping_N_iterations_single=1_000,
    limiting_N_iterations_optim=10_000,
    within_ci_N_iterations=10_000,
    optimizing_N_iterations_optimization=11_000,
    infer_widths_N_iterations=1_000,
    merge_N_iterations_optim=10_000,
    filter_N_iterations_optim=100_000,
    final_limiting_N_iterations_optim=10_000,
    N_bootstrap_for_widths=200,
    th_locus_prominence=5,
    th_locus_mean_fitness=1
):
    """
    Run the loci detection pipeline for a given chromosome.
    
    Parameters
    ----------
    cur_chrom : str
        Chromosome to analyze
    which : str, default='full'
        Which steps to run: 'full', single step, or comma-separated steps
    name : str, optional
        Project name (from config if not provided)
    N_loci : int, default=100
        Number of loci to detect
    N_loci_spacing : float or None, default=None
        If set, seed one locus per this many bp of SEARCHABLE sequence (padded telomere-to-telomere
        span minus the padded centromere) instead of a flat `N_loci`, via
        `detection.n_loci_from_spacing`. Makes the candidate density comparable across chromosomes;
        a flat count seeds chr21's 29 Mb as densely as chr1's 218 Mb. Overrides `N_loci`.
    loci_results_dir : str, optional
        Output directory (auto-generated if not provided)
    overwrite_preprocessing : bool, default=False
        Force recalculation of preprocessing caches (bootstrap signals and data_per_length_scale)
    skip_up_down : bool, default=False
        Skip up/down assignment
    use_original_rank : bool, default=False
        Use original rank from detection instead of re-ranking
    length_scales_for_residuals : str, default='01234567'
        Length scales to use for residuals
    detection_N_iterations_base : int, default=3000
        Base number of iterations for detection
    detection_max_N_iterations : int, default=20_000
        Maximum iterations for detection
    detection_final_N_iterations : int, default=250_000
        Final iterations for detection
    detection_blocked_distance_th : float, default=2e5
        Blocked distance threshold
    ranking_N_iterations : int, default=500
        Number of iterations for ranking
    th_locus_prominence : float, default=5
        Threshold for locus prominence filtering
    th_locus_mean_fitness : float, default=1
        Threshold for the mean directed fitness post-processing filter
    """
    
    # Each stochastic stage below also gets its own stream: preceding stages and
    # preprocessing may run, load from cache, or be skipped during a resumed run.
    seed_task(derive_seed('loci_detection', cur_chrom))

    # Define all available steps
    which_options = [
        'detection',
        'flipping',
        'ranking',
        'within_ci_filtering',
        'limiting',
        'optimizing_intermediate',
        'loci_widths_intermediate',
        'merging',
        'optimizing',
        'loci_widths_intermediate_2',
        'filter_loci_intermediate_1',
        'final_within_ci_filtering',
        'final_filter_loci',
        'final_limiting',
        'final_loci_widths',
        # 'one_by_one'
    ]

    which_fast = [
        'detection',
        'flipping',
        'optimizing',
        'final_within_ci_filtering',
        'final_filter_loci',
        'final_loci_widths',
    ]
    
    # Parse which steps to run
    if hasattr(which, '__iter__') and len(which)==1:
        which = which[0]
    if isinstance(which, str):
        if which == 'full':
            which_steps = which_options
        elif which == 'fast':
            which_steps = which_fast
        elif '+' in which:
            which_start = which[:which.find('+')]
            assert which_start in which_options, f'Invalid option {which_start} in {which}'
            which_steps = which_options[which_options.index(which_start):]
        elif which in which_options:
            which_steps = [which]
        else:
            raise ValueError(f"Unknown which mode: {which}. Use 'full', 'fast', or a step name (with or without '+')")
    else:
        assert hasattr(which, '__iter__'), which
        which_steps = which
    
    # Resolve name and directories
    name = name if name is not None else config['name']
    
    output_dir = os.path.join(loci_results_dir, 'detection', cur_chrom)
       
    if N_loci_spacing:
        # One locus per N_loci_spacing bp of searchable sequence, derived from the same blocked
        # regions the residual search uses (see detection.n_loci_from_spacing).
        N_loci_flat, N_loci = N_loci, n_loci_from_spacing(
            cur_chrom, N_loci_spacing, blocked_distance_th=detection_blocked_distance_th)
        logger.info(f'N_loci from spacing: one locus per {N_loci_spacing/1e6:g} Mb of searchable '
                    f'sequence on {cur_chrom} -> {N_loci} loci (flat N_loci={N_loci_flat} ignored)')
    logger.info(f'Running loci detection for chrom={cur_chrom}, name={name} and a maximum of {N_loci} loci.')
    logger.info(f'Steps to run: {" - ".join(which_steps)}')
    logger.info(f'Output will be saved to {output_dir}')
    
    # Parse length scales
    length_scales_for_residuals = [int(x) for x in length_scales_for_residuals]
    
    # Create filename template
    filenames = {w: f'{w}.pickle' for w in which_options + ['final_selection_points']}

    # Calculate bootstrap signals before loading data per length scale
    logger.info(f'Calculating bootstrap signals for {cur_chrom}')
    bootstrap_sampling_of_signal(
        cur_chrom=cur_chrom,
        final_events_df=final_events_df,
        N_bootstrap=N_bootstrap,
        calc_new_force_new=overwrite_preprocessing,
        calc_new_filename=os.path.join(
            loci_results_dir, 'signal_bootstrap', f'{cur_chrom}_N_{N_bootstrap}.pickle'))
    
    # Load relevant data
    data_per_length_scale = collect_data_per_length_scale(
        final_events_df, cur_chrom, N_bootstrap=N_bootstrap, N_kernel=N_kernel, loci_results_dir=loci_results_dir,
        calc_new_force_new=overwrite_preprocessing,
        calc_new_filename=os.path.join(loci_results_dir, 'data_per_length_scale', f'{cur_chrom}.pickle'))

    # Initialize results dictionary
    RESULTS = {w: None for w in which_options}
    RESULTS['final_selection_points'] = None
    
    # Detection step
    if 'detection' in which_steps:
        seed_task(derive_seed('loci_detection', cur_chrom, 'detection'))
        logger.info(f'Running detection')
        log_debug(logger, f'Output: {output_dir}/{filenames["detection"]}')
        
        RESULTS['detection'], _, _ = detect_tsgs_ogs_for_all_length_scales(
            cur_chrom=cur_chrom,
            blocked_distance_th=detection_blocked_distance_th,
            force_up_down=not skip_up_down,
            N_iterations_base=detection_N_iterations_base,
            max_N_iterations=detection_max_N_iterations,
            final_N_iterations=detection_final_N_iterations,
            N_loci=N_loci,
            max_fitness=1_000,
            length_scales_for_residuals=length_scales_for_residuals,
            data_per_length_scale=data_per_length_scale,
            calc_new_force_new=overwrite,
            calc_new_filename=os.path.join(output_dir, filenames['detection']))
    
    # Flipping step
    if 'flipping' in which_steps:
        seed_task(derive_seed('loci_detection', cur_chrom, 'flipping'))
        logger.info(f'Running flipping')
        log_debug(logger, f'Output: {output_dir}/{filenames["flipping"]}')
        
        if RESULTS['detection'] is None:
            RESULTS['detection'] = _load_stage(output_dir, filenames, 'detection')
        
        RESULTS['flipping'] = flip_up_down_assignment(
            cur_chrom=cur_chrom,
            final_selection_points=RESULTS['detection'],
            data_per_length_scale=data_per_length_scale,
            n_neighbors=10,
            N_iterations=flipping_N_iterations,
            N_iterations_single=flipping_N_iterations_single,
            calc_new_force_new=overwrite,
            calc_new_filename=os.path.join(output_dir, filenames['flipping']))
    
    # Ranking step
    if 'ranking' in which_steps:
        seed_task(derive_seed('loci_detection', cur_chrom, 'ranking'))
        logger.info(f'Running ranking')
        log_debug(logger, f'Output: {output_dir}/{filenames["ranking"]}')
        
        if RESULTS['flipping'] is None:
            RESULTS['flipping'] = _load_stage(output_dir, filenames, 'flipping')
        
        if use_original_rank:
            logger.info(f'Using original rank from detection. Skipping rank_loci() function.')
            RESULTS['ranking'] = copy_list_of_selection_points(RESULTS['flipping'])
        else:
            ranking_locus_iterations = rank_loci(
                cur_chrom=cur_chrom,
                best_selection_points=RESULTS['flipping'],
                data_per_length_scale=data_per_length_scale,
                show_progress=False,
                log_progress=True,
                force_up_down=not skip_up_down,
                max_n_clusters=None,
                N_iterations=ranking_N_iterations,
                n_cores=-1,
                max_fitness=1_000,
                calc_new_force_new=overwrite,
                calc_new_filename=os.path.join(output_dir, filenames['ranking']))
            RESULTS['ranking'] = ranking_locus_iterations[-1][0]
    
    # Within CI filtering step
    if 'within_ci_filtering' in which_steps:
        seed_task(derive_seed('loci_detection', cur_chrom, 'within_ci_filtering'))
        logger.info(f'Running within_ci_filtering')
        log_debug(logger, f'Output: {output_dir}/{filenames["within_ci_filtering"]}')
        
        if RESULTS['ranking'] is None:
            if use_original_rank:
                if RESULTS['flipping'] is None:
                    RESULTS['flipping'] = _load_stage(output_dir, filenames, 'flipping')
                RESULTS['ranking'] = copy_list_of_selection_points(RESULTS['flipping'])
            else:
                ranking_locus_iterations = open_pickle(os.path.join(output_dir, filenames['ranking']))
                RESULTS['ranking'] = ranking_locus_iterations[-1][0]
        
        RESULTS['within_ci_filtering'] = within_ci_fitness_filter(
            cur_chrom=cur_chrom,
            ranked_selection_points=RESULTS['ranking'],
            data_per_length_scale=data_per_length_scale,
            show_progress=False,
            log_progress=True,
            N_iterations_optimization=within_ci_N_iterations,
            calc_new_force_new=overwrite,
            calc_new_filename=os.path.join(output_dir, filenames['within_ci_filtering']))
    
    # Limiting step
    if 'limiting' in which_steps:
        seed_task(derive_seed('loci_detection', cur_chrom, 'limiting'))
        logger.info(f'Running limiting')
        log_debug(logger, f'Output: {output_dir}/{filenames["limiting"]}')
        
        if RESULTS['within_ci_filtering'] is None:
            RESULTS['within_ci_filtering'] = _load_stage(output_dir, filenames, 'within_ci_filtering')
        
        RESULTS['limiting'] = limiting_fitness(
            cur_chrom=cur_chrom,
            raw_selection_points=RESULTS['within_ci_filtering'],
            data_per_length_scale=data_per_length_scale,
            max_iterations=15,
            allow_all_fitness_change=True,
            N_iterations_optim=limiting_N_iterations_optim,
            max_deviation=0.0001,
            blocked_distance_th=2e5,
            show_progress=False,
            loss_threshold=0.25,
            within_ci_threshold=0.025,
            calc_new_force_new=overwrite,
            calc_new_filename=os.path.join(output_dir, filenames['limiting']))
    
    # Optimizing intermediate step
    if 'optimizing_intermediate' in which_steps:
        seed_task(derive_seed('loci_detection', cur_chrom, 'optimizing_intermediate'))
        logger.info(f'Running optimizing_intermediate')
        log_debug(logger, f'Output: {output_dir}/{filenames["optimizing_intermediate"]}')
        
        if RESULTS['limiting'] is None:
            RESULTS['limiting'] = _load_stage(output_dir, filenames, 'limiting')
        
        RESULTS['optimizing_intermediate'], all_losses = final_optimization_step(
            cur_chrom=cur_chrom,
            final_selection_points=RESULTS['limiting'],
            data_per_length_scale=data_per_length_scale,
            n_neighbors_optimization=10,
            N_iterations_optimization=optimizing_N_iterations_optimization,
            max_pos_change=1e5,
            calc_new_force_new=overwrite,
            calc_new_filename=os.path.join(output_dir, filenames['optimizing_intermediate']))
    
    # locus widths intermediate step
    if 'loci_widths_intermediate' in which_steps:
        seed_task(derive_seed('loci_detection', cur_chrom, 'loci_widths_intermediate'))
        logger.info(f'Running loci_widths_intermediate')
        log_debug(logger, f'Output: {output_dir}/{filenames["loci_widths_intermediate"]}')
        
        if RESULTS['optimizing_intermediate'] is None:
            RESULTS['optimizing_intermediate'] = _load_stage(output_dir, filenames, 'optimizing_intermediate')
        
        RESULTS['loci_widths_intermediate'] = infer_loci_widths(
            cur_chrom=cur_chrom,
            final_selection_points=RESULTS['optimizing_intermediate'],
            loci_results_dir=loci_results_dir,
            data_per_length_scale=data_per_length_scale,
            num_bootstrap_iterations=N_bootstrap_for_widths,
            max_pos_change=1e5,
            max_deviation=0.00001,
            N_bootstrap=N_bootstrap,
            num_optimization_iterations=infer_widths_N_iterations,
            n_jobs=-1,
            calc_new_force_new=overwrite,
            calc_new_filename=os.path.join(output_dir, filenames['loci_widths_intermediate']))
    
    # Merging step
    if 'merging' in which_steps:
        seed_task(derive_seed('loci_detection', cur_chrom, 'merging'))
        logger.info(f'Running merging')
        log_debug(logger, f'Output: {output_dir}/{filenames["merging"]}')
        
        if RESULTS['optimizing_intermediate'] is None:
            RESULTS['optimizing_intermediate'] = _load_stage(output_dir, filenames, 'optimizing_intermediate')
        
        if RESULTS['loci_widths_intermediate'] is None:
            RESULTS['loci_widths_intermediate'] = _load_stage(output_dir, filenames, 'loci_widths_intermediate')
        
        RESULTS['merging'], merged_conv, removed_loci, loci_to_remove = merge_overlapping_loci(
            cur_chrom=cur_chrom,
            selection_points=RESULTS['optimizing_intermediate'],
            loci_widths=RESULTS['loci_widths_intermediate'],
            data_per_length_scale=data_per_length_scale,
            n_iterations_optim=merge_N_iterations_optim,
            show_progress_optim=False,
            max_deviation_optim=0.00001,
            calc_new_force_new=overwrite,
            calc_new_filename=os.path.join(output_dir, filenames['merging']))
    
    # Optimizing step
    if 'optimizing' in which_steps:
        seed_task(derive_seed('loci_detection', cur_chrom, 'optimizing'))
        logger.info(f'Running optimizing')
        log_debug(logger, f'Output: {output_dir}/{filenames["optimizing"]}')
        
        input_source = 'flipping' if which == 'fast' else 'merging'

        if RESULTS[input_source] is None:
            RESULTS[input_source] = _load_stage(output_dir, filenames, input_source)
   
        RESULTS['optimizing'], _ = final_optimization_step(
            cur_chrom=cur_chrom,
            final_selection_points=RESULTS[input_source],
            data_per_length_scale=data_per_length_scale,
            n_neighbors_optimization=10,
            N_iterations_optimization=optimizing_N_iterations_optimization,
            max_pos_change=1e5,
            calc_new_force_new=overwrite,
            calc_new_filename=os.path.join(output_dir, filenames['optimizing']))
    
    # locus widths intermediate 2 step
    if 'loci_widths_intermediate_2' in which_steps:
        seed_task(derive_seed('loci_detection', cur_chrom, 'loci_widths_intermediate_2'))
        logger.info(f'Running loci_widths_intermediate_2')
        log_debug(logger, f'Output: {output_dir}/{filenames["loci_widths_intermediate_2"]}')
        
        if RESULTS['optimizing'] is None:
            RESULTS['optimizing'] = _load_stage(output_dir, filenames, 'optimizing')
        
        RESULTS['loci_widths_intermediate_2'] = infer_loci_widths(
            cur_chrom=cur_chrom,
            final_selection_points=RESULTS['optimizing'],
            loci_results_dir=loci_results_dir,
            data_per_length_scale=data_per_length_scale,
            num_bootstrap_iterations=N_bootstrap_for_widths,
            max_pos_change=1e5,
            max_deviation=0.00001,
            N_bootstrap=N_bootstrap,
            num_optimization_iterations=infer_widths_N_iterations,
            n_jobs=-1,
            calc_new_force_new=overwrite,
            calc_new_filename=os.path.join(output_dir, filenames['loci_widths_intermediate_2']))
    
    # Filter loci intermediate 1 step
    if 'filter_loci_intermediate_1' in which_steps:
        seed_task(derive_seed('loci_detection', cur_chrom, 'filter_loci_intermediate_1'))
        logger.info(f'Running filter_loci_intermediate_1')
        log_debug(logger, f'Output: {output_dir}/{filenames["filter_loci_intermediate_1"]}')
        
        if RESULTS['loci_widths_intermediate_2'] is None:
            RESULTS['loci_widths_intermediate_2'] = _load_stage(output_dir, filenames, 'loci_widths_intermediate_2')
        if RESULTS['optimizing'] is None:
            RESULTS['optimizing'] = _load_stage(output_dir, filenames, 'optimizing')
        
        RESULTS['filter_loci_intermediate_1'] = filter_loci(
            cur_chrom=cur_chrom,
            final_selection_points=RESULTS['optimizing'],
            loci_widths=RESULTS['loci_widths_intermediate_2'],
            data_per_length_scale=data_per_length_scale,
            final_events_df=final_events_df,
            n_iterations_optim=filter_N_iterations_optim,
            show_progress_optim=False,
            max_deviation_optim=0.00001,
            # Both thresholds must be passed here as well as to final_filter_loci below: without
            # them this call silently used filter_loci's own signature defaults, so configuring
            # either key changed only the final stage. That is the smaller one -- on TCGA this
            # intermediate filter culls 1896 -> 989 loci against final_filter_loci's 989 -> 928, so
            # the un-plumbed stage governed ~15x as many loci as the configurable one.
            th_locus_prominence=th_locus_prominence,
            th_locus_mean_fitness=th_locus_mean_fitness,
            calc_new_force_new=overwrite,
            calc_new_filename=os.path.join(output_dir, filenames['filter_loci_intermediate_1']))
    
    # Final within CI filtering step
    if 'final_within_ci_filtering' in which_steps:
        seed_task(derive_seed('loci_detection', cur_chrom, 'final_within_ci_filtering'))
        logger.info(f'Running final_within_ci_filtering')
        log_debug(logger, f'Output: {output_dir}/{filenames["final_within_ci_filtering"]}')
        
        input_source = 'optimizing' if which == 'fast' else 'filter_loci_intermediate_1'
        
        if RESULTS[input_source] is None:
            RESULTS[input_source] = _load_stage(output_dir, filenames, input_source)
        
        RESULTS['final_within_ci_filtering'] = within_ci_fitness_filter(
            cur_chrom=cur_chrom,
            ranked_selection_points=RESULTS[input_source],
            data_per_length_scale=data_per_length_scale,
            remove_empty_loci=False,
            show_progress=False,
            log_progress=True,
            N_iterations_optimization=within_ci_N_iterations,
            calc_new_force_new=overwrite,
            calc_new_filename=os.path.join(output_dir, filenames['final_within_ci_filtering']))
    
    # Final filter loci step
    if 'final_filter_loci' in which_steps:
        seed_task(derive_seed('loci_detection', cur_chrom, 'final_filter_loci'))
        logger.info(f'Running final_filter_loci')
        log_debug(logger, f'Output: {output_dir}/{filenames["final_filter_loci"]}')
        
        if RESULTS['final_within_ci_filtering'] is None:
            RESULTS['final_within_ci_filtering'] = _load_stage(output_dir, filenames, 'final_within_ci_filtering')
        
        RESULTS['final_filter_loci'] = filter_loci(
            cur_chrom=cur_chrom,
            final_selection_points=RESULTS['final_within_ci_filtering'],
            loci_widths=None,
            data_per_length_scale=data_per_length_scale,
            final_events_df=final_events_df,
            n_iterations_optim=filter_N_iterations_optim,
            show_progress_optim=False,
            max_deviation_optim=0.00001,
            th_locus_prominence=th_locus_prominence,
            th_locus_mean_fitness=th_locus_mean_fitness,
            calc_new_force_new=overwrite,
            calc_new_filename=os.path.join(output_dir, filenames['final_filter_loci']))

    # Final limiting step
    if 'final_limiting' in which_steps:
        seed_task(derive_seed('loci_detection', cur_chrom, 'final_limiting'))
        logger.info(f'Running final_limiting')
        log_debug(logger, f'Output: {output_dir}/{filenames["final_limiting"]}')
        
        if RESULTS['final_filter_loci'] is None:
            RESULTS['final_filter_loci'] = _load_stage(output_dir, filenames, 'final_filter_loci')
        
        RESULTS['final_limiting'] = limiting_fitness(
            cur_chrom=cur_chrom,
            raw_selection_points=RESULTS['final_filter_loci'],
            data_per_length_scale=data_per_length_scale,
            max_iterations=10,
            allow_all_fitness_change=True,
            N_iterations_optim=final_limiting_N_iterations_optim,
            max_deviation=0.0001,
            blocked_distance_th=2e5,
            show_progress=False,
            loss_threshold=0.125,
            within_ci_threshold=0.01,
            ls_i_to_check=(6, 7),
            calc_new_force_new=overwrite,
            calc_new_filename=os.path.join(output_dir, filenames['final_limiting']))
        
        RESULTS['final_selection_points'] = copy_list_of_selection_points(RESULTS['final_limiting'])
        save_pickle(RESULTS['final_selection_points'], os.path.join(output_dir, filenames['final_selection_points']))
    
    # In fast mode, final_limiting is skipped, so set final_selection_points from final_filter_loci
    else:
        if 'final_filter_loci' in which_steps and 'final_limiting' not in which_steps:
            if RESULTS['final_filter_loci'] is None:
                RESULTS['final_filter_loci'] = _load_stage(output_dir, filenames, 'final_filter_loci')
            RESULTS['final_selection_points'] = copy_list_of_selection_points(RESULTS['final_filter_loci'])
            save_pickle(RESULTS['final_selection_points'], os.path.join(output_dir, filenames['final_selection_points']))


    # Final locus widths step
    if 'final_loci_widths' in which_steps:
        seed_task(derive_seed('loci_detection', cur_chrom, 'final_loci_widths'))
        logger.info(f'Running final_loci_widths')
        log_debug(logger, f'Output: {output_dir}/{filenames["final_loci_widths"]}')
        
        if RESULTS['final_selection_points'] is None:
            RESULTS['final_selection_points'] = _load_stage(output_dir, filenames, 'final_selection_points')
        
        RESULTS['final_loci_widths'] = infer_loci_widths(
            cur_chrom=cur_chrom,
            final_selection_points=RESULTS['final_selection_points'],
            loci_results_dir=loci_results_dir,
            data_per_length_scale=data_per_length_scale,
            num_bootstrap_iterations=N_bootstrap_for_widths,
            max_pos_change=1e5,
            max_deviation=0.00001,
            N_bootstrap=N_bootstrap,
            num_optimization_iterations=infer_widths_N_iterations,
            n_jobs=-1,
            calc_new_force_new=overwrite,
            calc_new_filename=os.path.join(output_dir, filenames['final_loci_widths']))
    
    # # One by one step
    # if 'one_by_one' in which_steps:
    #     logger.info(f'Running one_by_one')
    #     log_debug(logger, f'Output: {output_dir}/{filenames["one_by_one"]}')
        
    #     if RESULTS['final_selection_points'] is None:
    #         RESULTS['final_selection_points'] = _load_stage(output_dir, filenames, 'final_selection_points')
        
    #     RESULTS['one_by_one'] = add_loci_one_by_one(
    #         cur_chrom=chrom,
    #         raw_selection_points=RESULTS['final_selection_points'],
    #         data_per_length_scale=data_per_length_scale,
    #         show_progress=False)
    
    logger.info(f'Done! Loci detection for chrom={cur_chrom}, name={name}')
    
    return RESULTS


def cached_loci_chromosomes(loci_results_dir):
    """Chromosome caches consumed by combination, independent of retained events."""
    return [chrom for chrom in CHROMS[:-1]
            if os.path.exists(os.path.join(loci_results_dir, 'data_per_length_scale', f'{chrom}.pickle'))]


@CALC_NEW()
def combine_loci(
    loci_results_dir: str,
    processed_events: Optional[pd.DataFrame] = None,
    calculate_p_value: bool = False,
    p_value_threshold: float = 0.05,
    mean_fitness_threshold: Optional[float] = None,
    permutation_null: Optional[pd.DataFrame] = None,
    p_values_strategy: str = 'zpool',
    overwrite: bool = False,
    mode: str = 'detection',
    final_reoptimization_N_iterations: int = 100_000,
) -> Tuple[pd.DataFrame, Dict, Dict, pd.DataFrame]:
    """
    Combine results from all chromosomes after loci detection or assignment
    
    Loads final selection points and peak widths from all chromosomes. When
    calculate_p_value=True, scores and filters loci, then refits only chromosomes
    where filtering removed loci. Chromosomes retaining every locus keep their fit.
    
    Parameters
    ----------
    loci_results_dir : str
        Directory where per-chromosome results are stored
    final_events_df : pd.DataFrame, optional
        Final events dataframe. If not provided, will be loaded from config.
    
    Returns
    -------
    pd.DataFrame
        Combined and scored loci dataframe across all chromosomes
    """

    # Load selection points and peak widths from all chromosomes
    all_selection_points = {}
    all_loci_widths = {}
    all_data_per_length_scale = {}

    for cur_chrom in cached_loci_chromosomes(loci_results_dir):
        data_per_length_scale_file = os.path.join(loci_results_dir, 'data_per_length_scale', f'{cur_chrom}.pickle')
        logger.info(f"Loading results for {cur_chrom}")
        
        # Load final selection points
        final_selection_points_file = os.path.join(loci_results_dir, mode, cur_chrom, 'final_selection_points.pickle')
        if not os.path.exists(final_selection_points_file):
            logger.error(f"Missing final selection points for {cur_chrom} at {final_selection_points_file}")
            raise FileNotFoundError(f"Expected {final_selection_points_file}")
        
        final_selection_points = open_pickle(final_selection_points_file)
        all_selection_points[cur_chrom] = final_selection_points
        
        # Load peak widths
        peak_widths_file = os.path.join(loci_results_dir, mode, cur_chrom, 'final_loci_widths.pickle')
        if not os.path.exists(peak_widths_file):
            logger.error(f"Missing peak widths for {cur_chrom} at {peak_widths_file}")
            raise FileNotFoundError(f"Expected {peak_widths_file}")
        
        peak_widths = open_pickle(peak_widths_file)
        all_loci_widths[cur_chrom] = peak_widths
        
        # Validation
        assert len(peak_widths) == len(final_selection_points[0]), (
            f'{cur_chrom}: length mismatch - {len(peak_widths)} peak widths vs '
            f'{len(final_selection_points[0])} selection points'
        )

        all_data_per_length_scale[cur_chrom] = open_pickle(data_per_length_scale_file)
        
        log_debug(logger, f"  ✓ {cur_chrom}: {len(peak_widths)} loci")
    
    log_debug(logger, f'Loaded results from {len(all_selection_points)} chromosomes')
    
    # Run build_and_score_loci to create final loci dataframe
    log_debug(logger, "Building and scoring loci dataframe")
    loci_df = build_final_loci_df(
        all_selection_points=all_selection_points,
        all_loci_widths=all_loci_widths,
        final_events_df=processed_events
    )
    
    if calculate_p_value:
        from spice.length_scales import LENGTH_SCALE_NAMES
        if permutation_null is None or not len(permutation_null):
            raise ValueError(
                'calculate_p_value=True needs a permutation null. Build one with `spice permute` '
                '(or let loci detection build it inline) -- see spice.tsg_og.permutation.')
        final_loci_df = assign_p_values(loci_df, permutation_null, strategy=p_values_strategy)
        # assign_p_values: p_value_raw = raw p, p_value = BH-FDR q. Remap to canonical raw p / FDR q.
        final_loci_df['q_value'] = final_loci_df['p_value']
        final_loci_df['p_value'] = final_loci_df.pop('p_value_raw')
        for ls in LENGTH_SCALE_NAMES:
            final_loci_df[f'q_value_{ls}'] = final_loci_df.pop(f'p_value_{ls}')      # BH-FDR per scale
            final_loci_df[f'p_value_{ls}'] = final_loci_df.pop(f'p_value_raw_{ls}')  # raw per scale
        # The table BEFORE any drop -- returned so the caller can persist it. The calibration figures
        # (QQ, cumulative, p-value histograms, length-scale coherence) need every detected locus with
        # its p/q; reading them off a table filtered to the significant subset measures the threshold
        # rather than the calibration.
        unfiltered_loci_df = final_loci_df.copy()

        # BOTH post-null drops, applied together so the selection points and widths stay in step with
        # the table. They are deliberately here and not in detection:
        #   q_value          -- significance against the permutation null.
        #   mean fitness     -- an ABSOLUTE floor on the same statistic the p-value ranks
        #                       (permutation.fitness_statistic). spice also has a detection-time
        #                       version of this, `th_locus_mean_fitness`; applying it THERE also
        #                       filters the permutation null, because the null is built by re-running
        #                       detection on permuted events, and a null exists to produce weak loci.
        #                       Measured on HMF: the null collapsed 13,964 -> 1,311 loci with 32 of 46
        #                       (chrom,direction) strata empty and nothing could be scored. Applied
        #                       here the null is intact and this selects among scored loci.
        n_before = len(final_loci_df)
        fit = fitness_statistic(final_loci_df)
        keep_all = (final_loci_df['q_value'].to_numpy() < p_value_threshold)
        if mean_fitness_threshold is not None:
            keep_all &= (fit > mean_fitness_threshold)
        final_loci_df = final_loci_df.assign(_keep=keep_all)
        filtered_selection_points = dict()
        filtered_loci_widths = dict()
        changed_chromosomes = set()
        for cur_chrom in list(all_selection_points.keys()):
            keep = (final_loci_df.query('chrom == @cur_chrom')
                                 .sort_values('rank_on_chrom')['_keep'].to_numpy())
            if not keep.all():
                changed_chromosomes.add(cur_chrom)
            filtered_selection_points[cur_chrom] = [
                [x for i, x in enumerate(track) if keep[i]] for track in all_selection_points[cur_chrom]]
            filtered_loci_widths[cur_chrom] = [
                x for i, x in enumerate(all_loci_widths[cur_chrom]) if keep[i]]
        final_loci_df = final_loci_df[final_loci_df['_keep']].drop(columns='_keep').reset_index(drop=True)

        # Dropping non-significant loci leaves the survivors' fitness stale for the reduced model.
        # Refit them (positions frozen) with the same per-locus-neighborhood optimizer as used before
        n_to_reoptimize = sum(len(filtered_selection_points[c][0]) for c in changed_chromosomes)
        if not changed_chromosomes:
            logger.info('All loci retained; keeping the original fit without reoptimization')
        elif n_to_reoptimize:
            logger.info(f'Reoptimizing fitness of {n_to_reoptimize} surviving loci after filtering '
                        f'({final_reoptimization_N_iterations} iterations/locus-neighborhood)')
        fitness_cols = [f'fitness_{ls}_{d}' for ls in LENGTH_SCALE_NAMES for d in ['gain', 'loss']]
        for cur_chrom, chrom_selection_points in filtered_selection_points.items():
            if cur_chrom not in changed_chromosomes or len(chrom_selection_points[0]) == 0:
                continue
            reoptimized_selection_points, _ = final_optimization_step(
                cur_chrom=cur_chrom,
                final_selection_points=chrom_selection_points,
                data_per_length_scale=all_data_per_length_scale[cur_chrom],
                n_neighbors_optimization=10,
                N_iterations_optimization=final_reoptimization_N_iterations,
                max_pos_change=1e5)
            filtered_selection_points[cur_chrom] = reoptimized_selection_points
            cur_index = final_loci_df.query('chrom == @cur_chrom').sort_values('rank_on_chrom').index
            final_loci_df.loc[cur_index, fitness_cols] = np.stack(
                [[locus[0].fitness for locus in ls_track] for ls_track in reoptimized_selection_points], axis=1)
        gain_cols = [f'fitness_{ls}_gain' for ls in LENGTH_SCALE_NAMES]
        final_loci_df['type'] = np.where((final_loci_df[gain_cols] > 0).any(axis=1), 'OG', 'TSG')

        cut = f'q_value < {p_value_threshold}' + (
            f' and mean fitness > {mean_fitness_threshold}' if mean_fitness_threshold is not None else '')
        logger.info(f'Assigned fitness p/q from the permutation null (global BH-FDR; p_value = raw, '
                    f'q_value = BH-FDR q) and kept {len(final_loci_df)}/{n_before} loci with {cut}')
    else:
        # Skip p-value filtering and use all loci. `final_loci_df` must still be bound here --
        # only the branch above promotes `loci_df` to it, so without this the shared log/return
        # below raised UnboundLocalError and `combine` was unusable whenever the p-value was off.
        # The path was dead until now: the CLI reads calculate_p_value from the config
        # (cli.py: calc_p) and passes it straight through, and the pipeline's loci.yaml always
        # sets it true, so only a config that turns the p-value off reaches this.
        logger.info('Skipping p-value filtering (calculate_p_value=False)')
        final_loci_df = loci_df
        unfiltered_loci_df = loci_df
        filtered_selection_points = all_selection_points
        filtered_loci_widths = all_loci_widths

    log_debug(logger, f'Final loci dataframe: {len(final_loci_df)} loci across {final_loci_df["chrom"].nunique()} chromosomes')   
    return final_loci_df, filtered_selection_points, filtered_loci_widths, unfiltered_loci_df



def run_loci_assignment_per_chrom(
    reference_loci_df: pd.DataFrame,
    cur_chrom: str,
    final_events_df: pd.DataFrame,
    loci_results_dir: str,
    N_bootstrap: int = 1_000,
    N_kernel: int = 100_000,
    within_ci_N_iterations: int = 10_000,
    N_iterations_optim: int = 11_000,
    overwrite: bool = False,
    overwrite_preprocessing: bool = False,
) -> Tuple[List, List]:
    """
    Run loci assignment for a single chromosome using provided loci positions.

    This function takes pre-defined loci positions and optimizes their fitness values
    by: 1) creating dummy selection points with zero fitness
    2) optimizing fitness with fixed positions
    3) filtering by CI constraints

    Parameters
    ----------
    reference_loci_df : pd.DataFrame
        DataFrame with columns: chrom, start, end, type (OG or TSG)
    cur_chrom : str
        Chromosome to process
    final_events_df : pd.DataFrame
        Final events dataframe for this sample
    loci_results_dir : str
        Directory to save results
    N_bootstrap : int
        Number of bootstrap samples
    N_kernel : int
        Kernel size
    within_ci_N_iterations : int
        Iterations for CI filtering
    N_iterations_optim : int
        Iterations for optimization
    overwrite : bool
        Force recalculation
    overwrite_preprocessing : bool
        Force recalculation of preprocessing caches (bootstrap signals and data_per_length_scale)

    Returns
    -------
    Tuple[List, List]
        (selection_points, loci_widths)
    """
    logger.info(f'Running loci assignment for {cur_chrom}')

    # Per-chromosome stream, as in run_loci_detection_per_chrom above.
    seed_task(derive_seed('loci_assignment', cur_chrom))

    output_dir = os.path.join(loci_results_dir, 'assignment', cur_chrom)
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f'Calculating bootstrap signals for {cur_chrom}')
    bootstrap_sampling_of_signal(
        cur_chrom=cur_chrom,
        final_events_df=final_events_df,
        N_bootstrap=N_bootstrap,
        calc_new_force_new=overwrite_preprocessing,
        calc_new_filename=os.path.join(
            loci_results_dir, 'signal_bootstrap', f'{cur_chrom}_N_{N_bootstrap}.pickle'))

    # Load data per length scale
    data_per_length_scale = collect_data_per_length_scale(
        final_events_df, cur_chrom, N_bootstrap=N_bootstrap, N_kernel=N_kernel,
        loci_results_dir=loci_results_dir,
        calc_new_force_new=overwrite_preprocessing,
        calc_new_filename=os.path.join(loci_results_dir, 'data_per_length_scale', f'{cur_chrom}.pickle'))
    
    # Filter reference_loci for this chromosome
    chrom_loci = reference_loci_df.query('chrom == @cur_chrom').copy()
    if len(chrom_loci) == 0:
        logger.warning(f'No loci defined for {cur_chrom}')
        return None, None
    
    logger.info(f'Found {len(chrom_loci)} loci to assign for {cur_chrom}')
    
    # Step 1: Create dummy selection_points with zero fitness at specified positions
    dummy_selection_points = []
    for loci_pos in chrom_loci['pos'].values:
        dummy_selection_points.append(
            [SelectionPoints(loci=[[loci_pos, 0]]) for _ in range(8)]
        )
    
    # Step 2: Optimize selection points with fixed positions
    logger.info(f'Optimizing fitness for {cur_chrom} with fixed positions')
    up_down_order = (chrom_loci['type'] == 'OG').values
    
    seed_task(derive_seed('loci_assignment', cur_chrom, 'optimizing'))
    optimized_selection_points, _, _ = _optimize_selection_points(
        N_iterations_optim,
        dummy_selection_points,
        data_per_length_scale,
        cur_chrom,
        best_loss=np.inf,
        show_progress=False,
        N_iterations_base=0,
        up_down_order=up_down_order,
        allow_pos_change=False,  # Keep positions fixed
    )
    optimized_selection_points = list(zip(*optimized_selection_points))
    
    # Step 3: Apply within CI filtering
    logger.info(f'Applying within-CI filtering for {cur_chrom}')
    seed_task(derive_seed('loci_assignment', cur_chrom, 'within_ci_filtering'))
    filtered_selection_points = within_ci_fitness_filter(
        cur_chrom,
        ranked_selection_points=optimized_selection_points,
        data_per_length_scale=data_per_length_scale,
        remove_empty_loci=False,
        show_progress=False,
        log_progress=False,
        N_iterations_optimization=within_ci_N_iterations,
        calc_new_force_new=overwrite,
        calc_new_filename=os.path.join(output_dir, 'assignment_within_ci_filtered.pickle')
    )

    # Save results
    save_pickle(filtered_selection_points, os.path.join(output_dir, 'final_selection_points.pickle'))

    # Infer widths (placeholder - set to small widths for now)
    N_loci = sum(len(chrom_loci.query('type == @t')) for t in ['OG', 'TSG'])
    loci_widths = [1e6] * N_loci  # Default width of 1 Mbp
    save_pickle(loci_widths, os.path.join(output_dir, 'final_loci_widths.pickle'))
    
    logger.info(f'✓ Assignment for {cur_chrom}: {N_loci} loci')
    return filtered_selection_points, loci_widths


def loci_assignment(
    name: str = None,
    processed_events: Optional[pd.DataFrame] = None,
    N_bootstrap: int = 1_000,
    N_kernel: int = 100_000,
    within_ci_N_iterations: int = 10_000,
    N_iterations_optim: int = 11_000,
    p_value_threshold: float = 0.05,
    permutation_null: Optional[pd.DataFrame] = None,
    p_values_strategy: str = 'zpool',
    overwrite: bool = False,
    overwrite_preprocessing: bool = False,
    calculate_p_value: bool = True,
):
    """
    Assign fitness values to pre-defined loci positions.
    
    This is an alternative to de-novo loci detection. Instead of detecting new loci,
    this function takes pre-defined loci positions from a file and assigns fitness values
    by optimizing them against the event data.
    
    Parameters
    ----------
    name : str, optional
        Project name (from config if not provided)
    processed_events : pd.DataFrame, optional
        Final events dataframe. If not provided, will be loaded from config.
    N_bootstrap : int
        Number of bootstrap samples
    N_kernel : int
        Kernel size
    within_ci_N_iterations : int
        Iterations for within-CI filtering
    N_iterations_optim : int
        Iterations for optimization
    length_scales_for_residuals : str
        Length scales to use (placeholder, not used in assignment)
    overwrite : bool
        Force recalculation
    overwrite_preprocessing : bool
        Force recalculation of preprocessing caches
    cores : int
        Number of cores for parallelization (not used in current version)

    Notes
    -----
    Requires config['input_files']['reference_loci'] to point to a TSV file with columns:
    - chrom: Chromosome (e.g., 'chr1', 'chr2', etc.)
    - start: Start position (bp)
    - end: End position (bp)
    - type: 'OG' for oncogenes (copy-number gains) or 'TSG' for tumor suppressors (copy-number losses)
    """
    logger.info('='*80)
    logger.info('LOCI ASSIGNMENT PIPELINE')
    logger.info('='*80)
    
    # Resolve name and directories
    name = name if name is not None else config['name']
    results_dir = config['directories']['results_dir']
    loci_results_dir = os.path.join(results_dir, name, 'loci_of_selection')
    os.makedirs(loci_results_dir, exist_ok=True)
    
    logger.info(f'Project name: {name}')
    logger.info(f'Results directory: {loci_results_dir}')
        
    # Load loci positions from config
    reference_loci_file = config['input_files'].get('reference_loci')
    if reference_loci_file and os.path.exists(reference_loci_file):
        logger.info(f'Loading loci positions from {reference_loci_file}')
        reference_loci_df = pd.read_csv(reference_loci_file, sep='\t')
    else:
        if files is None:
            raise FileNotFoundError(
                "importlib.resources unavailable for reference_loci file"
            )
        try:
            resource_name = os.path.basename(reference_loci_file or 'all_460_loci.tsv')
            content = files('spice').joinpath('reference_loci', resource_name).read_text()
            logger.info('Loading loci positions from package resources')
            reference_loci_df = pd.read_csv(StringIO(content), sep='\t')
        except (TypeError, ImportError, AttributeError, FileNotFoundError) as exc:
            raise FileNotFoundError(
                "reference_loci file not found. Please set config['input_files']['reference_loci'] "
                "to point to a TSV file with columns: chrom, start, end, type (OG or TSG)"
            ) from exc
    logger.info(f'Loaded {len(reference_loci_df)} loci positions')
    
    # Validate reference_loci format
    required_cols = ['chrom', 'pos', 'type']
    missing_cols = [c for c in required_cols if c not in reference_loci_df.columns]
    if missing_cols:
        raise ValueError(f"reference_loci must have columns {required_cols}, missing: {missing_cols}")
    
    # Validate type values are OG or TSG
    invalid_types = set(reference_loci_df['type'].unique()) - {'OG', 'TSG'}
    if invalid_types:
        raise ValueError(f"type column must contain only 'OG' or 'TSG', found: {invalid_types}")
    
    chromosomes = processed_events['chrom'].unique()
    assert set(chromosomes).issubset(set(['chr' + str(x) for x in range(1, 23)] + ['chrX', 'chrY'])), (
        f"Unexpected chromosomes in final events: {set(chromosomes) - set(['chr' + str(x) for x in range(1, 23)] + ['chrX', 'chrY'])}"
    )
    logger.info(f'Found {len(chromosomes)} unique chromosomes in final events: {chromosomes}')

    # Run per-chromosome assignment
    logger.info(f'Running loci assignment per chromosome')
    for cur_chrom in chromosomes:
        run_loci_assignment_per_chrom(
            reference_loci_df=reference_loci_df,
            cur_chrom=cur_chrom,
            final_events_df=processed_events,
            loci_results_dir=loci_results_dir,
            N_bootstrap=N_bootstrap,
            N_kernel=N_kernel,
            within_ci_N_iterations=within_ci_N_iterations,
            N_iterations_optim=N_iterations_optim,
            overwrite=overwrite,
            overwrite_preprocessing=overwrite_preprocessing,
        )

    # Combine results
    logger.info('Combining per-chromosome results')
    final_loci_df, filtered_selection_points, filtered_loci_widths, _unfiltered = combine_loci(
        loci_results_dir=loci_results_dir,
        processed_events=processed_events,
        p_value_threshold=p_value_threshold,
        permutation_null=permutation_null,
        p_values_strategy=p_values_strategy,
        calculate_p_value=calculate_p_value,
        overwrite=overwrite,
        mode='assignment'
    )
    
    logger.info('='*80)
    logger.info('LOCI ASSIGNMENT PIPELINE COMPLETED')
    logger.info('='*80)
    
    return final_loci_df


def build_final_loci_df(
    all_selection_points: Dict,
    all_loci_widths: Dict,
    final_events_df: pd.DataFrame,

) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Build loci_df, add scoring/summary columns, and compute added_events per locus.
    Returns loci_df and per-chrom locus width stds.
    """
    if sum(len(x) for x in all_loci_widths.values()) == 0:
        logger.warning('No loci to build a final loci dataframe from; returning an empty dataframe')
        return pd.DataFrame(columns=['chrom'])

    loci_df = create_loci_df(all_selection_points, all_loci_widths, nr_stds_widths=2,
                                min_widths_is_small_kernel=True)
    
    loci_df = calculate_events_per_loci_df(loci_df,
                                            all_selection_points=all_selection_points,
                                            final_events_df=final_events_df)
    
    return loci_df
