"""Main loci detection pipeline for de-novo TSG/OG detection."""

import os
import sys
from io import StringIO
from typing import Dict, Tuple, List, Optional
import pandas as pd
import numpy as np

from spice import config, data_loaders
from spice.length_scales import DEFAULT_LENGTH_SCALE_BOUNDARIES
from spice.utils import (open_pickle, save_pickle, CALC_NEW,
                         calc_telomere_bound_whole_arm_whole_chrom)
from spice.logging import log_debug, get_logger
from spice.tsg_og.detection import (
    collect_data_per_length_scale, detect_tsgs_ogs_for_all_length_scales, rank_loci, within_ci_fitness_filter,
    flip_up_down_assignment, final_optimization_step, limiting_fitness, infer_loci_widths, merge_overlapping_loci,
    calc_mse_loss, filter_loci, _optimize_selection_points, SelectionPoints)
from spice.tsg_og.signal_bootstrap import bootstrap_sampling_of_signal
from spice.tsg_og.simulation import copy_list_of_selection_points, convolution_simulation_per_ls
from spice.tsg_og.plateaus import categorize_events_by_plateau_overlap
from spice.tsg_og.loci import (
    create_loci_df, assign_p_values, calculate_events_per_loci_df)

if sys.version_info >= (3, 9):
    from importlib.resources import files
else:
    try:
        from importlib_resources import files
    except ImportError:
        files = None

logger = get_logger('loci_detection_main')
CHROMS = ['chr' + str(x) for x in range(1, 23)] + ['chrX', 'chrY']

def run_loci_detection_per_chrom(
    final_events_df,
    cur_chrom,
    which='default',
    name=None,
    N_loci=100,
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
    th_locus_prominence=5
):
    """
    Run the loci detection pipeline for a given chromosome.
    
    Parameters
    ----------
    cur_chrom : str
        Chromosome to analyze
    which : str, default='default'
        Which steps to run: 'default', single step, or comma-separated steps
    name : str, optional
        Project name (from config if not provided)
    N_loci : int, default=100
        Number of loci to detect
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
    """
    
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
        if which == 'default':
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
            raise ValueError(f"Unknown which mode: {which}. Use 'default', 'fast', or a step name with '+'")
    else:
        assert hasattr(which, '__iter__'), which
        which_steps = which
    
    # Resolve name and directories
    name = name if name is not None else config['name']
    
    output_dir = os.path.join(loci_results_dir, 'detection', cur_chrom)
       
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
        logger.info(f'Running flipping')
        log_debug(logger, f'Output: {output_dir}/{filenames["flipping"]}')
        
        if RESULTS['detection'] is None:
            RESULTS['detection'], _, _ = open_pickle(os.path.join(output_dir, filenames['detection']))
        
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
        logger.info(f'Running ranking')
        log_debug(logger, f'Output: {output_dir}/{filenames["ranking"]}')
        
        if RESULTS['flipping'] is None:
            RESULTS['flipping'] = open_pickle(os.path.join(output_dir, filenames['flipping']))
        
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
        logger.info(f'Running within_ci_filtering')
        log_debug(logger, f'Output: {output_dir}/{filenames["within_ci_filtering"]}')
        
        if RESULTS['ranking'] is None:
            if use_original_rank:
                if RESULTS['flipping'] is None:
                    RESULTS['flipping'] = open_pickle(os.path.join(output_dir, filenames['flipping']))
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
        logger.info(f'Running limiting')
        log_debug(logger, f'Output: {output_dir}/{filenames["limiting"]}')
        
        if RESULTS['within_ci_filtering'] is None:
            RESULTS['within_ci_filtering'] = open_pickle(os.path.join(output_dir, filenames['within_ci_filtering']))
        
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
        logger.info(f'Running optimizing_intermediate')
        log_debug(logger, f'Output: {output_dir}/{filenames["optimizing_intermediate"]}')
        
        if RESULTS['limiting'] is None:
            RESULTS['limiting'] = open_pickle(os.path.join(output_dir, filenames['limiting']))
        
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
        logger.info(f'Running loci_widths_intermediate')
        log_debug(logger, f'Output: {output_dir}/{filenames["loci_widths_intermediate"]}')
        
        if RESULTS['optimizing_intermediate'] is None:
            RESULTS['optimizing_intermediate'] = open_pickle(os.path.join(output_dir, filenames['optimizing_intermediate']))
        
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
        logger.info(f'Running merging')
        log_debug(logger, f'Output: {output_dir}/{filenames["merging"]}')
        
        if RESULTS['optimizing_intermediate'] is None:
            RESULTS['optimizing_intermediate'] = open_pickle(os.path.join(output_dir, filenames['optimizing_intermediate']))
        
        if RESULTS['loci_widths_intermediate'] is None:
            RESULTS['loci_widths_intermediate'] = open_pickle(os.path.join(output_dir, filenames['loci_widths_intermediate']))
        
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
        logger.info(f'Running optimizing')
        log_debug(logger, f'Output: {output_dir}/{filenames["optimizing"]}')
        
        input_source = 'flipping' if which == 'fast' else 'merging'

        if RESULTS[input_source] is None:
            RESULTS[input_source] = open_pickle(os.path.join(output_dir, filenames[input_source]))
   
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
        logger.info(f'Running loci_widths_intermediate_2')
        log_debug(logger, f'Output: {output_dir}/{filenames["loci_widths_intermediate_2"]}')
        
        if RESULTS['optimizing'] is None:
            RESULTS['optimizing'] = open_pickle(os.path.join(output_dir, filenames['optimizing']))
        
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
        logger.info(f'Running filter_loci_intermediate_1')
        log_debug(logger, f'Output: {output_dir}/{filenames["filter_loci_intermediate_1"]}')
        
        if RESULTS['loci_widths_intermediate_2'] is None:
            RESULTS['loci_widths_intermediate_2'] = open_pickle(os.path.join(output_dir, filenames['loci_widths_intermediate_2']))
        if RESULTS['optimizing'] is None:
            RESULTS['optimizing'] = open_pickle(os.path.join(output_dir, filenames['optimizing']))
        
        RESULTS['filter_loci_intermediate_1'] = filter_loci(
            cur_chrom=cur_chrom,
            final_selection_points=RESULTS['optimizing'],
            loci_widths=RESULTS['loci_widths_intermediate_2'],
            data_per_length_scale=data_per_length_scale,
            final_events_df=final_events_df,
            n_iterations_optim=filter_N_iterations_optim,
            show_progress_optim=False,
            max_deviation_optim=0.00001,
            calc_new_force_new=overwrite,
            calc_new_filename=os.path.join(output_dir, filenames['filter_loci_intermediate_1']))
    
    # Final within CI filtering step
    if 'final_within_ci_filtering' in which_steps:
        logger.info(f'Running final_within_ci_filtering')
        log_debug(logger, f'Output: {output_dir}/{filenames["final_within_ci_filtering"]}')
        
        input_source = 'optimizing' if which == 'fast' else 'filter_loci_intermediate_1'
        
        if RESULTS[input_source] is None:
            RESULTS[input_source] = open_pickle(os.path.join(output_dir, filenames[input_source]))
        
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
        logger.info(f'Running final_filter_loci')
        log_debug(logger, f'Output: {output_dir}/{filenames["final_filter_loci"]}')
        
        if RESULTS['final_within_ci_filtering'] is None:
            RESULTS['final_within_ci_filtering'] = open_pickle(os.path.join(output_dir, filenames['final_within_ci_filtering']))
        
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
            calc_new_force_new=overwrite,
            calc_new_filename=os.path.join(output_dir, filenames['final_filter_loci']))
    
    # Final limiting step
    if 'final_limiting' in which_steps:
        logger.info(f'Running final_limiting')
        log_debug(logger, f'Output: {output_dir}/{filenames["final_limiting"]}')
        
        if RESULTS['final_filter_loci'] is None:
            RESULTS['final_filter_loci'] = open_pickle(os.path.join(output_dir, filenames['final_filter_loci']))
        
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
                RESULTS['final_filter_loci'] = open_pickle(os.path.join(output_dir, filenames['final_filter_loci']))
            RESULTS['final_selection_points'] = copy_list_of_selection_points(RESULTS['final_filter_loci'])
            save_pickle(RESULTS['final_selection_points'], os.path.join(output_dir, filenames['final_selection_points']))


    # Final locus widths step
    if 'final_loci_widths' in which_steps:
        logger.info(f'Running final_loci_widths')
        log_debug(logger, f'Output: {output_dir}/{filenames["final_loci_widths"]}')
        
        if RESULTS['final_selection_points'] is None:
            RESULTS['final_selection_points'] = open_pickle(os.path.join(output_dir, filenames['final_selection_points']))
        
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
    #         RESULTS['final_selection_points'] = open_pickle(os.path.join(output_dir, filenames['final_selection_points']))
        
    #     RESULTS['one_by_one'] = add_loci_one_by_one(
    #         cur_chrom=chrom,
    #         raw_selection_points=RESULTS['final_selection_points'],
    #         data_per_length_scale=data_per_length_scale,
    #         show_progress=False)
    
    logger.info(f'Done! Loci detection for chrom={cur_chrom}, name={name}')
    
    return RESULTS


@CALC_NEW()
def combine_loci(
    loci_results_dir: str,
    processed_events: Optional[pd.DataFrame] = None,
    p_values_N_random: int = 10_000,
    p_values_N_iterations: int = 1_000,
    post_p_value_N_iterations: int = 25_000,
    calculate_p_value: bool = False,
    p_value_threshold: float = 0.05,
    p_values_mode: str = 'random',
    overwrite: bool = False,
    mode: str = 'detection',
) -> pd.DataFrame:
    """
    Combine results from all chromosomes after loci detection or assignment
    
    Loads final selection points and peak widths from all chromosomes. If overwrite=True,
    runs full_filter_by_p_values to filter by significance. Otherwise uses all loci
    without p-value filtering.
    
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

    for i, cur_chrom in enumerate(CHROMS[:-1], 1):
        
        # Check if data_per_length_scale file exists (indicator that chromosome was processed)
        data_per_length_scale_file = os.path.join(loci_results_dir, 'data_per_length_scale', f'{cur_chrom}.pickle')
        if not os.path.exists(data_per_length_scale_file):
            continue
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
    
    # Conditionally run full_filter_by_p_values based on calculate_p_value config
    if calculate_p_value:
        # Run full_filter_by_p_values to filter loci by significance
        logger.info('Running p-value filtering')
        filtered_selection_points, filtered_loci_widths, final_p_values = full_filter_by_p_values(
            all_selection_points=all_selection_points,
            all_loci_widths=all_loci_widths,
            all_data_per_length_scale=all_data_per_length_scale,
            loci_df=None,
            output_dir=loci_results_dir,
            N_random=p_values_N_random,
            p_values_N_iterations=p_values_N_iterations,
            post_p_value_N_iterations=post_p_value_N_iterations,
            final_events_df=processed_events,
            p_value_threshold=p_value_threshold,
            p_values_mode=p_values_mode,
            overwrite=overwrite,
        )
        logger.info(f'Filtered to {sum(len(x) for x in filtered_loci_widths.values())} significant loci (p < {p_value_threshold})')
    else:
        # Skip p-value filtering and use all loci
        logger.info('Skipping p-value filtering (calculate_p_value=False)')
        filtered_selection_points = all_selection_points
        filtered_loci_widths = all_loci_widths
        final_p_values = None
    
    # Run build_and_score_loci to create final loci dataframe
    log_debug(logger, "Building and scoring loci dataframe")
    loci_df = build_final_loci_df(
        all_selection_points=filtered_selection_points,
        all_loci_widths=filtered_loci_widths,
        final_events_df=processed_events,
        final_p_values=final_p_values,
    )
    
    log_debug(logger, f'Final loci dataframe: {len(loci_df)} loci across {loci_df["chrom"].nunique()} chromosomes')   
    return loci_df, filtered_selection_points, filtered_loci_widths


def process_final_events_for_loci_routines(
    final_events_df: Optional[pd.DataFrame] = None,
    length_scale_boundaries: Dict[str, Tuple[float, float]] = DEFAULT_LENGTH_SCALE_BOUNDARIES,
    remove_plateaus: bool = True,
    remove_chrY: bool = True,
    drop_duplicates: bool = True,
    use_observed_centromeres: bool = True,
    skip_assertions: bool = False,
) -> pd.DataFrame:
    """
    Process and filter copy-number events for loci detection analysis.
    
    This function performs comprehensive preprocessing of final events including:
    - Filtering by chromosome and telomere/centromere boundaries
    - Re-calculating event positions relative to centromeres and telomeres
    - Removing whole chromosome/arm events and keeping internal events
    - Filtering by event width within length scale boundaries
    - Removing duplicate events and plateau-overlapping events
    - Using observed centromere positions for improved classification
    
    Parameters
    ----------
    final_events_df : pd.DataFrame, optional
        DataFrame of final events. If None, loads from default location.
    length_scale_boundaries : dict
        Dictionary mapping length scale names to (min, max) width tuples.
    remove_plateaus : bool, default=True
        Whether to remove events overlapping copy-number plateaus.
    remove_chrY : bool, default=True
        Whether to exclude chrY events from analysis.
    drop_duplicates : bool, default=True
        Whether to remove duplicate event entries.
    use_observed_centromeres : bool, default=True
        Whether to use empirically observed centromere positions for classification.
    skip_assertions : bool, default=False
        Whether to skip data quality assertions (for debugging).
    
    Returns
    -------
    pd.DataFrame
        Filtered and processed events dataframe containing only internal events
        within the specified length scale boundaries, with updated position
        classifications and centromere/telomere annotations.
    """

    CENTROMERES_OBSERVED = data_loaders.load_centromeres(observed=True, extended=False)

    if final_events_df is None:
        log_debug(logger, "Loading final events dataframe from file")
        final_events_df = data_loaders.load_final_events()

    log_debug(logger, f"Loaded {len(final_events_df)} events events across {final_events_df['sample'].nunique()} samples and {final_events_df['id'].nunique()} IDs")
    
    if remove_chrY:
        final_events_df = final_events_df.query('chrom != "chrY"').reset_index(drop=True).copy()

    # Remove IDs where the number of events does not match
    valid_ids = (final_events_df.groupby('id').size().loc[
        (final_events_df.groupby('id').size() ==
        final_events_df.groupby('id')['events_per_chrom'].first())].index.values)
    if len(valid_ids) < final_events_df['id'].nunique():
        logger.warning(f'Found {final_events_df["id"].nunique() - len(valid_ids)} IDs ({100*(final_events_df["id"].nunique() - len(valid_ids)) / final_events_df["id"].nunique():.4f}%) with inconsistent number of events')
        final_events_df = final_events_df.query('id in @valid_ids').copy()
        log_debug(logger, 'Removed invalid IDs, where the number of events did not match "events_per_chrom"')
        log_debug(logger, f'Events now have: {final_events_df["sample"].nunique()} samples, {final_events_df["id"].nunique()} IDs and {len(final_events_df)} events')

    final_events_df = final_events_df.loc[~final_events_df['telomere_bound']].reset_index(drop=True).copy()

    # Re-calculate centromere/telomere/whole arm/whole chrom assignment and only keep internal events
    final_events_df = final_events_df.join(data_loaders.load_centromeres(extended=True), on='chrom')
    final_events_df[
        ['centromere_bound_l', 'centromere_bound_r', 'telomere_bound_l',
        'telomere_bound_r', 'telomere_bound', 'whole_arm', 'whole_chrom']] = np.stack(
            calc_telomere_bound_whole_arm_whole_chrom(final_events_df, return_left_and_right=True), axis=1)
    
    # Adjust start/end, especially important for chrX where the telomere assignment is off
    final_events_df.loc[final_events_df['telomere_bound_l'], 'start'] = 0
    final_events_df.loc[final_events_df['telomere_bound_r'], 'end'] = final_events_df.loc[
        final_events_df['telomere_bound_r'], 'chrom_length']

    final_events_df['whole_arm'] = final_events_df.eval('(telomere_bound_l and centromere_bound_r) or (telomere_bound_r and centromere_bound_l)')
    final_events_df.loc[final_events_df.query('whole_chrom').index, 'whole_arm'] = False

    final_events_df['centromere_bound'] = np.logical_or(final_events_df['centromere_bound_l'].values, final_events_df['centromere_bound_r'].values)
    final_events_df['whole_centromere'] = np.logical_and(final_events_df['centromere_bound_l'].values, final_events_df['centromere_bound_r'].values)

    # Check whether any event is within 1Mbp of the centromere and remove them
    # Note that observed centromeres are removed in create_features_pipeline
    log_debug(logger, 'Remove whole centromere events and events within 1Mbp of the centromere')
    centromeres = data_loaders.load_centromeres(extended=True)
    final_events_df = (final_events_df
        .drop(columns=['centro_start', 'centro_end'], errors='ignore')
        .join(centromeres, on='chrom'))
    centromeres_pad = data_loaders.load_centromeres(extended=True, pad=5e6).rename(
        columns={'centro_start': 'centro_start_pad', 'centro_end': 'centro_end_pad'})
    final_events_df = (final_events_df
        .drop(columns=['centro_start_pad', 'centro_end_pad'], errors='ignore')
        .join(centromeres_pad, on='chrom'))
    final_events_df['inside_centromere'] = final_events_df.eval('start>=centro_start_pad-2 and end<=centro_end_pad+2')    
    
    final_events_df = final_events_df.query('not whole_centromere and not inside_centromere').drop(columns=['whole_centromere', 'inside_centromere']).copy()
    assert len(final_events_df) > 0, 'No events left after removing whole centromeres and events within 1Mbp of the centromere. Please check the centromere definitions and event coordinates.'
    log_debug(logger, f'Events now have: {final_events_df["sample"].nunique()} samples, {final_events_df["id"].nunique()} IDs and {len(final_events_df)} events')

    short_chroms = ['chr13', 'chr14', 'chr15', 'chr21', 'chr22']
    final_events_df['short_arm'] = False
    final_events_df.loc[final_events_df.query('chrom in @short_chroms').index, 'short_arm'] = True
    final_events_df.loc[final_events_df.query('chrom in @short_chroms and whole_arm').index, 'whole_chrom'] = True
    final_events_df.loc[final_events_df.query('chrom in @short_chroms and whole_arm').index, 'whole_arm'] = False
    final_events_df.loc[final_events_df.query('chrom in @short_chroms and telomere_bound_r and start <= centro_end_pad+2').index, 'whole_chrom'] = True

    # Events that are within 0.95-1.05 of the arm size are considered whole arm
    telomere_bound_events = final_events_df.query('telomere_bound and not whole_chrom and not whole_arm').copy()
    telomere_bound_events['left_bound'] = telomere_bound_events.eval('start <= 100000')
    telomere_bound_events['right_bound'] = telomere_bound_events.eval('end >= chrom_length - 100000')
    telomere_bound_events.loc[telomere_bound_events['chrom'].isin(short_chroms), 'left_bound'] = telomere_bound_events.loc[telomere_bound_events['chrom'].isin(short_chroms)].eval('start <= centro_end')
    telomere_bound_events['arm_size'] = telomere_bound_events['centro_start']
    telomere_bound_events.loc[telomere_bound_events['right_bound'], 'arm_size'] = telomere_bound_events.loc[telomere_bound_events['right_bound'], 'chrom_length'] - telomere_bound_events.loc[telomere_bound_events['right_bound'], 'centro_end']
    telomere_bound_events.loc[telomere_bound_events['chrom'].isin(short_chroms), 'arm_size'] = telomere_bound_events.loc[telomere_bound_events['chrom'].isin(short_chroms), 'chrom_length'] - telomere_bound_events.loc[telomere_bound_events['chrom'].isin(short_chroms), 'centro_end']
    telomere_bound_events['within_arm'] = telomere_bound_events['width'] < telomere_bound_events['arm_size']
    telomere_bound_events['width_norm'] = telomere_bound_events['width'] / telomere_bound_events['arm_size']
    whole_chrom_indices = telomere_bound_events.query('(left_bound and right_bound)').index.values
    whole_arm_indices = telomere_bound_events.query('not (left_bound and right_bound) and width_norm > 0.95 and width_norm < 1.05').index.values
    final_events_df.loc[whole_chrom_indices, ['whole_chrom']] = True
    final_events_df.loc[whole_arm_indices, ['whole_arm']] = True

    final_events_df['pos'] = final_events_df.apply(
        lambda x: 'whole_chrom' if x['whole_chrom'] else 'whole_arm' if x['whole_arm'] else
        'centromere_bound' if x['centromere_bound'] else 'telomere_bound' if x['telomere_bound'] else 'internal', axis=1)

    # Refine centromere-bound classification using empirically observed centromere positions per length scale
    # This improves upon the theoretical centromere definitions by using data-driven boundaries
    if use_observed_centromeres:
        old_n_centromere = (final_events_df['pos'] == 'centromere_bound').sum()
        for cur_chrom in final_events_df['chrom'].unique():
            for cur_length_scale in ['small', 'mid1', 'mid2', 'large']:
                cur_length_scale_border = length_scale_boundaries[cur_length_scale]
                cur_events = final_events_df.query('chrom == @cur_chrom and pos == "internal" and width > @cur_length_scale_border[0] and width <= @cur_length_scale_border[1]')
                is_centromere_bound_new = (
                    ((cur_events['start']>=CENTROMERES_OBSERVED[cur_length_scale].loc[cur_chrom, 'centro_start']) & (cur_events['start']<=CENTROMERES_OBSERVED[cur_length_scale].loc[cur_chrom, 'centro_end'])) |
                    ((cur_events['end']>=CENTROMERES_OBSERVED[cur_length_scale].loc[cur_chrom, 'centro_start']) & (cur_events['end']<=CENTROMERES_OBSERVED[cur_length_scale].loc[cur_chrom, 'centro_end'])))
                
                cur_ind = cur_events.loc[(is_centromere_bound_new & ~cur_events['telomere_bound'])].index

                final_events_df.loc[cur_ind, 'pos'] = 'centromere_bound'

        new_n_centromere = (final_events_df['pos'] == 'centromere_bound').sum()
        log_debug(logger, f'Assigned {new_n_centromere - old_n_centromere} new events as centromere bound using observed centromeres')

    # Filter to only internal events within the specified length scale boundaries
    # This removes whole chromosome, whole arm, centromere-bound, and telomere-bound events
    old_n = len(final_events_df)
    min_width = DEFAULT_LENGTH_SCALE_BOUNDARIES['small'][0]
    max_width = DEFAULT_LENGTH_SCALE_BOUNDARIES['large'][1]
    final_events_df = final_events_df.query('pos == "internal" and width >= @min_width and width <= @max_width').copy()
    log_debug(logger, f'Only kept internal events: {len(final_events_df)} remaining (dropped {old_n - len(final_events_df)})')

    assert skip_assertions or not final_events_df.isna().sum().any()

    # Remove duplicate entries to only get unique events
    if drop_duplicates:
        old_len = len(final_events_df)
        final_events_df = final_events_df.drop_duplicates(['id', 'chrom', 'type', 'start', 'end'], keep='first').copy()
        log_debug(logger, f'Dropped {old_len - len(final_events_df)} duplicates -> {len(final_events_df)} events')

    final_events_df = final_events_df.reset_index(drop=True).copy()

    # Remove plateau events
    final_events_df['plateau'] = 'neither_left_nor_right'
    if config['input_files'].get('plateaus', None) is not None:
        log_debug(logger, "Loading plateaus data")
        plateaus_df = pd.read_csv(config['input_files']['plateaus'], sep='\t', index_col=None)
        log_debug(logger, f"Loaded plateaus data: {len(plateaus_df)} entries")
        if plateaus_df is not None:
            final_events_df = categorize_events_by_plateau_overlap(plateaus_df, final_events_df)
            log_debug(logger, f'Categorized {len(final_events_df)} events by plateau overlap ({len(plateaus_df)} plateaus): {dict(final_events_df["plateau"].value_counts())}')
            if remove_plateaus:
                plateau_events = final_events_df.query('plateau != "neither_left_nor_right"')
                log_debug(logger, f"Filtering out {len(plateau_events)} events overlapping plateaus")
                final_events_df = final_events_df.query('plateau == "neither_left_nor_right"').copy().reset_index(drop=True)

    # Very important for some downstream analysis that requires unique indices
    final_events_df = final_events_df.reset_index(drop=True)

    return final_events_df


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
    p_values_N_random: int = 10_000,
    p_values_N_iterations: int = 1_000,
    post_p_value_N_iterations: int = 25_000,
    p_values_mode: str = 'random',
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
    final_loci_df, filtered_selection_points, filtered_loci_widths = combine_loci(
        loci_results_dir=loci_results_dir,
        processed_events=processed_events,
        p_values_N_random=p_values_N_random,
        p_values_N_iterations=p_values_N_iterations,
        post_p_value_N_iterations=post_p_value_N_iterations,
        p_values_mode=p_values_mode,
        calculate_p_value=calculate_p_value,
        overwrite=overwrite,
        mode='assignment'
        ###
    )
    
    logger.info('='*80)
    logger.info('LOCI ASSIGNMENT PIPELINE COMPLETED')
    logger.info('='*80)
    
    return final_loci_df


def build_final_loci_df(
    all_selection_points: Dict,
    all_loci_widths: Dict,
    final_events_df: pd.DataFrame,
    final_p_values=None

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
    
    if final_p_values is not None:
        assert len(final_p_values) == len(loci_df), (
            f'Length mismatch between provided final_p_values ({len(final_p_values)}) '
            f'and loci_df ({len(loci_df)})'
        )
        loci_df['p_value'] = final_p_values
    else:
        loci_df['p_value'] = 'not calculated'

    return loci_df


@CALC_NEW()
def full_filter_by_p_values(
    all_selection_points,
    all_loci_widths,
    all_data_per_length_scale,
    output_dir,
    loci_df=None,
    p_value_threshold=0.05,
    p_values_mode='random',
    N_random=10_000,
    p_values_N_iterations=1_000,
    post_p_value_N_iterations=25_000,
    final_events_df=None,
    overwrite=False,
):
    from spice import directories, config
    if loci_df is None:
        assert final_events_df is not None, "final_events_df must be provided if loci_df is None"
        loci_df = create_loci_df(all_selection_points, all_loci_widths, nr_stds_widths=2,
                                   min_widths_is_small_kernel=True)
        loci_df = calculate_events_per_loci_df(loci_df,
                                                all_selection_points=all_selection_points,
                                                final_events_df=final_events_df)

    log_debug(logger, 'Calculating p-values for loci')
    loci_df = assign_p_values(
        loci_df,
        N_random=N_random,
        n_iterations_optim=p_values_N_iterations,
        output_dir=output_dir,
        data_per_length_scale=all_data_per_length_scale,
        overwrite=overwrite,
        mode=p_values_mode,
    )

    if len(loci_df) == 0:
        # Empty bucket: no candidate loci at all (e.g. no events of this type/size on this chromosome)
        logger.warning('No candidate loci found; returning an empty locus set')
        return {}, {}, np.array([])

    n_significant = len(loci_df.query("p_value < @p_value_threshold"))
    logger.info(f'Out of {len(loci_df)} loci, {n_significant} ({n_significant/len(loci_df):.2%}%) are significant at p < {p_value_threshold} ')

    try:
        filtered_selection_points = {}
        filtered_loci_widths = {}
        final_p_values = []
        for cur_chrom in loci_df['chrom'].unique():
            cur_sp = [list(x) for x in copy_list_of_selection_points(all_selection_points[cur_chrom])]
            cur_loci = loci_df.query('chrom == @cur_chrom')
            is_significant = cur_loci.sort_values('rank_on_chrom').eval('p_value < @p_value_threshold').values
            filtered_selection_points[cur_chrom] = [
                [x for i, x in enumerate(ls_x) if is_significant[i]] for ls_x in cur_sp]
            filtered_loci_widths[cur_chrom] = [
                x for i, x in enumerate(all_loci_widths[cur_chrom]) if is_significant[i]]
            final_p_values.append(cur_loci.sort_values('rank_on_chrom').loc[is_significant]['p_value'].values)
        final_p_values = np.concatenate(final_p_values) if final_p_values else np.array([])
        assert np.all(final_p_values < p_value_threshold), f'{np.sum(final_p_values >= p_value_threshold)} loci with p >= {p_value_threshold} after filtering'
        assert len(final_p_values) == sum(len(x) for x in filtered_loci_widths.values()), (
            f'Length mismatch between final_p_values ({len(final_p_values)}) and filtered loci '
            f'({sum(len(x) for x in filtered_loci_widths.values())})'
        )
        if len(final_p_values) == 0:
            # Empty bucket: loci existed but none survived the p-value cut
            logger.warning('No loci passed the p-value filtering!')
            return filtered_selection_points, filtered_loci_widths, final_p_values

        logger.info('Optimizing selection points after p-value filtering')
        for cur_chrom in loci_df['chrom'].unique():
            cur_sp = filtered_selection_points[cur_chrom]
            if len(cur_sp[0]) == 0:
                # Empty bucket for this chromosome specifically: nothing left to optimize
                log_debug(logger, f'No significant loci left on {cur_chrom} after filtering; skipping optimization')
                continue
            log_debug(logger, f'Optimizing selection points on {cur_chrom}')
            allowed_fitness_change = np.stack([[x[0].fitness != 0 for x in y] for y in cur_sp])
            up_down_order = np.array([any([cur_sp[j][cluster_j][0].fitness > 0 for j in range(0, 8, 2)])
                                for cluster_j in range(len(cur_sp[0]))])
            cur_conv = convolution_simulation_per_ls(cur_chrom, all_data_per_length_scale[cur_chrom], cur_sp)
            cur_mse = calc_mse_loss(all_data_per_length_scale[cur_chrom], cur_conv)
            cur_sp_optim, _, _ = _optimize_selection_points(
                post_p_value_N_iterations,
                list(zip(*cur_sp)),
                all_data_per_length_scale[cur_chrom],
                cur_chrom,
                best_loss=cur_mse,
                show_progress=False,
                N_iterations_base=0,
                max_fitness=[1.1*max([y[0].fitness for y in x]) for x in cur_sp],
                loci_to_optimize=None,
                final_iteration=False,
                allowed_fitness_change=allowed_fitness_change,
                max_deviation=0.00001,
                allow_pos_change=False,
                up_down_order=up_down_order,
                blocked_distance_th=2e5
            )
            filtered_selection_points[cur_chrom] = list(zip(*cur_sp_optim))

        return filtered_selection_points, filtered_loci_widths, final_p_values

    except (ZeroDivisionError, IndexError, ValueError, KeyError, AssertionError) as e:
        # Catch-all for empty/mis-sized buckets we didn't anticipate above (e.g. mask/list
        # misalignment) - treat as "nothing significant" rather than aborting the whole run.
        logger.warning(f'p-value filtering failed ({type(e).__name__}: {e}); treating as an empty bucket and returning no loci')
        return {}, {}, np.array([])
