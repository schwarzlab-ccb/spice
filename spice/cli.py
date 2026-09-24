#!/usr/bin/env python
"""Command-line interface for SPICE."""

import glob
import os
import argparse
import re

# Import base package only; defer submodule imports until after config is loaded
import pandas as pd

import spice

#: default permutations for the fitness p-value null; mirrors
#: spice.tsg_og.permutation.DEFAULT_K, duplicated because the argument parser is built
#: before any spice module (and so any config) may be imported.
_DEFAULT_K = 16
from spice.utils import save_pickle, open_pickle
# No other SPICE imports here!


def _apply_seed(args, logger):
    """Fix this run's RNG seed: --seed, else params.seed from the config, else the default.

    Every SPICE command is stochastic, so this is what makes a run repeatable; see
    spice.random_state for what the seed does and does not cover (wall-clock limits are outside it).
    """
    from spice import config
    from spice.random_state import set_seed

    cli_seed = getattr(args, 'seed', None)
    seed = set_seed(cli_seed if cli_seed is not None else config['params'].get('seed'))
    logger.info(f'Random seed: {seed} (from {"--seed" if cli_seed is not None else "config params.seed"})')
    return seed


def main_event_inference(args):
    """Run event inference pipeline."""
    # Handle 'all' or empty arguments, and expand trailing + syntax (e.g., split+)
    step_order = ['preprocessing', 'split', 'all_solutions', 'disambiguate', 'large_chroms', 'combine']
    valid_steps = step_order + ['all']
    steps_input = getattr(args, 'event_steps', None)
    
    if steps_input is None or steps_input == ['all']:
        which = step_order.copy()
    elif any('+' in x for x in steps_input):
        assert len(steps_input) == 1, 'Only one step with + is allowed'
        assert steps_input[0].endswith('+'), 'Only trailing + syntax is supported'
        which = step_order[step_order.index(steps_input[0][:-1]):]
    else:
        which = steps_input

    invalid_steps = [step for step in which if step not in valid_steps]
    if invalid_steps:
        raise ValueError(f"Invalid step(s): {', '.join(invalid_steps)}. Valid steps are: preprocessing, split, all_solutions, disambiguate, large_chroms, combine")

    # Load configuration before importing submodules that may read it
    spice.load_config(args.config_path)
    from spice import config

    # Handle --clean early to avoid expensive imports
    if args.clean:
        import shutil
        name = config.get('name', None)
        if not name:
            raise ValueError("Config file must specify a 'name' field.")
        results_events_dir = os.path.join(config['directories']['results_dir'], name, 'events')
        print(f'Cleaning intermediate files at {results_events_dir}')
        for wgd in ['nowgd', 'wgd']:
            shutil.rmtree(os.path.join(results_events_dir, wgd), ignore_errors=True)
        print('Done cleaning.')
        return
    
    # Now do the expensive imports
    from spice.data_loaders import load_final_events, resolve_copynumber_file
    from spice.utils import timeout, FunctionTimeoutError
    from spice.logging import configure_logging, get_logger
    from spice.cli_functions import save_fail_reports, step_aware_cleanup, _run_batch
    from spice.preprocessing.split_input import split_tsv_file
    from spice.event_inference.pipeline import (
        full_paths_from_graph_with_sv_wrapper, solve_with_knn_wrapper, solve_with_mcmc_wrapper,
        combine_final_events)

    if 'name' not in config or not config['name']:
        raise ValueError("Config file must specify a 'name' field.")
    if 'input_files' not in config or 'copynumber' not in config['input_files']:
        raise ValueError("Config file must specify 'input_files.copynumber'.")
    
    # Create logger AFTER imports to avoid it being disabled by medicc's logging.config.dictConfig
    log_level = 'DEBUG' if args.debug else config['params'].get('logging_level', 'INFO')

    configure_logging(
        log_mode=args.log,
        log_dir=config['directories']['log_dir'],
        config_name=config['name'],
        level=log_level,
    )
    logger = get_logger('SPICE', spice_prefix=False)

    if args.debug and args.cores and args.cores > 1:
        logger.warning("Debug mode with multiple cores may lead to interleaved log messages.")

    name = config['name']
    if ' ' in name:
        logger.error("Project name must not contain spaces.")
        return
    directories = config['directories']
    results_dir = os.path.join(directories['results_dir'], name)
    results_events_dir = os.path.join(directories['results_dir'], name, 'events')
    log_dir = os.path.join(directories['log_dir'])
    plots_base_dir = os.path.join(directories['plot_dir'], name)
    for cur_dir in [results_events_dir, log_dir, plots_base_dir]:
        if not os.path.exists(cur_dir):
            logger.info(f"Creating directory {cur_dir}")
            os.makedirs(cur_dir)

    logger.info('Running SPICE: Selection Patterns In somatic Copy-number Events')
    logger.info(f'Running event inference for project name {name} with config file {args.config_path}')
    _apply_seed(args, logger)

    logger.info(f'Results will be stored in {results_events_dir}')
    logger.info(f'Running the following steps: {", ".join(which)}')

    selected_ids = args.ids.split(',') if args.ids is not None else None
    if selected_ids is not None:
        logger.info(f'Selecting only IDs: {selected_ids}')

    # Check number of samples and warn if many
    import pandas as pd
    copynumber_file = resolve_copynumber_file(return_raw=True)
    df = pd.read_csv(copynumber_file, sep='\t', usecols=['sample_id'])
    n_samples = df['sample_id'].nunique()
    logger.info(f'Input copy-number file contains {n_samples} unique samples.')
    if n_samples > 50:
        logger.warning("=" * 80)
        logger.warning("!!! WARNING !!!")
        logger.warning(f"Large number of input samples detected (N={n_samples}).")
        logger.warning("")
        logger.warning("SPICE can be very slow when processing many samples in serial mode.")
        logger.warning("For large datasets, consider using --cores to parallelize, or splitting")
        logger.warning("the run across --ids batches on a cluster.")
        logger.warning("=" * 80)

    total_cn = config['params'].get('total_cn', False)
    logger.info(f'Copy-number mode: {"total_cn" if total_cn else "haplotype-specific"}')

    # Clean old files
    if not args.keep_old and args.ids is None:
        if config['params'].get('skip_existing', False):
            raise ValueError("If skip_existing=True in config, have to use --keep-old to avoid deleting existing files.")
        logger.info('Cleaning old intermediate files')
        step_aware_cleanup(results_events_dir, which)

    # Run preprocessing only when requested
    if 'preprocessing' in which and args.run_preprocessing:
        from spice.preprocessing.extra_preprocessing import main as extra_preprocessing_main
        logger.info('Starting extra preprocessing step (pre-split)')
        extra_preprocessing_main(
            unique_chroms=bool(args.pre_unique_chroms),
            total_cn=total_cn,
            skip_phasing=bool(args.pre_skip_phasing),
            skip_centromeres=bool(args.pre_skip_centromeres),
        )
    elif 'preprocessing' in which and not args.run_preprocessing:
        logger.warning('Skipping preprocessing step: requested via --event-steps but --run-preprocessing was not passed. Use --run-preprocessing to actually run it.')

    chrom_segments_file = resolve_copynumber_file()

    if 'split' in which:
        logger.info('Starting splitting of the input TSV')
        split_tsv_file(name, keep_old=args.keep_old, cores=args.cores, selected_ids=selected_ids)

    # Collect per-ID failures to report at end
    failed_reports = []

    if 'all_solutions' in which:
        logger.info('Starting inference of all solutions')
        skip_existing = config['params'].get('skip_existing', False)
        for wgd_status in ['nowgd', 'wgd']:
            is_wgd = (wgd_status == 'wgd')
            # sorted for deterministic output
            cur_ids = [x.replace('.pickle', '')
                    for x in sorted(os.listdir(os.path.join(str(results_events_dir), wgd_status, 'chrom_data_full')))]
            if selected_ids is not None:
                cur_ids = [x for x in cur_ids if x in selected_ids]

            @timeout(config['params']['time_limit_all_solutions'], mode="auto")
            def run_full_paths(cur_id):
                output_file = os.path.join(results_events_dir, wgd_status, 'full_paths_multiple_solutions', f'{cur_id}.pickle')
                if skip_existing and os.path.exists(output_file):
                    logger.info(f"Skipping all_solutions for {cur_id} ({wgd_status}) as {output_file} exists.")
                    return {'status': 'skipped', 'cur_id': cur_id, 'step': 'all_solutions'}
                return full_paths_from_graph_with_sv_wrapper(
                    cur_id=cur_id,
                    is_wgd=(wgd_status == 'wgd'),
                    chrom_segments_file=chrom_segments_file,
                    sv_data_file=config['input_files'].get('sv', None),
                    chrom_file=os.path.join(results_events_dir, wgd_status, 'chrom_data_full', f'{cur_id}.pickle'),
                    sv_matching_threshold=config['params']['sv_matching_threshold'],
                    use_cache=config['params']['use_cache'],
                    total_cn=total_cn,
                    all_loh_solutions=config['params']['all_loh_solutions'],
                    output_file=output_file,
                    save_output=True,
                    skip_loh_checks=True,
                )
            results = _run_batch(cur_ids, args.cores, f'All solutions ({wgd_status})', run_full_paths, logger)
            cur_failed_reports = [r for r in results if isinstance(r, dict) and r.get('status') == 'failed']
            failed_reports.extend(cur_failed_reports)
            save_fail_reports(cur_failed_reports, cur_step=wgd_status + '_all_solutions')
        save_fail_reports(failed_reports)

    if 'disambiguate' in which:
        logger.info('Starting KNN disambiguation of solutions with multiple paths')
        skip_existing = config['params'].get('skip_existing', False)
        full_paths_multiple_solutions_dirs=[os.path.join(results_events_dir, 'nowgd', 'full_paths_multiple_solutions'),
                                        os.path.join(results_events_dir, 'wgd', 'full_paths_multiple_solutions')]
        for wgd_status in ['nowgd', 'wgd']:
            if not os.path.exists(os.path.join(str(results_events_dir), wgd_status, 'full_paths_multiple_solutions')):
                logger.info(f"Directory {os.path.join(str(results_events_dir), wgd_status, 'full_paths_multiple_solutions')} does not exist, skipping disambiguation for {wgd_status}")
                continue
            is_wgd = (wgd_status == 'wgd')
            cur_ids = [x.replace('.pickle', '')
                    for x in sorted(os.listdir(os.path.join(str(results_events_dir), wgd_status, 'full_paths_multiple_solutions')))]
            if selected_ids is not None:
                cur_ids = [x for x in cur_ids if x in selected_ids]
            def run_knn(cur_id):
                output_file = os.path.join(results_events_dir, wgd_status, 'knn_solved_chroms', f'{cur_id}.pickle')
                if skip_existing and os.path.exists(output_file):
                    logger.info(f"Skipping disambiguate for {cur_id} ({wgd_status}) as {output_file} exists.")
                    return {'status': 'skipped', 'cur_id': cur_id, 'step': 'disambiguate'}
                return solve_with_knn_wrapper(
                    output_file=output_file ,
                    cur_id=cur_id ,
                    is_wgd=is_wgd ,
                    chrom_segments_file=chrom_segments_file,
                    k=config['params']['knn_k'],
                    full_paths_multiple_solutions_dirs=full_paths_multiple_solutions_dirs,
                    save_all_scores=None,
                    perform_loh_checks=True,
                    single_width_bin=True
                )
            results = _run_batch(cur_ids, args.cores, f'Disambiguate solutions ({wgd_status})', run_knn, logger)
            cur_failed_reports = [r for r in results if isinstance(r, dict) and r.get('status') == 'failed']
            failed_reports.extend(cur_failed_reports)
            save_fail_reports(cur_failed_reports, cur_step=wgd_status + '_disambiguate')
        save_fail_reports(failed_reports)

    if 'large_chroms' in which:
        logger.info('Starting MCMC inference for large chromosomes with many events')
        skip_existing = config['params'].get('skip_existing', False)
        skip_loh_check = config['params'].get('skip_loh_check_for_large_chroms', True)
        for wgd_status in ['nowgd', 'wgd']:
            if not os.path.exists(os.path.join(str(results_events_dir), wgd_status, 'chrom_data_large')):
                logger.info(f"Directory {os.path.join(str(results_events_dir), wgd_status, 'chrom_data_large')} does not exist, skipping large chromosomes for {wgd_status}")
                continue
            is_wgd = (wgd_status == 'wgd')
            cur_ids = [x.replace('.pickle', '')
                    for x in sorted(os.listdir(os.path.join(str(results_events_dir), wgd_status, 'chrom_data_large')))]
            if selected_ids is not None:
                cur_ids = [x for x in cur_ids if x in selected_ids]

            def run_mcmc(cur_id):
                output_file = os.path.join(results_events_dir, wgd_status, 'mcmc_solved_chroms_large', f'{cur_id}.pickle')
                if skip_existing and os.path.exists(output_file):
                    logger.info(f"Skipping large_chroms for {cur_id} ({wgd_status}) as {output_file} exists.")
                    return {'status': 'skipped', 'cur_id': cur_id, 'step': 'large_chroms'}

                return solve_with_mcmc_wrapper(
                    output_file=output_file,
                    chrom_file=os.path.join(results_events_dir, wgd_status, 'chrom_data_large', f'{cur_id}.pickle'),
                    is_wgd=is_wgd,
                    chrom_segments_file=chrom_segments_file,
                    sv_data_file=config['input_files'].get('sv', None),
                    k=config['params']['knn_k'],
                    total_cn=total_cn,
                    save_all_scores=None,
                    n_iteration_scale=config['params']['mcmc_n_iterations_scale'],
                    log_progress=True,
                    fail_on_empty=False,
                    skip_loh_check=skip_loh_check,
                    min_T=config['params']['mcmc_min_T'],
                    max_T=config['params']['mcmc_max_T'],
                    # Runaway ceilings; None (default) = off
                    max_iterations=config['params'].get('mcmc_max_iterations', None),
                    loh_solve_time_limit=config['params'].get('mcmc_loh_solve_time_limit', None),
                )

            results = _run_batch(cur_ids, args.cores, f'Large chromosomes ({wgd_status})', run_mcmc, logger)
            cur_failed_reports = [r for r in results if isinstance(r, dict) and r.get('status') == 'failed']
            failed_reports.extend(cur_failed_reports)
            save_fail_reports(cur_failed_reports, cur_step=wgd_status + '_large_chroms')
        save_fail_reports(failed_reports)


    if 'combine' in which:
        logger.info('Starting combination of final events from all solving methods')
        solved_dirs = (
            [os.path.join(results_events_dir, wgd, 'knn_solved_chroms') for wgd in ['nowgd', 'wgd']] +
            [os.path.join(results_events_dir, wgd, 'full_paths_single_solution') for wgd in ['nowgd', 'wgd']] +
            [os.path.join(results_events_dir, wgd, 'mcmc_solved_chroms_large') for wgd in ['nowgd', 'wgd']]
        )
        combine_final_events(
            solved_dirs=solved_dirs,
            chrom_segments_file=chrom_segments_file,
            sv_data=config['input_files'].get('sv', None),
            sv_matching_threshold=config['params']['sv_matching_threshold'],
            output_dir=results_dir
        )

    save_fail_reports(failed_reports, logger=logger)
    logger.info(f'Done. Results are in {results_events_dir}')


def _load_plot_selection_points(output_dir, mode, chrom):
    """Prefer the combined detection fit used by the final table and CI scores."""
    combined = os.path.join(output_dir, mode, 'final_loci_detection_filtered.pickle')
    if mode == 'detection' and os.path.exists(combined):
        points = open_pickle(combined)
        if chrom not in points:
            raise ValueError(f'Combined detection fit has no {chrom}; recombine before plotting')
        return points[chrom], True
    original = os.path.join(output_dir, mode, chrom, 'final_selection_points.pickle')
    return open_pickle(original), False


def main_plotting(args):
    """Run plotting mode."""
    import pandas as pd
    
    # Load configuration
    spice.load_config(args.config_path)
    from spice import config
    from spice.data_loaders import load_final_events, resolve_copynumber_file
    from spice.logging import configure_logging, get_logger
    from spice import plot as spice_plot
    from matplotlib import pyplot as plt

    if 'name' not in config or not config['name']:
        raise ValueError("Config file must specify a 'name' field.")
    
    # Create logger
    log_level = 'DEBUG' if args.debug else config['params'].get('logging_level', 'INFO')
    configure_logging(
        log_mode=args.log,
        log_dir=config['directories']['log_dir'],
        config_name=config['name'],
        level=log_level,
    )
    logger = get_logger('SPICE', spice_prefix=False)

    name = config['name']
    plots_base_dir = os.path.join(config['directories']['plot_dir'], name)
    if not os.path.exists(plots_base_dir):
        logger.info(f"Creating directory {plots_base_dir}")
        os.makedirs(plots_base_dir)

    logger.info('Running SPICE: Plotting Mode')
    logger.info(f'Plotting for project name {name} with config file {args.config_path}')
    _apply_seed(args, logger)

    # Load required inputs based on mode
    if args.plot_events_per_sample is not None:
        chrom_segments_file = resolve_copynumber_file()
        chrom_segments = pd.read_csv(
            chrom_segments_file, sep='\t', index_col=['sample_id', 'chrom', 'allele']).sort_index()
        final_events_df = load_final_events()
        cur_sample = args.plot_events_per_sample
        logger.info(f'Plotting inferred events for sample: {cur_sample}')
        fig, axs = spice_plot.plot_inferred_events_per_sample(
            cur_sample,
            chrom_segments,
            final_events_df,
            unit_size=args.plot_unit_size,
        )
        out_path = os.path.join(plots_base_dir, f'{cur_sample}_events{"_unit_size" if args.plot_unit_size else ""}.png')
        fig.savefig(out_path, bbox_inches='tight')
        logger.info(f'Saved plot to {out_path}')
    elif args.plot_events_per_id is not None:
        chrom_segments_file = resolve_copynumber_file()
        chrom_segments = pd.read_csv(
            chrom_segments_file, sep='\t', index_col=['sample_id', 'chrom', 'allele']).sort_index()
        final_events_df = load_final_events()
        cur_id = args.plot_events_per_id
        logger.info(f'Plotting inferred events for id: {cur_id}')
        # Derive WIDTH_FULL from matplotlib defaults if not provided
        WIDTH_FULL = plt.rcParams.get('figure.figsize', (15, 5))[0]
        fig = spice_plot.plot_inferred_events_per_id(
            cur_id,
            chrom_segments,
            final_events_df,
            single_row=True,
            show_legend=True,
            figsize=(WIDTH_FULL, 1.25/5*WIDTH_FULL),
            lw=3,
            markersize=4
        )
        safe_id = cur_id.replace(':', '_')
        out_path = os.path.join(plots_base_dir, f'{safe_id}_events.png')
        fig.savefig(out_path, bbox_inches='tight')
        logger.info(f'Saved plot to {out_path}')
    elif args.plot_loci_on_chrom is not None:
        from spice.tsg_og.detection import convolution_simulation_per_ls
        
        cur_chrom = args.plot_loci_on_chrom
        detection_assignment = args.loci_mode
        output_dir = os.path.join(config['directories']['results_dir'], name, 'loci_of_selection')
        
        logger.info(f'Plotting all loci on {cur_chrom} ({detection_assignment} mode)')
        
        data_per_ls = open_pickle(os.path.join(output_dir, 'data_per_length_scale', f'{cur_chrom}.pickle'))
        selection_points, combined_fit = _load_plot_selection_points(
            output_dir, detection_assignment, cur_chrom)
        
        simulated_conv = convolution_simulation_per_ls(
            cur_chrom, data_per_ls, selection_points)
        fig, axs = plt.subplots(figsize=(25, 15), nrows=4, ncols=1, sharex=True)
        spice_plot.plot_tsg_og_results(
            cur_chrom, data_per_ls, simulated_conv=simulated_conv,
            plot_signal_bounds=True, fig=fig,
            final_selection_points=selection_points)
        
        out_path = os.path.join(plots_base_dir, f'{cur_chrom}_loci_{detection_assignment}.png')
        fig.savefig(out_path, bbox_inches='tight')
        logger.info(f'Saved plot to {out_path}')
    elif args.plot_single_locus is not None:
        from spice.tsg_og.detection import convolution_simulation_per_ls
        
        detection_assignment = args.loci_mode
        final_loci_df = pd.read_csv(
            os.path.join(config['directories']['results_dir'], name, f'final_loci_{detection_assignment}.tsv'),
            sep='\t', index_col=0)
        
        loci_index = args.plot_single_locus
        if loci_index not in final_loci_df.index:
            raise ValueError(f'Locus index {loci_index} not found in final loci dataframe for {detection_assignment} mode. Available indices: {final_loci_df.index.tolist()}')
        cur_locus = final_loci_df.loc[loci_index]
        cur_chrom = cur_locus['chrom']
        output_dir = os.path.join(config['directories']['results_dir'], name, 'loci_of_selection')
        
        logger.info(f'Plotting locus index {loci_index} on {cur_chrom} ({detection_assignment} mode)')
        
        data_per_ls = open_pickle(os.path.join(output_dir, 'data_per_length_scale', f'{cur_chrom}.pickle'))
        selection_points, combined_fit = _load_plot_selection_points(
            output_dir, detection_assignment, cur_chrom)
        simulated_conv = convolution_simulation_per_ls(
            cur_chrom, data_per_ls, selection_points)
        
        if combined_fit:
            # Original ranks can have gaps after filtering; the combined fit is compact.
            chrom_rows = final_loci_df[final_loci_df.chrom == cur_chrom].sort_values('rank_on_chrom')
            cluster_i = chrom_rows.index.get_loc(loci_index)
        else:
            cluster_i = int(final_loci_df.loc[loci_index, 'rank_on_chrom'])
        fig, axs = plt.subplots(figsize=(40, 13), nrows=1, ncols=4)
        spice_plot.plot_tsg_og_results(
            cur_chrom, data_per_ls, simulated_conv=simulated_conv,
            cluster_i=cluster_i, relative_window_size=3,
            orientation='v', fig=fig, xlim=(1e7, 5e7),
            final_selection_points=selection_points)
        
        out_path = os.path.join(plots_base_dir, f'{cur_chrom}_locus_{loci_index}_{detection_assignment}.png')
        fig.savefig(out_path, bbox_inches='tight')
        logger.info(f'Saved plot to {out_path}')

    logger.info('Done plotting.')



def _permutation_unit_dir(loci_results_dir, seed):
    return os.path.join(loci_results_dir, 'permutations', f's{seed}')


def _invalidate_permutation_tables(loci_results_dir, index):
    """Invalidate derived tables before a unit can change, including concurrent scatter jobs."""
    from spice.tsg_og.permutation import NULL_FILENAME
    for path in (os.path.join(_permutation_unit_dir(loci_results_dir, index), 'unit_loci.tsv'),
                 os.path.join(loci_results_dir, NULL_FILENAME)):
        try:
            os.remove(path)
        except FileNotFoundError:
            pass


def _check_permutation_chrom_mode(unit_dir, chrom, mode, write=False):
    """Keep scattered hybrid fits from being mixed with old arm-null caches."""
    import json
    marker = os.path.join(unit_dir, f'permutation_mode_{chrom}.json')
    if os.path.exists(marker):
        with open(marker) as handle:
            recorded = json.load(handle)
        if recorded != {'mode': mode, 'chrom': chrom}:
            raise ValueError(f'Permutation cache mode mismatch for {chrom}; use a fresh output directory')
    elif mode == 'chromosome_hybrid':
        old_cache = (os.path.exists(os.path.join(unit_dir, 'detection', chrom)) or
                     os.path.exists(os.path.join(unit_dir, 'data_per_length_scale', f'{chrom}.pickle')))
        if not write or old_cache:
            raise ValueError(f'Hybrid permutation cache lacks provenance for {chrom}; use a fresh output directory')
    if write:
        os.makedirs(unit_dir, exist_ok=True)
        with open(marker, 'w') as handle:
            json.dump({'mode': mode, 'chrom': chrom}, handle)
            handle.write('\n')


def _check_permutation_combine_modes(unit_dir, mode):
    """Validate exactly the cache scope used by combine_loci, including eventless chromosomes."""
    from spice.main_loci_functions import cached_loci_chromosomes
    for chrom in cached_loci_chromosomes(unit_dir):
        _check_permutation_chrom_mode(unit_dir, chrom, mode)


def _run_permutation_unit(raw_events, loci_params, loci_results_dir, chroms, seed, permute_mode,
                          steps, args, config):
    """Detect loci on ONE positionally-permuted copy of the cohort; return its loci table.

    This is the unit of work the null is built from, and it deliberately goes through the SAME
    entry points as the real run -- process_final_events_for_loci_routines, then
    run_loci_detection_per_chrom per chromosome, then combine_loci with the p-value off. That is
    the whole point of the permutation null: its loci are produced by the identical cascade
    (including every event-preprocessing and filtering step), so the fitness statistic is
    comparable to the observed one. Hybrid mode preprocesses before permutation
    and passes that fixed event set directly into detection and combination.
    """
    from spice.main_loci_functions import run_loci_detection_per_chrom, combine_loci
    from spice.logging import get_logger
    from spice.random_state import derive_seed
    from spice.tsg_og import permutation
    logger = get_logger('SPICE', spice_prefix=False)

    processed, n_moved, n_fixed = permutation.prepare_permutation_events(
        raw_events, seed=derive_seed('permutation', seed), mode=permute_mode, loci_params=loci_params)
    logger.info(f'  [permutation s{seed}] moved {n_moved:,} internal events'
                + (f', left {n_fixed:,} internal events fixed' if n_fixed else ''))
    unit_dir = _permutation_unit_dir(loci_results_dir, seed)
    # Reject leftover incompatible caches before invalidating tables or fitting.
    _check_permutation_combine_modes(unit_dir, permute_mode)
    for chrom in chroms:
        _check_permutation_chrom_mode(unit_dir, chrom, permute_mode, write=True)
    _invalidate_permutation_tables(loci_results_dir, seed)
    for chrom in chroms:
        run_loci_detection_per_chrom(
            final_events_df=processed, cur_chrom=chrom, which=steps,
            overwrite=args.overwrite,
            overwrite_preprocessing=(loci_params['overwrite_preprocessing'] and args.overwrite),
            name=config['name'],
            N_loci=loci_params['N_loci'], N_loci_spacing=loci_params.get('N_loci_spacing'),
            loci_results_dir=unit_dir,
            skip_up_down=loci_params['skip_up_down'], N_bootstrap=loci_params['N_bootstrap'],
            N_kernel=loci_params['N_kernel'], use_original_rank=loci_params['use_original_rank'],
            detection_N_iterations_base=loci_params['detection_N_iterations_base'],
            detection_max_N_iterations=loci_params['detection_max_N_iterations'],
            detection_final_N_iterations=loci_params['detection_final_N_iterations'],
            detection_blocked_distance_th=loci_params['detection_blocked_distance_th'],
            ranking_N_iterations=loci_params['ranking_N_iterations'],
            flipping_N_iterations=loci_params['flipping_N_iterations'],
            flipping_N_iterations_single=loci_params['flipping_N_iterations_single'],
            limiting_N_iterations_optim=loci_params['limiting_N_iterations_optim'],
            optimizing_N_iterations_optimization=loci_params['optimizing_N_iterations_optimization'],
            infer_widths_N_iterations=loci_params['infer_widths_N_iterations'],
            merge_N_iterations_optim=loci_params['merge_N_iterations_optim'],
            filter_N_iterations_optim=loci_params['filter_N_iterations_optim'],
            final_limiting_N_iterations_optim=loci_params['final_limiting_N_iterations_optim'],
            N_bootstrap_for_widths=loci_params['N_bootstrap_for_widths'],
            within_ci_N_iterations=loci_params['within_ci_N_iterations'],
            th_locus_prominence=loci_params['th_locus_prominence'],
            th_locus_mean_fitness=loci_params['th_locus_mean_fitness'],
        )
    _check_permutation_combine_modes(unit_dir, permute_mode)
    loci_df, _, _, _ = combine_loci(loci_results_dir=unit_dir, processed_events=processed,
                                 calculate_p_value=False, mode='detection')
    loci_df['permutation_mode'] = permute_mode
    out = os.path.join(unit_dir, 'unit_loci.tsv')
    loci_df.to_csv(out, sep='\t', index=False)
    return loci_df


def _permutation_detection_steps(requested, configured):
    """A fresh null needs a complete cascade even when the observed run only resumes/combines."""
    for candidate in (requested, configured):
        if not isinstance(candidate, str):
            candidate = [step for step in candidate if step != 'combine']
            if len(candidate) == 1:
                candidate = candidate[0]
        if isinstance(candidate, str):
            if candidate in ('fast', 'full'):
                return candidate
            if candidate == 'detection+':
                return 'full'
        elif candidate and candidate[0] == 'detection' and 'final_loci_widths' in candidate:
            return candidate
    raise ValueError('Building a permutation null needs a complete detection cascade. Set '
                     'loci_detection.loci_steps to fast, full, or a complete list of stages; '
                     'use --loci-steps combine only as a command-line override.')


def _build_permutation_null(raw_events, config, loci_params, loci_results_dir, chroms, K,
                            permute_mode, steps, args):
    """Build the pooled permutation null inline: K permuted cohorts, detected and pooled.

    Serial by design here -- each unit is itself a full genome detection pass, so the useful
    parallelism is across units on a cluster (`spice permute --index I --chrom C` + `--pool`), not
    across threads inside one process.
    """
    from spice.logging import get_logger
    from spice.tsg_og import permutation
    logger = get_logger('SPICE', spice_prefix=False)
    frames = []
    for seed in range(1, K + 1):
        logger.info(f'Permutation {seed}/{K}')
        frames.append(_run_permutation_unit(raw_events, loci_params, loci_results_dir, chroms,
                                            seed, permute_mode, steps, args, config))
    return permutation.null_from_loci(frames)


def main_permute(args):
    """Build the positional-permutation null (`spice permute`).

    Three usages, all writing under <loci_results_dir>/permutations/:
      spice permute --config c.yaml                  build the whole null in-process, then pool
      spice permute --config c.yaml --index 3 --chrom chr7    one scatter unit
      spice permute --config c.yaml --pool           pool the units already on disk

    The unit path exists because each permutation is itself a full detection pass, so the useful
    parallelism is across units on a cluster rather than across threads in one process.
    """
    spice.load_config(args.config_path)
    from spice import config
    from spice.logging import configure_logging, get_logger
    from spice.main_loci_functions import run_loci_detection_per_chrom, combine_loci
    from spice.data_loaders import load_final_events
    from spice.random_state import derive_seed
    from spice.tsg_og import permutation

    if 'name' not in config or not config['name']:
        raise ValueError("Config file must specify a 'name' field.")
    log_level = 'DEBUG' if args.debug else config['params'].get('logging_level', 'INFO')
    configure_logging(log_mode=args.log, log_dir=config['directories']['log_dir'],
                      config_name=config['name'], level=log_level)
    logger = get_logger('SPICE', spice_prefix=False)
    logger.info('Running SPICE: Permutation-Null Mode')
    _apply_seed(args, logger)

    loci_params = config['loci_detection']
    loci_results_dir = os.path.join(config['directories']['results_dir'], config['name'],
                                    'loci_of_selection')
    os.makedirs(loci_results_dir, exist_ok=True)
    K = args.permutations or int(loci_params.get('p_values_K', permutation.DEFAULT_K))
    mode = args.mode or loci_params.get('p_values_permute_mode', 'rotate')
    permutation.validate_permutation_strategy(mode, loci_params.get('p_values_strategy', 'zpool'))
    steps = args.loci_steps or loci_params['loci_steps']
    if hasattr(steps, '__iter__') and not isinstance(steps, str) and len(steps) == 1:
        steps = steps[0]
    perm_root = os.path.join(loci_results_dir, 'permutations')
    null_path = os.path.join(loci_results_dir, permutation.NULL_FILENAME)

    # ---- pool-only: no detection, just combine the units already on disk ----
    if args.pool:
        unit_dirs = sorted(glob.glob(os.path.join(perm_root, 's*')),
                           key=lambda d: int(os.path.basename(d)[1:]))
        if not unit_dirs:
            raise SystemExit(f'no permutation units under {perm_root} -- run the units first')
        raw_for_pool = None
        frames = []
        for d in unit_dirs:
            idx = int(os.path.basename(d)[1:])
            f = os.path.join(d, 'unit_loci.tsv')
            if args.overwrite or not os.path.exists(f):
                # A scattered `--index N --chrom C` run detects but does not combine, so the unit
                # table may be missing. Rebuild it here: the permutation is deterministic (its
                # stream derives from the base seed and the index), so re-deriving the permuted
                # events costs seconds and reproduces exactly what the scatter detected.
                logger.info(f'  s{idx}: combining its per-chromosome results')
                if raw_for_pool is None:
                    raw_for_pool = load_final_events()
                processed, _, _ = permutation.prepare_permutation_events(
                    raw_for_pool, seed=derive_seed('permutation', idx), mode=mode, loci_params=loci_params)
                _check_permutation_combine_modes(d, mode)
                loci_df, _, _, _ = combine_loci(loci_results_dir=d, processed_events=processed,
                                             calculate_p_value=False, mode='detection')
                loci_df['permutation_mode'] = mode
                loci_df.to_csv(f, sep='\t', index=False)
            unit_frame = pd.read_csv(f, sep='\t')
            permutation.validate_null_mode(unit_frame, mode)
            # Untagged historical units remain usable with legacy modes only.
            unit_frame['permutation_mode'] = mode
            frames.append(unit_frame)
        null_df = permutation.null_from_loci(frames)
        null_df.to_csv(null_path, sep='\t', index=False)
        logger.info(f'Pooled {len(frames)} permutation units -> {len(null_df):,} null loci '
                    f'at {null_path}')
        return

    if args.chrom is not None and args.index is None:
        raise ValueError('--chrom names a unit within one permutation and requires --index')

    events_df = load_final_events()
    chroms_all = [c for c in sorted(events_df['chrom'].unique(),
                                    key=lambda x: (len(x), x)) if c != 'chrY']

    # ---- one unit: a single (seed, chrom) so a cluster can scatter ----
    if args.index is not None and args.chrom is not None:
        processed, n_moved, n_fixed = permutation.prepare_permutation_events(
            events_df, seed=derive_seed('permutation', args.index), mode=mode, loci_params=loci_params)
        logger.info(f'Permutation s{args.index} ({mode}): moved {n_moved:,} internal events'
                    + (f', left {n_fixed:,} internal events fixed' if n_fixed else ''))
        unit_dir = _permutation_unit_dir(loci_results_dir, args.index)
        _check_permutation_chrom_mode(unit_dir, args.chrom, mode, write=True)
        _invalidate_permutation_tables(loci_results_dir, args.index)
        _detect_one(run_loci_detection_per_chrom, processed, args.chrom, steps, loci_params,
                    unit_dir, config, args)
        logger.info(f'Unit s{args.index}/{args.chrom} complete. Once every (index, chrom) unit is '
                    f'done, `spice permute --pool` combines each permutation and pools them.')
        return

    # ---- one whole permutation, or all K ----
    seeds = [args.index] if args.index is not None else list(range(1, K + 1))
    frames = []
    for seed in seeds:
        logger.info(f'Permutation {seed}' + (f'/{K}' if args.index is None else ''))
        frames.append(_run_permutation_unit(events_df, loci_params, loci_results_dir, chroms_all,
                                            seed, mode, steps, args, config))
    if args.index is not None:
        logger.info(f'Permutation s{args.index} complete; pool with `spice permute --pool`.')
        return
    null_df = permutation.null_from_loci(frames)
    null_df.to_csv(null_path, sep='\t', index=False)
    logger.info(f'Built the permutation null from {K} permutations: {len(null_df):,} loci '
                f'-> {null_path}')


def _detect_one(run_loci_detection_per_chrom, processed, chrom, steps, loci_params, out_dir,
                config, args):
    """One chromosome of detection into `out_dir`, with the run's own detection parameters."""
    run_loci_detection_per_chrom(
        final_events_df=processed, cur_chrom=chrom, which=steps, overwrite=args.overwrite,
        overwrite_preprocessing=(loci_params['overwrite_preprocessing'] and args.overwrite),
        name=config['name'], N_loci=loci_params['N_loci'],
        N_loci_spacing=loci_params.get('N_loci_spacing'), loci_results_dir=out_dir,
        skip_up_down=loci_params['skip_up_down'], N_bootstrap=loci_params['N_bootstrap'],
        N_kernel=loci_params['N_kernel'], use_original_rank=loci_params['use_original_rank'],
        detection_N_iterations_base=loci_params['detection_N_iterations_base'],
        detection_max_N_iterations=loci_params['detection_max_N_iterations'],
        detection_final_N_iterations=loci_params['detection_final_N_iterations'],
        detection_blocked_distance_th=loci_params['detection_blocked_distance_th'],
        ranking_N_iterations=loci_params['ranking_N_iterations'],
        flipping_N_iterations=loci_params['flipping_N_iterations'],
        flipping_N_iterations_single=loci_params['flipping_N_iterations_single'],
        limiting_N_iterations_optim=loci_params['limiting_N_iterations_optim'],
        optimizing_N_iterations_optimization=loci_params['optimizing_N_iterations_optimization'],
        infer_widths_N_iterations=loci_params['infer_widths_N_iterations'],
        merge_N_iterations_optim=loci_params['merge_N_iterations_optim'],
        filter_N_iterations_optim=loci_params['filter_N_iterations_optim'],
        final_limiting_N_iterations_optim=loci_params['final_limiting_N_iterations_optim'],
        N_bootstrap_for_widths=loci_params['N_bootstrap_for_widths'],
        within_ci_N_iterations=loci_params['within_ci_N_iterations'],
        th_locus_prominence=loci_params['th_locus_prominence'],
        th_locus_mean_fitness=loci_params['th_locus_mean_fitness'])

def main_loci_detection(args):
    """Run loci detection mode (de-novo)."""
    # Load configuration
    spice.load_config(args.config_path)
    from spice import config
    from spice.logging import configure_logging, get_logger

    if 'name' not in config or not config['name']:
        raise ValueError("Config file must specify a 'name' field.")

    loci_results_dir = os.path.join(config['directories']['results_dir'], config['name'], 'loci_of_selection')
    os.makedirs(loci_results_dir, exist_ok=True)

    # Create logger
    log_level = 'DEBUG' if args.debug else config['params'].get('logging_level', 'INFO')
    configure_logging(
        log_mode=args.log,
        log_dir=config['directories']['log_dir'],
        config_name=config['name'],
        level=log_level,
    )
    logger = get_logger('SPICE', spice_prefix=False)

    logger.info('Running SPICE: Loci Detection Mode (De-Novo)')
    logger.info(f'Project name: {config["name"]}')
    _apply_seed(args, logger)

    # Use the loci detection pipeline
    from spice.main_loci_functions import run_loci_detection_per_chrom, process_final_events_for_loci_routines
    from spice.data_loaders import load_final_events
    from spice.tsg_og import permutation
    
    # Get loci detection parameters from config
    loci_params = config['loci_detection']
    final_events_df = load_final_events()

    logger.info('Processing final events for loci detection')
    processed_events = process_final_events_for_loci_routines(
        final_events_df=final_events_df,
        remove_plateaus=loci_params.get('remove_plateaus', True),
        remove_chrY=loci_params.get('remove_chrY', True),
        drop_duplicates=loci_params.get('drop_duplicates', True),
        use_observed_centromeres=loci_params.get('use_observed_centromeres', True),
    )
    
    chromosomes = processed_events['chrom'].unique()
    assert set(chromosomes).issubset(set(['chr' + str(x) for x in range(1, 23)] + ['chrX', 'chrY'])), (
        f"Unexpected chromosomes in final events: {set(chromosomes) - set(['chr' + str(x) for x in range(1, 23)] + ['chrX', 'chrY'])}"
    )
    logger.info(f'Found {len(chromosomes)} unique chromosomes in final events: {chromosomes}')
    
    # Check sample count and warn if too low
    n_samples = processed_events['sample'].nunique()
    if n_samples < 1000:
        logger.warning('='*80)
        logger.warning('!!!  WARNING: LOW SAMPLE COUNT DETECTED !!!')
        logger.warning(f'Only {n_samples} samples found in processed events.')
        logger.warning('We recommend at least 1000 samples for reliable results.')
        logger.warning('Results may be unreliable with fewer samples.')
        logger.warning('='*80)

    if 'loci_steps' in args and args.loci_steps is not None:
        steps_to_run = args.loci_steps
    else:
        steps_to_run = loci_params['loci_steps']
    if hasattr(steps_to_run, '__iter__') and len(steps_to_run) == 1:
        steps_to_run = steps_to_run[0]
    logger.info(f'Running the following loci detection steps: {steps_to_run}')

    # `--chrom` runs a single chromosome as a parallel scatter unit (detection + its fitness p-value
    # part), skipping the cross-chromosome combine.
    if args.chrom is not None:
        if args.chrom not in list(chromosomes):
            raise ValueError(f'--chrom {args.chrom} not among detected chromosomes: {list(chromosomes)}')
        chromosomes = [args.chrom]
        logger.info(f'Restricting to a single chromosome (scatter unit): {args.chrom}')
    # --chrom is a per-chromosome scatter unit (detection + p-value part); with 'combine' the loop
    # below skips every chromosome and then returns early -> a green run that produces nothing.
    # Reject it rather than silently no-op.
    if args.chrom is not None and steps_to_run == "combine":
        raise ValueError("--chrom runs one chromosome's detection and is incompatible "
                         "with --loci-steps combine (the cross-chromosome combine runs without --chrom).")
    calc_p = loci_params.get('calculate_p_value', True)
    p_values_K = int(loci_params.get('p_values_K', permutation.DEFAULT_K))
    p_values_strategy = loci_params.get('p_values_strategy', 'zpool')
    p_values_permute_mode = loci_params.get('p_values_permute_mode', 'rotate')
    if calc_p:
        permutation.validate_permutation_strategy(p_values_permute_mode, p_values_strategy)
    p_thresh = loci_params['p_value_threshold'] # loci with q_value >= this are dropped
    # Absolute floor on the same statistic the p-value ranks, applied POST-NULL beside that drop.
    # Absent/None = no floor. Distinct from detection's `th_locus_mean_fitness`, which filters the
    # permutation null too -- see the note in combine_loci.
    mean_fit_thresh = loci_params.get('mean_fitness_threshold')
    if calc_p:
        logger.info(f'Fitness p-value (permutation null): K={p_values_K}, '
                    f'strategy={p_values_strategy}, permute_mode={p_values_permute_mode}; '
                    f'keeping loci with q_value < {p_thresh}')
    else:
        logger.info('Fitness p-value disabled (calculate_p_value=false)')

    for chrom in chromosomes:
        if steps_to_run == "combine":
            continue
        logger.info(f'Processing {chrom}...')
        run_loci_detection_per_chrom(
            final_events_df=processed_events,
            cur_chrom=chrom,
            which=steps_to_run,
            overwrite=args.overwrite,
            overwrite_preprocessing=(loci_params['overwrite_preprocessing'] and args.overwrite),
            name=config['name'],
            N_loci=loci_params['N_loci'],
            N_loci_spacing=loci_params.get('N_loci_spacing'),   # overrides N_loci per chromosome
            loci_results_dir=loci_results_dir,
            skip_up_down=loci_params['skip_up_down'],
            N_bootstrap=loci_params['N_bootstrap'],
            N_kernel=loci_params['N_kernel'],
            use_original_rank=loci_params['use_original_rank'],
            detection_N_iterations_base=loci_params['detection_N_iterations_base'],
            detection_max_N_iterations=loci_params['detection_max_N_iterations'],
            detection_final_N_iterations=loci_params['detection_final_N_iterations'],
            detection_blocked_distance_th=loci_params['detection_blocked_distance_th'],
            ranking_N_iterations=loci_params['ranking_N_iterations'],
            flipping_N_iterations=loci_params['flipping_N_iterations'],
            flipping_N_iterations_single=loci_params['flipping_N_iterations_single'],
            limiting_N_iterations_optim=loci_params['limiting_N_iterations_optim'],
            optimizing_N_iterations_optimization=loci_params['optimizing_N_iterations_optimization'],
            infer_widths_N_iterations=loci_params['infer_widths_N_iterations'],
            merge_N_iterations_optim=loci_params['merge_N_iterations_optim'],
            filter_N_iterations_optim=loci_params['filter_N_iterations_optim'],
            final_limiting_N_iterations_optim=loci_params['final_limiting_N_iterations_optim'],
            N_bootstrap_for_widths=loci_params['N_bootstrap_for_widths'],
            within_ci_N_iterations=loci_params['within_ci_N_iterations'],
            th_locus_prominence=loci_params['th_locus_prominence'],
            th_locus_mean_fitness=loci_params['th_locus_mean_fitness'],
        )

    if args.chrom is not None:
        logger.info(f'Per-chromosome step for {args.chrom} complete (detection); skipping combine. The '
                    f'permutation null is built once per cohort, at combine time or by `spice permute`.')
        return

    if not (steps_to_run in ['fast', 'full', 'combine'] or 'combine' in steps_to_run or '+' in steps_to_run):
        logger.info(steps_to_run)
        logger.warning("Loci detection steps do not include 'combine'. Final combination of loci across chromosomes will be skipped.")
        return

    # Combine results from all chromosomes
    logger.info('Combining all loci detection results across chromosomes')
    from spice.main_loci_functions import combine_loci # has to be imported here
    # The null is built ONCE per cohort, not per chromosome: it is the pooled set of loci detection
    # finds on K positionally-permuted copies of the events. Reuse a null a previous `spice permute`
    # wrote when one is present, else build it inline -- which costs K full detection passes, so for
    # a genome-scale cohort prefer `spice permute` scattered over a cluster.
    null_df = None
    if calc_p:
        null_path = os.path.join(loci_results_dir, permutation.NULL_FILENAME)
        if os.path.exists(null_path) and not args.overwrite:
            null_df = pd.read_csv(null_path, sep='\t')
            permutation.validate_null_mode(null_df, p_values_permute_mode)
            logger.info(f'Loaded permutation null: {len(null_df):,} loci from {null_path}')
        else:
            logger.info(f'No permutation null at {null_path}; building it inline (K={p_values_K})')
            null_df = _build_permutation_null(
                raw_events=final_events_df, config=config, loci_params=loci_params,
                loci_results_dir=loci_results_dir, chroms=list(chromosomes), K=p_values_K,
                permute_mode=p_values_permute_mode,
                steps=_permutation_detection_steps(steps_to_run, loci_params['loci_steps']),
                args=args)
            null_df.to_csv(null_path, sep='\t', index=False)
            logger.info(f'Wrote permutation null ({len(null_df):,} loci) to {null_path}')
    final_loci_df, filtered_selection_points, filtered_loci_widths, unfiltered_loci_df = combine_loci(
        loci_results_dir=loci_results_dir,
        processed_events=processed_events,
        calculate_p_value=calc_p,
        p_value_threshold=loci_params['p_value_threshold'],
        mean_fitness_threshold=mean_fit_thresh,
        permutation_null=null_df,
        p_values_strategy=p_values_strategy,
        overwrite=args.overwrite,
        mode='detection',
        final_reoptimization_N_iterations=loci_params['final_reoptimization_N_iterations'],
    )

    # Save final combined loci results
    final_loci_output_path = os.path.join(config['directories']['results_dir'], config['name'], 'final_loci_detection.tsv')
    final_loci_df.to_csv(final_loci_output_path, sep='\t', index=True)
    logger.info(f'Saved final combined loci detection results to {final_loci_output_path}')
    # ... and the PRE-DROP table beside it. Whenever a threshold actually removes loci, the canonical
    # table is the CALLED set, which is the wrong input for a calibration read: the QQ, cumulative,
    # p-value-histogram and length-scale figures need every detected locus with its p/q. Identical to
    # the canonical table on a cohort that drops nothing.
    unfiltered_output_path = os.path.join(config['directories']['results_dir'], config['name'],
                                          'final_loci_detection_unfiltered.tsv')
    unfiltered_loci_df.to_csv(unfiltered_output_path, sep='\t', index=True)
    logger.info(f'Saved the pre-filter loci table ({len(unfiltered_loci_df)} loci) to {unfiltered_output_path}')
    save_pickle(filtered_selection_points, os.path.join(config['directories']['results_dir'], config['name'], 'loci_of_selection', 'detection', 'final_loci_detection_filtered.pickle'))
    save_pickle(filtered_loci_widths, os.path.join(config['directories']['results_dir'], config['name'], 'loci_of_selection', 'detection', 'final_loci_detection_filtered_widths.pickle'))

    # Genome-wide diagnostic: how much of the signal the (filtered, reoptimized) loci explain
    from spice.tsg_og.detection import calc_genome_wide_within_ci
    genome_wide_within_ci, within_ci_df = calc_genome_wide_within_ci(
        loci_results_dir, mode='detection', all_selection_points=filtered_selection_points)
    logger.info(f'Genome-wide fraction of signal within the bootstrap CI (filtered, reoptimized loci): {genome_wide_within_ci:.4f}')
    within_ci_output_path = os.path.join(config['directories']['results_dir'], config['name'], 'within_ci_detection.tsv')
    within_ci_df.to_csv(within_ci_output_path, sep='\t', index=True)
    logger.info(f'Saved per-chromosome within-CI fractions to {within_ci_output_path}')

    logger.info('Loci detection pipeline completed.')



def _load_permutation_null_or_none(loci_results_dir):
    """Read the pooled permutation null if `spice permute` has written one, else None.

    Assignment mode has no per-chromosome detection of its own to hang an inline build off, so it
    consumes a null built beforehand; combine_loci raises a pointed error if the p-value is on and
    no null is available.
    """
    from spice.tsg_og import permutation
    path = os.path.join(loci_results_dir, permutation.NULL_FILENAME)
    if os.path.exists(path):
        frame = pd.read_csv(path, sep='\t')
        mode = spice.config['loci_detection'].get('p_values_permute_mode', 'rotate')
        permutation.validate_permutation_strategy(mode, spice.config['loci_detection'].get('p_values_strategy', 'zpool'))
        permutation.validate_null_mode(frame, mode)
        return frame
    return None

def main_loci_assignment(args):
    """Run loci assignment mode (assign fitness to predefined loci)."""
    # Load configuration
    spice.load_config(args.config_path)
    from spice import config
    from spice.logging import configure_logging, get_logger

    if 'name' not in config or not config['name']:
        raise ValueError("Config file must specify a 'name' field.")

    # Create logger
    log_level = 'DEBUG' if args.debug else config['params'].get('logging_level', 'INFO')
    configure_logging(
        log_mode=args.log,
        log_dir=config['directories']['log_dir'],
        config_name=config['name'],
        level=log_level,
    )
    logger = get_logger('SPICE', spice_prefix=False)

    logger.info('Running SPICE: Loci Assignment Mode')
    logger.info(f'Project name: {config["name"]}')
    _apply_seed(args, logger)
    
    # Run loci assignment pipeline
    from spice.main_loci_functions import loci_assignment, process_final_events_for_loci_routines
    from spice.data_loaders import load_final_events
    
    # Get loci assignment parameters from config
    loci_params = config['loci_detection']
    final_events_df = load_final_events()

    logger.info('Processing final events for loci detection')
    processed_events = process_final_events_for_loci_routines(
        final_events_df=final_events_df,
        remove_plateaus=loci_params.get('remove_plateaus', True),
        remove_chrY=loci_params.get('remove_chrY', True),
        drop_duplicates=loci_params.get('drop_duplicates', True),
        use_observed_centromeres=loci_params.get('use_observed_centromeres', True),
    )
    
    # Check sample count and warn if too low
    n_samples = processed_events['sample'].nunique()
    if n_samples < 500:
        logger.warning('='*80)
        logger.warning('!!!  WARNING: LOW SAMPLE COUNT DETECTED !!!')
        logger.warning(f'Only {n_samples} samples found in processed events.')
        logger.warning('We recommend at least 500 samples for reliable results.')
        logger.warning('Results may be unreliable with fewer samples.')
        logger.warning('='*80)

    final_loci_df = loci_assignment(
        name=config['name'],
        processed_events=processed_events,
        N_bootstrap=loci_params['N_bootstrap'],
        N_kernel=loci_params['N_kernel'],
        within_ci_N_iterations=loci_params['loci_assignment_within_ci_N_iterations'],
        N_iterations_optim=loci_params['loci_assignment_N_iterations'],
        calculate_p_value=loci_params['calculate_p_value'],
        p_value_threshold=loci_params['p_value_threshold'],
        permutation_null=_load_permutation_null_or_none(
            os.path.join(config['directories']['results_dir'], config['name'],
                         'loci_of_selection')),
        p_values_strategy=loci_params.get('p_values_strategy', 'zpool'),
        overwrite=args.overwrite,
        overwrite_preprocessing=(loci_params['overwrite_preprocessing'] and args.overwrite),
    )

    # Save final combined loci results
    final_loci_output_path = os.path.join(config['directories']['results_dir'], config['name'], 'final_loci_assignment.tsv')
    final_loci_df.to_csv(final_loci_output_path, sep='\t', index=True)
    logger.info(f'Saved final combined loci assignment results to {final_loci_output_path}')

    logger.info('Loci assignment pipeline completed.')


def main():
    """Main CLI entry point for SPICE."""
    parser = argparse.ArgumentParser(
        description='SPICE: Selection Patterns In somatic Copy-number Events',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
        
    parser.epilog = """
Examples:
  # Event inference
  spice event_inference --config <path/to/config>
  spice event_inference --config <path/to/config> --event-steps split all_solutions
  spice event_inference --config <path/to/config> --cores 8
  spice event_inference --config <path/to/config> --clean
  
  # Plotting
  spice plotting --config <path/to/config> --plot-events-per-sample "sample_1"
  spice plotting --config <path/to/config> --plot-events-per-id "sample_1:chr1:cn_a"
  
  # Loci detection (de-novo)
  spice loci_detection --config <path/to/config>
  
  # Loci assignment (fitness assignment to predefined loci)
  spice loci_assignment --config <path/to/config>
    """
    
    # Create subparsers for different modes
    subparsers = parser.add_subparsers(
        dest='mode',
        required=True,
        help='SPICE mode to run'
    )
    
    # Common arguments shared by all modes
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument(
        '--config', '-c',
        required=True,
        type=str,
        dest='config_path',
        help='Path to a YAML config file to merge over defaults'
    )
    common_parser.add_argument(
        '--log',
        type=str,
        choices=['terminal', 'file', 'both'],
        default='terminal',
        help='Logging output mode: terminal (console only), file (log file only), or both (default: terminal)'
    )
    common_parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable DEBUG logging globally, overriding config logging_level'
    )
    common_parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Base RNG seed, overriding "seed" in the config params. Every random draw derives from '
             'it, so the same seed on the same input reproduces the run (see spice.random_state).'
    )
    
    # ===== EVENT INFERENCE SUBPARSER =====
    parser_event = subparsers.add_parser(
        'event_inference',
        parents=[common_parser],
        help='Run event inference pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Infer discrete copy-number events from allele-specific profiles'
    )
    parser_event.add_argument(
        '--event-steps',
        nargs='+',
        default=argparse.SUPPRESS,
        help='Steps to run: preprocessing, split, all_solutions, disambiguate, large_chroms, combine (default: all). Use a trailing + (e.g., split+) to run that step and all subsequent steps.'
    )
    parser_event.add_argument(
        '--cores', '-j',
        type=int,
        default=None,
        help='Number of cores to use for parallel processing (default: 1)'
    )
    parser_event.add_argument(
        '--keep-old',
        action='store_true',
        help='Keep old intermediate files instead of overwriting them'
    )
    parser_event.add_argument(
        '--clean',
        action='store_true',
        help='Clean intermediate files and exit'
    )
    parser_event.add_argument(
        '--ids',
        type=str,
        default=None,
        help='Comma-separated list of sample IDs to process'
    )
    parser_event.add_argument(
        '--run-preprocessing',
        action='store_true',
        help='Run the extra preprocessing step before split (default: off)'
    )
    parser_event.add_argument(
        '--pre-unique-chroms',
        dest='pre_unique_chroms',
        action='store_true',
        help='Preprocessing: keep only unique chromosomes'
    )
    parser_event.add_argument(
        '--pre-skip-phasing',
        dest='pre_skip_phasing',
        action='store_true',
        help='Preprocessing: skip MEDICC2 phasing'
    )
    parser_event.add_argument(
        '--pre-skip-centromeres',
        dest='pre_skip_centromeres',
        action='store_true',
        help='Preprocessing: skip centromere binning'
    )
    parser_event.set_defaults(func=main_event_inference)
    
    # ===== PLOTTING SUBPARSER =====
    parser_plot = subparsers.add_parser(
        'plotting',
        parents=[common_parser],
        help='Plot inferred events and loci',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Generate visualizations of inferred copy-number events and selection loci'
    )
    plot_group = parser_plot.add_mutually_exclusive_group(required=True)
    plot_group.add_argument(
        '--plot-events-per-sample',
        dest='plot_events_per_sample',
        type=str,
        help='Sample ID to plot events for'
    )
    plot_group.add_argument(
        '--plot-events-per-id',
        dest='plot_events_per_id',
        type=str,
        help='Chromosome allele ID to plot events for (format: sample:chr:cn_a|cn_b)'
    )
    plot_group.add_argument(
        '--plot-loci-on-chrom',
        dest='plot_loci_on_chrom',
        type=str,
        help='Chromosome to plot all loci for (e.g., chr1)'
    )
    plot_group.add_argument(
        '--plot-single-locus',
        dest='plot_single_locus',
        type=int,
        help='Locus index to plot (from final_loci_*.tsv)'
    )
    parser_plot.add_argument(
        '--plot-unit-size',
        dest='plot_unit_size',
        action='store_true',
        help='Use unit_size for plotting events (only for --plot-events-per-sample)'
    )

    parser_plot.add_argument(
        '--loci-mode',
        type=str,
        choices=['detection', 'assignment'],
        default='detection',
        help='Loci mode: detection or assignment (for loci plotting modes)'
    )
    parser_plot.set_defaults(func=main_plotting)
    
    # ===== LOCI DETECTION SUBPARSER =====
    parser_loci = subparsers.add_parser(
        'loci_detection',
        parents=[common_parser],
        help='Detect recurrent copy-number loci (de-novo mode)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Identify recurrent copy-number loci across chromosomes using de-novo detection'
    )
    parser_loci.add_argument(
        '--loci-steps',
        nargs='+',
        default=None,
        help='Steps to run. If not present will use "loci_steps" from config. Use "fast" for the accelerated subset, "full" for the full pipeline, or a trailing + (e.g., split+) to run that step and all subsequent steps.'
    )
    parser_loci.add_argument(
        '--cores', '-j',
        type=int,
        default=1,
        help='Parallel workers where the step supports them (default: 1)'
    )
    parser_loci.add_argument(
        '--overwrite',
        action='store_true',
        help='Run new and overwrite existing data'
    )
    parser_loci.add_argument(
        '--chrom',
        default=None,
        help='Run detection for this one chromosome only (a parallel scatter unit); skips the '
             'cross-chromosome combine. The permutation null is built once per cohort, not here.'
    )
    parser_loci.set_defaults(func=main_loci_detection)

    # ===== PERMUTATION-NULL SUBPARSER =====
    parser_perm = subparsers.add_parser(
        'permute',
        parents=[common_parser],
        help='Build the positional-permutation null for the fitness p-value',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Detect loci on positionally-permuted copies of the cohort and pool them into '
                    'the null the fitness p-value is read against. Run with no unit flags to build '
                    'the whole null in-process; use --index/--chrom for a single unit (so a cluster '
                    'can scatter over K x chromosomes) and then --pool to combine the units.'
    )
    parser_perm.add_argument('-k', '--permutations', type=int, default=None,
                             help=f'Number of permutations; default from the config key p_values_K '
                                  f'(fallback {_DEFAULT_K}). Do not lower it to save '
                                  f'time: the empirical p floors at 1/(pooled+1) and that floor '
                                  f'binds the FDR.')
    parser_perm.add_argument('--index', type=int, default=None,
                             help='Build only this permutation, 1-based. With --chrom, one scatter '
                                  'unit. Its RNG stream derives from the base --seed, so the null '
                                  'is reproducible and every index is a different permutation.')
    parser_perm.add_argument('--chrom', default=None,
                             help='Restrict a unit to this chromosome (requires --index).')
    parser_perm.add_argument('--pool', action='store_true',
                             help='Pool the per-unit tables already on disk into the null table '
                                  'and exit, without detecting anything.')
    parser_perm.add_argument('--mode', choices=('rotate', 'uniform', 'chromosome_hybrid'), default=None,
                             help="Positional model: 'rotate' shifts each (sample, chrom, arm) "
                                  "circularly, cutting only between events; 'uniform' places each "
                                  "event independently within its arm; chromosome_hybrid permits either arm "
                                  "and centromere-spanning long events. Default from p_values_permute_mode.")
    parser_perm.add_argument('--loci-steps', nargs='+', default=None,
                             help='Detection steps for the permuted cohorts; must match the real '
                                  'run or the null is not comparable. Default: the config value.')
    parser_perm.add_argument('--cores', '-j', type=int, default=1,
                             help='Parallel workers where supported (default: 1)')
    parser_perm.add_argument('--overwrite', action='store_true',
                             help='Rebuild units and the pooled null even if present')
    parser_perm.set_defaults(func=main_permute)

    # ===== LOCI ASSIGNMENT SUBPARSER =====
    parser_assign = subparsers.add_parser(
        'loci_assignment',
        parents=[common_parser],
        help='Assign fitness to predefined loci',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Assign fitness values to predefined loci positions'
    )
    parser_assign.add_argument(
        '--overwrite',
        action='store_true',
        help='Run new and overwrite existing data'
    )
    parser_assign.set_defaults(func=main_loci_assignment)
    
    # Parse arguments and call the appropriate function
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
