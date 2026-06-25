import os
from collections import namedtuple
import itertools

import pandas as pd
import numpy as np
import fire

from spice import config
from spice.data_loaders import load_sv_data
from spice.utils import (
    save_pickle, open_pickle, chrom_id_from_id,
    calc_telomere_bound_whole_arm_whole_chrom)
from spice.logging import get_logger, log_debug
from spice.event_inference.events_from_graph import (
    full_paths_from_graph_with_sv, create_events_df_from_single_path_solution)
from spice.event_inference.knn_graph import solve_with_knn, load_knn_train
from spice.event_inference.mcmc_for_large_chroms import mcmc_event_selection, create_best_events_df_from_mcmc
from spice.event_inference.data_structures import ChromData
from spice.pipeline_postprocessing import calc_summary_from_events_df
from spice.event_analysis.final_events import classify_event_position


logger = get_logger('spice.pipeline')


EVENT_DF_COLUMNS = [
    'sample', 'chrom', 'id', 'width', 'start', 'end', 'type', 'telomere_bound', 'whole_chrom', 'whole_arm',
    'diff', 'wgd', 'allele', 'chrom_id', 'events_per_chrom', 'chrom_length', 'solved', 'n_paths']
EVENT_DF_DTYPES = {
    'sample': "str",
    'chrom': "str",
    'id': "str",
    'width': "int64",
    'start': "int64",
    'end': "int64",
    'type': "str",
    'telomere_bound': "bool",
    'whole_chrom': "bool",
    'whole_arm': "bool",
    'diff': "str",
    'wgd': "str",
    'allele': "str",
    'chrom_id': "str",
    'events_per_chrom': "int64",
    'chrom_length': "int64",
    'solved': "str",
    'n_paths': "int64",
}


def full_paths_from_graph_with_sv_wrapper(
        cur_id, is_wgd, sv_data_file, chrom_segments_file, chrom_file,
        sv_matching_threshold=10, all_loh_solutions=True, total_cn=False, output_file=None,
        skip_loh_checks=False, use_cache=True, save_output=True):
    
    # Implemente here to debug the creation of the fail report
    # if cur_id == "sample_DEBUG:chr1:cn_a":
    #     raise ValueError("Debug ID")
    
    log_debug(logger, f"Full path solving for {cur_id} ({'WGD' if is_wgd else 'noWGD'})")
    chrom_id = chrom_id_from_id(cur_id)
    cur_sv_data = load_sv_data(sv_data_file=sv_data_file, chrom_id=chrom_id)
    chrom_segments = pd.read_csv(chrom_segments_file, sep='\t', index_col=['sample_id', 'chrom', 'allele'])
    chrom = open_pickle(chrom_file)

    full_paths = full_paths_from_graph_with_sv(
        cur_id, is_wgd, cur_sv_data, chrom_segments, chrom, path_limit=None,
        sv_matching_threshold=sv_matching_threshold, total_cn=total_cn,
        all_loh_solutions=all_loh_solutions, use_cache=use_cache, skip_loh_checks=skip_loh_checks)

    if len(full_paths.events) == 0 or len(full_paths.solutions) == 0:
        raise ValueError(f'No events (n = {len(full_paths.events)}) or no solutions (n = {len(full_paths.solutions)}) found')

    if full_paths.solved in ['sv', 'unamb']:
        output_file = output_file.replace('full_paths_multiple_solutions', 'full_paths_single_solution')
        log_debug(logger, f"Full path is completely solved with {full_paths.solved}")
        assert len(full_paths.solutions) == 1, f"solved_chrom.solutions: {full_paths.solutions}"
        output = create_events_df_from_single_path_solution(
            full_paths, cur_id, chrom_segments=chrom_segments, create_full=True, calc_telomere_bound=False)
    else:
        output = full_paths

    if save_output:
        assert output_file is not None, "output_file must be provided if save_output is True"
        log_debug(logger, f'saving file to {output_file}')
        save_pickle(output, output_file)
    else:
        return output


def create_knn_train_data(output_file, knn_train_data_ext=None, full_paths_single_solution_dirs=None,
                          knn_train_dataset='sv_unamb'):
    if knn_train_data_ext is not None and knn_train_data_ext != '':
        log_debug(logger, f'external knn_train_data provided at {knn_train_data_ext}, so using it')
        knn_train_data = open_pickle(knn_train_data_ext, fail_if_nonexisting=True)
    else:
        log_debug(logger, f'no external knn_train_data provided, so creating new one from files from {full_paths_single_solution_dirs}')
        assert full_paths_single_solution_dirs is not None
        if isinstance(full_paths_single_solution_dirs, str):
            full_paths_single_solution_dirs = full_paths_single_solution_dirs.split(' ')

        unamb_events = []
        for cur_dir in full_paths_single_solution_dirs:
            if not os.path.exists(cur_dir):
                logger.warning(f'{cur_dir} does not exist, skipping')
                continue
            for cur_file in os.listdir(cur_dir):
                if not cur_file.endswith('.pickle'):
                    continue
                cur_events = open_pickle(os.path.join(cur_dir, cur_file), fail_if_nonexisting=True)
                unamb_events.append(cur_events)
            
        unamb_events = pd.concat(unamb_events).reset_index(drop=True)
        assert unamb_events['solved'].isin(['sv', 'unamb']).all(), 'All events must be either unambiguous or solved by SVs'
        if knn_train_dataset not in ['sv', 'unamb', 'sv_unamb', 'unamb_sv']:
            raise ValueError(f"Invalid knn_train_dataset: {knn_train_dataset}. Must be in ['sv', 'unamb', 'sv_unamb', 'unamb_sv']")
        if knn_train_dataset == 'sv':
            log_debug(logger, 'selecting only SV solved events')
            unamb_events = unamb_events.query('solved == "sv"').copy()
        elif knn_train_dataset == 'unamb':
            log_debug(logger, 'selecting only unambiguous events')
            unamb_events = unamb_events.query('solved == "unamb"').copy()
        else:
            log_debug(logger, 'selecting all events (SV solved and unambiguous)')

        # recalculate this here because of bug in the pipeline (should be unnecessary as of 20.06.2024)
        unamb_events[['telomere_bound', 'whole_arm', 'whole_chrom']] = np.stack(
            calc_telomere_bound_whole_arm_whole_chrom(unamb_events), axis=1)

        log_debug(logger, f"Found {unamb_events['id'].nunique()} unique IDs with {len(unamb_events)} events")
        pre_post_wgds = ['nowgd', 'pre', 'post', 'all']

        knn_train_data = dict()
        for cur_type, telomere_bound, pre_post_wgd in itertools.product(['gain', 'loss'], [True, False], pre_post_wgds):        
            if pre_post_wgd == 'all':
                cur_mask_train = np.logical_and.reduce([
                    (unamb_events['type'] == cur_type),
                    (unamb_events['telomere_bound'] == telomere_bound),
                    (~unamb_events['whole_chrom']),
                    (~unamb_events['whole_arm'])
                ])
            else:
                cur_mask_train = np.logical_and.reduce([
                    (unamb_events['type'] == cur_type),
                    (unamb_events['telomere_bound'] == telomere_bound),
                    (unamb_events['wgd'] == pre_post_wgd),
                    (~unamb_events['whole_chrom']),
                    (~unamb_events['whole_arm'])
                ])
            knn_train_data[(cur_type, telomere_bound, pre_post_wgd)] = (
                unamb_events.loc[cur_mask_train, 'width'].values,
                unamb_events.loc[cur_mask_train, 'chrom_length'].values,
                unamb_events.loc[cur_mask_train, 'id'].values,
                )
    
    save_pickle(knn_train_data, output_file)


def solve_with_knn_wrapper(cur_id, chrom_segments_file, full_paths_multiple_solutions_dirs, is_wgd,
                           k=250, single_width_bin=True, perform_loh_checks=False,
                           save_all_scores=None, output_file=None):
    
    if isinstance(full_paths_multiple_solutions_dirs, str):
        full_paths_multiple_solutions_dirs = full_paths_multiple_solutions_dirs.split(' ')

    full_paths = open_pickle(os.path.join(full_paths_multiple_solutions_dirs[1 if is_wgd else 0], f'{cur_id}.pickle'), fail_if_nonexisting=True)
    assert getattr(full_paths, '__class__', None).__name__ == 'FullPaths', type(full_paths)
    # assert isinstance(full_paths, FullPaths) 

    cur_chrom_segments = pd.read_csv(chrom_segments_file, sep='\t', index_col=['sample_id', 'chrom', 'allele']).query('id == @cur_id')

    assert all([len(full_paths.events[0].diff) == len(full_paths.events[i].diff) for i in range(len(full_paths.events))]), 'All events must have the same length of diff string'
    assert len(cur_chrom_segments) == len(full_paths.events[0].diff)

    log_debug(logger, f"KNN-graph solving for {cur_id} ({'WGD' if is_wgd else 'noWGD'}). CN profile: {full_paths.cn_profile}, n_solutions: {full_paths.n_solutions}, n_events: {full_paths.n_events}")

    if full_paths.solved in ['sv', 'unamb']:
        raise ValueError('full_paths already solved')
    if len(full_paths.events) == 0:
        raise ValueError('full_paths does not have any events')

    knn_train_data = load_knn_train()

    final_events_df = solve_with_knn(
        full_paths,
        k=k,
        cur_chrom_segments=cur_chrom_segments,
        knn_train_data=knn_train_data,
        perform_loh_checks=perform_loh_checks,
        wgd_split=True,
        log10_distances=True,
        save_all_scores=save_all_scores,
        single_width_bin=single_width_bin)

    final_events_df = final_events_df[EVENT_DF_COLUMNS]

    if output_file is not None:
        save_pickle(final_events_df, output_file)
    else:
        return final_events_df


def solve_with_mcmc_wrapper(
        chrom_file, chrom_segments_file, sv_data_file, is_wgd, k=250, wgd_split=True,
        output_file=None, sv_matching_threshold=10, n_iterations=None, n_iteration_scale=100, perform_loh_checks=False,
        min_T=1, max_T=-6, swap_event_based_on_score=True, check_all_loh_solutions=False, total_cn=False,
        verbose=False, save_all_scores=None, log_progress=False, show_progress=False, fail_on_empty=True,
        skip_loh_check=True):
    assert (n_iterations is not None) ^ (n_iteration_scale is not None), 'Either n_iterations or n_iteration_scale must be provided'

    chrom_data = open_pickle(chrom_file, fail_if_nonexisting=True)
    assert isinstance(chrom_data, ChromData)
    cur_id = chrom_data.id

    log_debug(logger, f"MCMC solving for {cur_id} ({'WGD' if is_wgd else 'noWGD'})")
    assert is_wgd == chrom_data.has_wgd, f'is_wgd ({is_wgd}) does not match chrom_data.has_wgd ({chrom_data.has_wgd})'
    cur_chrom_segments = pd.read_csv(chrom_segments_file, sep='\t', index_col=['sample_id', 'chrom', 'allele']).query('id == @cur_id')

    if n_iterations is None:
        n_iterations = int(chrom_data.n_events * n_iteration_scale)

    knn_train_data = load_knn_train()

    chrom_id = chrom_id_from_id(cur_id)
    cur_sv_data = load_sv_data(sv_data_file=sv_data_file, chrom_id=chrom_id)
   
    mcmc_result = mcmc_event_selection(
                        cur_id=cur_id,
                        chrom_data=chrom_data,
                        chrom_segments=cur_chrom_segments,
                        sv_data=cur_sv_data,
                        acceptance_temp=None,
                        n_iterations=n_iterations,
                        knn_graph_k=k,
                        sv_matching_threshold=sv_matching_threshold,
                        knn_train_data=knn_train_data,
                        show_progress=show_progress,
                        simulated_annealing=True,
                        min_T=min_T,
                        max_T=max_T,
                        wgd_split=wgd_split,
                        loh_time_limit=None,
                        swap_event_based_on_score=swap_event_based_on_score,
                        perform_loh_checks=perform_loh_checks,
                        check_all_loh_solutions=check_all_loh_solutions,
                        total_cn=total_cn,
                        log_progress=log_progress,
                        calc_new_filename=None,
                        calc_new_verbose=False,
                        skip_loh_check=skip_loh_check
                        )
    
    # if it fails with SVs, rerun without them
    if mcmc_result is None and (cur_sv_data is not None and len(cur_sv_data) > 0):
        cur_sv_data = None
        log_debug(logger, 'No solution found with SVs, rerunning without SVs')
        mcmc_result = mcmc_event_selection(
                        cur_id=cur_id,
                        chrom_data=chrom_data,
                        chrom_segments=cur_chrom_segments,
                        sv_data=cur_sv_data,
                        acceptance_temp=None,
                        n_iterations=n_iterations,
                        knn_graph_k=k,
                        sv_matching_threshold=sv_matching_threshold,
                        knn_train_data=knn_train_data,
                        show_progress=show_progress,
                        simulated_annealing=True,
                        min_T=min_T,
                        max_T=max_T,
                        wgd_split=wgd_split,
                        loh_time_limit=None,
                        swap_event_based_on_score=swap_event_based_on_score,
                        check_all_loh_solutions=check_all_loh_solutions,
                        total_cn=total_cn,
                        perform_loh_checks=perform_loh_checks,
                        log_progress=log_progress,
                        calc_new_filename=None,
                        calc_new_verbose=False)

    if mcmc_result is not None:
        final_events_df = create_best_events_df_from_mcmc(mcmc_result, cur_chrom_segments, chrom_data, has_wgd=chrom_data.has_wgd)  
        final_events_df = final_events_df[EVENT_DF_COLUMNS]
    else:
        if fail_on_empty:
            raise ValueError('MCMC run failed, no solution was found.')
        final_events_df = None
   
    if save_all_scores is not None:
        log_debug(logger, f'Saving all scores to {save_all_scores}')
        save_pickle(mcmc_result, save_all_scores)

    if output_file is not None:
        log_debug(logger, f'Saving final events to {output_file}')
        save_pickle(final_events_df, output_file)

    return final_events_df

def combine_final_events(solved_dirs, chrom_segments_file=None, sv_data=None,
                         sv_matching_threshold=10, output_dir=None):

    if isinstance(solved_dirs, str):
        solved_dirs = solved_dirs.split(' ')

    chrom_segments = pd.read_csv(chrom_segments_file, sep='\t', index_col=['sample_id', 'chrom', 'allele'])

    solved_events = []
    for cur_dir_i, cur_dir in enumerate(solved_dirs):
        if not os.path.exists(cur_dir):
            log_debug(logger, f'{cur_dir} does not exist, skipping')
            continue
        cur_dir_name = '/'.join(cur_dir.split('/')[-2:])
        log_debug(logger, f'Directory {cur_dir_i+1}/{len(solved_dirs)}: {cur_dir}')
        n_files = len(os.listdir(os.path.join(cur_dir)))
        for i, cur_file in enumerate(os.listdir(os.path.join(cur_dir))):
            if not cur_file.endswith('.pickle'):
                continue
            log_debug(logger, f'Directory {cur_dir_i+1}/{len(solved_dirs)}, file {i+1}/{n_files} ({100*(i+1)/n_files:.2f}%): {cur_file} from {cur_dir_name}')
            cur_events = open_pickle(os.path.join(cur_dir, cur_file), fail_if_nonexisting=True)
            if cur_events is None:
                log_debug(logger, f'Data is None, skipping')
                continue
            solved_events.append(cur_events[EVENT_DF_COLUMNS].values)
    log_debug(logger, f'Found {len(solved_events)} solved IDs')

    if len(solved_events) == 0:
        raise ValueError(f'No solved events found in the provided directories: {solved_dirs}')
    final_events_df = pd.DataFrame(np.concatenate(solved_events, axis=0), columns=EVENT_DF_COLUMNS)
    log_debug(logger, f'Found a total of {len(final_events_df)} events in the final df')

    # missing_telomere_bound has to be before types are assigned
    missing_telomere_bound = np.logical_or(
        (final_events_df[['telomere_bound', 'whole_arm', 'whole_chrom']] == None).any(axis=1),
        (final_events_df[['telomere_bound', 'whole_arm', 'whole_chrom']].isna()).any(axis=1)
    )
    final_events_df.loc[missing_telomere_bound, ['telomere_bound', 'whole_arm', 'whole_chrom']] = (
        np.stack(calc_telomere_bound_whole_arm_whole_chrom(final_events_df.loc[missing_telomere_bound]), axis=1))

    final_events_df = final_events_df[EVENT_DF_COLUMNS]
    final_events_df = final_events_df.astype(dtype=EVENT_DF_DTYPES)

    final_events_df = final_events_df.join((final_events_df.groupby('id')['wgd'].first() != 'nowgd').to_frame('has_wgd'), on='id')
    final_events_df['loh_shortened_event'] = final_events_df['diff'].map(lambda x: '0' in x[x.find('1'):x.rfind('1')])
    final_events_df['pos'] = classify_event_position(final_events_df)

    # if knn_train_data is None:
    #     open_pickle(config['input_files']['knn_train'], fail_if_nonexisting=True)
    # unique_events_df = final_events_df.drop_duplicates(['id', 'diff', 'wgd']).copy().reset_index(drop=True)
    # log_debug(logger, 'Overlapping events with SVs')
    # unique_events_df = overlap_final_events_with_svs(unique_events_df, sv_data, sv_matching_threshold)
    # log_debug(logger, 'Calculating final event knn scores')
    # unique_events_df = calculate_final_events_knn_score(unique_events_df, knn_train_data, knn_k)
    # final_events_df = final_events_df.join(unique_events_df.set_index(['id', 'diff', 'wgd'])[['SV_overlap', 'knn_score']],
    #                                        on=['id', 'diff', 'wgd']).reset_index(drop=True)
    # log_debug(logger, f'Joined SV overlaps and knn scores into main df, new length = {len(final_events_df)}')
    # final_events_df = final_events_df.join(
    #     (chrom_segments.groupby('id')['end'].max() - chrom_segments.groupby('id')['start'].min()).to_frame('actual_chrom_length'),
    #     on='id')
    # final_events_df = (final_events_df
    #     .join((final_events_df.set_index('id')['wgd'] == 'pre').groupby('id').sum().to_frame('n_pre'), on='chrom')
    #     .join((final_events_df.set_index('id')['wgd'] == 'post').groupby('id').sum().to_frame('n_post'), on='chrom')
    #     )

    log_debug(logger, f'After postprocessing, found a total of {len(final_events_df)} events in the final df')
    log_debug(logger, 'Calculating summary')
    summary_df = calc_summary_from_events_df(final_events_df, chrom_segments)

    log_debug(logger, f'Saving final files to {output_dir}')
    if output_dir is not None:
        final_events_df.to_csv(os.path.join(output_dir, 'final_events.tsv'), sep='\t', index=False)
        summary_df.to_csv(os.path.join(output_dir, 'events_summary.tsv'), sep='\t', index=True)
        # save_pickle(final_events_df, os.path.join(output_dir, 'final_events.pickle'))
        # save_pickle(summary_df, os.path.join(output_dir, 'summary.pickle'))

    return final_events_df, summary_df


if __name__ == '__main__':
  fire.Fire()
