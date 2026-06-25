from collections import Counter, namedtuple, defaultdict
import os
import random
import heapq
from functools import cache

from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from spice import config
from spice.utils import (
    open_pickle, save_pickle, chrom_id_from_id, CALC_NEW, create_full_df_from_diff_df,
    calc_telomere_bound_whole_arm_whole_chrom)
from spice.logging import log_debug, get_logger
from spice.event_inference.knn_graph import calc_event_distances, EventDistData
from spice.data_loaders import load_chrom_lengths, load_centromeres
from spice.event_inference.events_from_graph import (
    get_starts_and_ends, create_random_start_end_pairs, get_events_diff_from_coords,
    get_events_diff_from_coords_wgd, loh_filters_for_graph_result_diffs, loh_filters_for_graph_result_diffs_wgd, 
    connect_start_ends_using_svs, get_wgd_single_solution, _deepcopy_fast)


logger = get_logger('mcmc_for_large_chroms')

sv_matching_threshold = config['params']['sv_matching_threshold']
Diff = namedtuple('Diff', ['diff', 'type', 'wgd'])

Mcmc_result = namedtuple('Mcmc_result',
                         ['cur_id', 'best_score', 'scores', 'sv_overlaps', 'loh_filter_passed', 'best_events', 'best_diffs', 'best_sv_overlaps', 'is_accepted_iteration', 'scores_full', 'sv_overlaps_full'])
Mcmc_result_full = namedtuple('Mcmc_result',
                         ['cur_id', 'best_score', 'scores', 'sv_overlaps', 'loh_filter_passed', 'best_events', 'best_diffs', 'best_sv_overlaps', 'is_accepted_iteration', 'scores_full', 'sv_overlaps_full', 'sv_selected_events', 'all_events', 'all_diffs'])


@CALC_NEW()
def mcmc_event_selection(
        cur_id,
        chrom_data,
        chrom_segments,
        sv_data=None,
        acceptance_temp=1,
        n_iterations=25,
        knn_graph_k=25,
        sv_matching_threshold=sv_matching_threshold,
        show_progress=False,
        log_progress=False,
        stop_after_no_improvement=None,
        simulated_annealing=False,
        max_T=-6,
        min_T=1,
        log10_distances=True,
        wgd_split=True,
        swap_event_based_on_score=True,
        loh_time_limit=None,
        check_all_loh_solutions=False,
        perform_loh_checks=None, # not yet implemented
        skip_loh_check=True,
        knn_train_data='sv_and_unamb',
        loh_check_before_distance_calculation=False, # needed to compare results with knn
        total_cn=False,
        return_full=False):
    
    from ortools import __version__
    if __version__ != '9.8.3296':
        raise ValueError(f"ortools must have version 9.8.3296, current version is {__version__}")

    has_wgd = chrom_data.has_wgd

    assert isinstance(knn_train_data, dict)

    cur_chrom = chrom_data.chrom
    cur_chrom_length = load_chrom_lengths().loc[cur_chrom]
    centromeres = load_centromeres().astype(int)
    centro_start = centromeres.loc[cur_chrom, 'centro_start']
    centro_end = centromeres.loc[cur_chrom, 'centro_end']
    
    cur_chrom_segments = chrom_segments.query('id == @cur_id')
    if len(cur_chrom_segments) == 0:
        raise ValueError(f'cur id {cur_id} not present in provided chrom segments')
    cur_segment_breakpoints_starts = cur_chrom_segments['start'].values
    cur_segment_breakpoints_ends = cur_chrom_segments['end'].values
    cn_profile = chrom_data.cn_profile
    assert len(cn_profile) == len(cur_chrom_segments), f'cn_profile ({len(cn_profile)}) and cur_chrom_segments ({len(cur_chrom_segments)}) do not have the same lengths'
    has_loh = (cn_profile == 0).any()
    if not has_loh:
        perform_loh_checks = False
    starts, ends = get_starts_and_ends(cn_profile, prior_profile=None, loh_adjust=True, total_cn=total_cn)
    n_events = chrom_data.n_events
    if len(starts) == 0:
        return None
    
    log_debug(logger, f'MCMC for {cur_id}. CN profile: {cn_profile}. {n_events} events. {"WGD" if has_wgd else "no WGD"}.')

    all_scores, all_sv_overlaps, all_loh_filter_passed, all_best_events, all_best_diffs, is_accepted_iteration = [], [], [], [], [], []
    best_score = np.inf
    best_sv_overlaps = 0
    cur_n_sv_overlaps = 0
    overall_best_score = np.inf
    overall_best_diffs = None
    last_best_iteration = 0
    event_distances = None
    overall_best_events = None
    distance_lookup = {}

    # select events based on SV overlap. Used differently for WGD and noWGD
    # For noWGD, events that overlap are pre-selected and the corresponding starts/ends are removed
    # For WGD (since it's impossible to know about pre and post events), no events are pre-selected but proposals
    # are accepted if they increase the number of SV overlaps
    if sv_data is not None:
        chrom_id = chrom_id_from_id(cur_id)
        cur_sv_data = sv_data.query('chrom_id == @chrom_id and (svclass == "DUP" or svclass == "DEL")').copy()
        if has_wgd:
            all_starts, all_ends = np.arange(len(cn_profile)), np.arange(1, len(cn_profile)+1)
            sv_selected_events, _, _ = connect_start_ends_using_svs(
                cur_chrom_segments, cur_sv_data, all_starts, all_ends, sv_matching_threshold=sv_matching_threshold)
            sv_selected_events_set = set([tuple(x) for x in sv_selected_events]) if sv_selected_events is not None else set()
            n_sv_overlaps = 0
            sv_selected_score = 0
        else:
            sv_selected_events, starts, ends = connect_start_ends_using_svs(
                cur_chrom_segments, cur_sv_data, starts, ends, sv_matching_threshold=sv_matching_threshold)
            if sv_selected_events is not None and len(sv_selected_events) > 0:
                n_sv_overlaps = len(sv_selected_events) if sv_selected_events.size > 0 else 0
                best_sv_overlaps = cur_n_sv_overlaps = n_sv_overlaps

                sv_selected_data, sv_selected_score = calc_event_distances_mcmc_wrapper(
                    sv_selected_events, knn_train_data, knn_graph_k, cur_chrom,
                    cur_segment_breakpoints_starts, cur_segment_breakpoints_ends, cn_profile, centro_start,
                    centro_end, cur_chrom_length, chrom_data.has_wgd, log10_distances=log10_distances,
                    wgd_split=wgd_split)
                sv_selected_score = sv_selected_score.sum()
            else:
                sv_selected_score = 0
                n_sv_overlaps = 0
    else:
        n_sv_overlaps = 0
        sv_selected_score = 0
        sv_selected_events = []
        sv_selected_events_set = []
        log_debug(logger, f'{n_sv_overlaps} SV overlaps for a total of {n_events} events. {len(starts)} events remaining')

    # if the chrom is completely solved by SVs
    if n_sv_overlaps > 0 and n_events <= 1:
        n_iterations = 0
        is_accepted_iteration.append(0)
        all_sv_overlaps.append(n_sv_overlaps)
        if n_events == 1:
            sv_selected_events = np.concatenate([sv_selected_events, np.array([[starts[0], ends[0]]])], axis=0)
        overall_best_events = sv_selected_events

    # create initial solution
    if has_wgd:
        event_proposal = get_wgd_single_solution(cn_profile, total_cn=total_cn)
        if event_proposal is None:
            return None
    else:
        starts, ends = np.array(starts), np.array(ends) # for whatever reason, it doesn't work without this
        if len(starts) == 1 and n_iterations > 0:
            raise ValueError('only one event')
        event_proposal = np.array(create_random_start_end_pairs(starts, ends, 1)[0])

        # Sometimes SVs can lead to a faulty initial state from which the MCMC cannot recover -> remove SVs in that case
        if n_sv_overlaps > 0 and 0 in cn_profile:
            log_debug(logger, 'Testing LOH-viability for initial solution')
            diffs = create_diff_and_check_loh(
                event_proposal, cn_profile, has_wgd, total_cn=total_cn, skip_loh_check=skip_loh_check,
                use_cache=True, check_all_solutions=False)
            if len(diffs) == 0:
                log_debug(logger, 'Initial solution using SVs was not compatible with LOHs -> rerunning without SVs')
                sv_selected_score = 0
                sv_selected_events_set = []
                sv_selected_events = None
                best_sv_overlaps = cur_n_sv_overlaps = n_sv_overlaps = 0
                starts, ends = get_starts_and_ends(cn_profile, prior_profile=None, loh_adjust=True, total_cn=total_cn)
                starts, ends = np.array(starts), np.array(ends)
                event_proposal = np.array(create_random_start_end_pairs(starts, ends, 1)[0])

    best_events = event_proposal

    log_debug(logger, f'Starting {n_iterations} iterations for mcmc')
    for iteration in tqdm(range(n_iterations), disable=not show_progress):
        logger.debug(f'iteration {iteration} / {n_iterations} ({int(100*iteration/n_iterations):d}%). Best score: {overall_best_score}')
        if log_progress and iteration % (max(1, n_iterations//10)) == 0:
            log_debug(logger, f'iteration {iteration} / {n_iterations} ({int(100*iteration/n_iterations):d}%). Best score: {overall_best_score}')
        
        force_accept = False
        diffs = None

        if iteration != 0: # needed in case the first guess is already optimal
            assert not has_wgd or len(best_events) == 2, f'best events has wrong length: {len(best_events)}'
            event_proposal = create_mcmc_proposal(
                best_events, has_wgd, iteration, cn_profile, swap_event_based_on_score, event_distances, total_cn=total_cn)
            assert not has_wgd or len(event_proposal) == 2, f'event proposal has wrong length: {len(event_proposal)}'
            if _proposal_is_unchanged(event_proposal, best_events, has_wgd):
                logger.debug(f'{iteration}: proposal rejected because nothing changed')
                all_scores.append(all_scores[-1] if len(all_scores) > 0 else np.inf)
                all_sv_overlaps.append(all_sv_overlaps[-1] if len(all_sv_overlaps) > 0 else 0 if len(all_sv_overlaps) > 0 else 0)
                continue
            
        # This is necessary to get same results as knn for some IDs where a LOH can shorten
        # a subsequent event but takes more time
        if loh_check_before_distance_calculation and 0 in cn_profile:
            diffs = create_diff_and_check_loh(
                event_proposal, cn_profile, has_wgd, total_cn=total_cn, skip_loh_check=skip_loh_check,
                use_cache=True, check_all_solutions=False, single_time_limit=loh_time_limit)
            if len(diffs) == 0:
                logger.debug(f'{iteration}: proposal rejected because early loh filter failed')
                all_scores.append(all_scores[-1] if len(all_scores) > 0 else np.inf)
                all_sv_overlaps.append(all_sv_overlaps[-1] if len(all_sv_overlaps) > 0 else 0)
                continue
            event_proposal = _get_events_from_diff(diffs[0], has_wgd)
            if has_wgd:
                event_proposal = [[tuple(x) for x in event_proposal[0]], [tuple(x) for x in event_proposal[1]]]
            else:
                event_proposal = [tuple(x) for x in event_proposal]

        # calculate knn graph score
        event_proposal_data, event_distances = calc_event_distances_mcmc_wrapper(
            event_proposal, knn_train_data, knn_graph_k, cur_chrom,
            cur_segment_breakpoints_starts, cur_segment_breakpoints_ends, cn_profile, centro_start,
            centro_end, cur_chrom_length, has_wgd, log10_distances=log10_distances,
            wgd_split=wgd_split, distance_lookup=distance_lookup)

        # a bit hacky: penalize events that were gained and then lost (only required for WGD)
        if has_wgd and any([(pre_event[1], pre_event[0]) in event_proposal[1] for pre_event in event_proposal[0]]):
            logger.debug(f'{iteration}: proposal rejected because it contains a doubled gain/loss pair')
            all_scores.append(all_scores[-1] if len(all_scores) > 0 else np.inf)
            all_sv_overlaps.append(all_sv_overlaps[-1] if len(all_sv_overlaps) > 0 else 0)
            continue

        cur_score = sv_selected_score + event_distances.sum()

        # for WGD only: check for SV overlaps (not that for noWGD, the SVs are pre-selected)
        if has_wgd and len(sv_selected_events_set) > 0:
            cur_n_sv_overlaps = (
                len(sv_selected_events_set.intersection(set(event_proposal[0]))) + 
                len(sv_selected_events_set.intersection(set(event_proposal[1]))))
            if cur_n_sv_overlaps > n_sv_overlaps:
                n_sv_overlaps = cur_n_sv_overlaps
                logger.debug(f'{iteration}: proposal accepted because of SV overlap increase ({n_sv_overlaps})')
                force_accept = True

        # accept / reject and process
        if simulated_annealing:
            assert max_T-min_T < 0 and max_T < min_T, f'min_T and max_T must be negative and min_T < max_T. ({min_T} {max_T})'
            T = 10**(min_T + iteration/n_iterations * (max_T-min_T))
        acceptance = (force_accept or
                      (cur_score < best_score) or
                      (simulated_annealing and (np.random.random() < np.exp((best_score - cur_score) / T))) or
                      (not simulated_annealing and (np.random.random() < np.exp((best_score - cur_score) / acceptance_temp))))

        if acceptance:
            logger.debug(f'{iteration}: proposal accepted (old: {best_score}, new: {cur_score})')
            
            if sv_data is not None and sv_selected_events is not None and not has_wgd: # note this can only happen for nowgd
                event_proposal = np.concatenate([sv_selected_events, event_proposal])

            if diffs is None: # only calculate diffs weren't already created in the loop
                diffs = create_diff_and_check_loh(
                    event_proposal, cn_profile, has_wgd, total_cn=total_cn, skip_loh_check=skip_loh_check,
                    use_cache=True, check_all_solutions=False, single_time_limit=loh_time_limit)
            if len(diffs) != 1:
                logger.debug(f'{iteration}: no LOH viable solution found, rejecting proposal')
                all_sv_overlaps.append(None)
                all_scores.append(None)
                all_loh_filter_passed.append(False)
                continue
            all_loh_filter_passed.append(True)

            best_score = cur_score
            best_events = _deepcopy_fast(event_proposal, has_wgd)
            if sv_data is not None and sv_selected_events is not None and not has_wgd:
                best_events = best_events[len(sv_selected_events):]
            is_accepted_iteration.append(iteration)
            all_best_events.append(best_events)
            all_best_diffs.append(diffs[0])
            # new overall best score
            if (force_accept or # force accept = WGD and more SV overlaps
                ((cur_score < overall_best_score) and
                 (cur_n_sv_overlaps >= best_sv_overlaps))
                ):
                overall_best_score = cur_score
                best_sv_overlaps = cur_n_sv_overlaps
                overall_best_events = _deepcopy_fast(event_proposal, has_wgd)
                overall_best_diffs = _deepcopy_fast(diffs[0], has_wgd)
                last_best_iteration = iteration
        else:
            logger.debug(f'{iteration}: proposal REJECTED (old: {best_score}, new: {cur_score})')
        all_scores.append(cur_score)
        all_sv_overlaps.append(n_sv_overlaps)
        
        if stop_after_no_improvement is not None and iteration - last_best_iteration >= stop_after_no_improvement:
            logger.debug(f'{iteration}: no improvement for {stop_after_no_improvement} iterations, stopping')
            break

    if len(is_accepted_iteration) == 0 or overall_best_events is None:
        logger.warning('no accepted iterations -> returning None')
        return None
    log_debug(logger, f'Done with mcmc iterations. Best score: {overall_best_score} at iteration {last_best_iteration} ({len(is_accepted_iteration)} accepted iterations)')

    # this needs to be done in case LOHs shorten subsequent events
    overall_best_diffs = create_diff_and_check_loh(
        overall_best_events, cn_profile, has_wgd, total_cn=total_cn, skip_loh_check=skip_loh_check,
        use_cache=True, check_all_solutions=(has_loh and check_all_loh_solutions),
        single_time_limit=loh_time_limit)
    if len(overall_best_diffs) != 1:
        logger.warning('no LOH viable solution found for best events -> returning None')
        return None
    overall_best_diffs = overall_best_diffs[0]

    assert best_sv_overlaps == (max(x for x in all_sv_overlaps if x is not None) if any(x is not None for x in all_sv_overlaps) else 0), (best_sv_overlaps, np.max(all_sv_overlaps))

    overall_best_events = _get_events_from_diff(overall_best_diffs, has_wgd)
    if ((has_wgd and any(
            [event[0]==event[1] for event in overall_best_events[0]] + 
            [event[0]==event[1] for event in overall_best_events[1]]
            )) or
        (not has_wgd and any([event[0]==event[1] for event in overall_best_events]))):
        raise ValueError('Invalid empty event (start == end) found. This is due to incorrect distance calculation')
    
    if not skip_loh_check:
        log_debug(logger, f'Finished final LOH check')
    best_events_data, overall_best_score = calc_event_distances_mcmc_wrapper(
        overall_best_events, knn_train_data, knn_graph_k, cur_chrom,
        cur_segment_breakpoints_starts, cur_segment_breakpoints_ends, cn_profile, centro_start,
        centro_end, cur_chrom_length, chrom_data.has_wgd, log10_distances=log10_distances,
        wgd_split=wgd_split)  

    overall_best_score = overall_best_score.sum()
    if len(all_scores) == 0: # in case of SV-only solution
        all_scores.append(overall_best_score)
    scores = np.array(all_scores)[is_accepted_iteration]

    mcmc_result = Mcmc_result(
        cur_id=cur_id,
        best_score=overall_best_score,
        scores=scores,
        sv_overlaps=np.array(all_sv_overlaps)[is_accepted_iteration],
        loh_filter_passed=all_loh_filter_passed,
        best_events=overall_best_events,
        best_diffs=overall_best_diffs,
        best_sv_overlaps=best_sv_overlaps,
        is_accepted_iteration=np.array(is_accepted_iteration),
        scores_full=np.array(all_scores),
        sv_overlaps_full=np.array(all_sv_overlaps),
        )
    
    if return_full:
        mcmc_result = Mcmc_result_full(
            *mcmc_result, sv_selected_events, all_events=all_best_events if has_wgd else np.array(all_best_events),
            all_diffs=all_best_diffs)

    return mcmc_result


def get_event_dist_data_from_mcmc_proposal(
        event_proposal, cur_chrom, cur_segment_breakpoints_starts,
        cur_segment_breakpoints_ends, cn_profile, centro_start, centro_end, cur_chrom_length, has_wgd=False):

    if has_wgd:
        assert len(event_proposal) == 2
        wgd_status = np.array(['pre']*len(event_proposal[0]) + ['post']*len(event_proposal[1]))
        event_proposal = np.concatenate([
            np.array(pre_post) for pre_post in event_proposal if len(pre_post) > 0])
    else:
        wgd_status = np.array(['nowgd']*len(event_proposal))
        event_proposal = np.array(event_proposal)

    # event_proposal_sorted swaps start and end for losses so start is always smaller than end
    event_proposal_sorted = np.sort(event_proposal, axis=1)

    starts = cur_segment_breakpoints_starts[event_proposal_sorted[:, 0]]
    ends = cur_segment_breakpoints_ends[event_proposal_sorted[:, 1]-1]
    widths = ends - starts
    assert np.all(widths > 0), (event_proposal, widths)
    type = np.where(event_proposal[:, 0] < event_proposal[:, 1], 'gain', 'loss')

    (is_telomere_bound,
        is_whole_arm,
        is_whole_chrom) = calc_telomere_bound_whole_arm_whole_chrom(
            (event_proposal_sorted, cn_profile, starts, ends, centro_start, centro_end))

    event_proposal_data = EventDistData(
        chrom=np.array([cur_chrom]*len(event_proposal)),
        starts=starts,
        ends=ends,
        widths=widths,
        type=type,
        is_telomere_bound=is_telomere_bound,
        is_whole_chrom=is_whole_chrom,
        is_whole_arm=is_whole_arm,
        wgd=wgd_status,
        chrom_lengths=cur_chrom_length
        )

    return event_proposal_data


def create_mcmc_proposal(best_events, has_wgd, iteration, cn_profile=None, swap_event_based_on_score=True,
                         event_distances=None, total_cn=False):
    if has_wgd:
        return _create_mcmc_proposal_wgd(best_events, iteration, cn_profile, swap_event_based_on_score,
                                         event_distances, total_cn=total_cn)
    else:
        return _create_mcmc_proposal_nowgd(best_events, iteration, swap_event_based_on_score, event_distances)


def _create_mcmc_proposal_wgd(cur_events, iteration, cn_profile, swap_event_based_on_score=True, event_distances=None,
                              loh_lookup=None, total_cn=False):
    assert cn_profile is not None
    # only necessary for simple_swap
    if swap_event_based_on_score and iteration % 4 == 3:
        swap_event_based_on_score = False

    if loh_lookup is None:
        loh_lookup = {}
    
    new_events = None

    n = 0
    while new_events is None:
        random_var = random.choice(range(7))
        if random_var == 0:
            new_events = proposal_wgd_add_bp(cur_events, cn_profile, total_cn=total_cn)
            cur_transition = 'add_bp'
        elif random_var == 1:
            new_events = proposal_wgd_remove_bp(cur_events, cn_profile, total_cn=total_cn)
            cur_transition = 'remove_bp'
        elif random_var == 2:
            new_events, option = proposal_wgd_extend_shorten_pre_gain(cur_events, cn_profile, return_selected_option=True,
                                                                      total_cn=total_cn)
            cur_transition = 'extend_shorten_pre_gain-' + option
        elif random_var == 3:
            new_events, option = proposal_wgd_extend_shorten_pre_loss(cur_events, cn_profile, return_selected_option=True,
                                                                      total_cn=total_cn)
            cur_transition = 'extend_shorten_pre_loss-' + option
        elif random_var == 4:
            new_events, option = proposal_wgd_switch_loh_loss(cur_events, cn_profile, return_pre_post=True)
            cur_transition = 'switch_loh_loss-' + option
        elif random_var >= 5: # this has double the probability because it's the fastest
            new_events = proposal_wgd_simple_swap(cur_events, swap_event_based_on_score, event_distances)
            cur_transition = 'simple_swap'

        n += 1
        if n > 1_000:
            # this can only happen on the very first proposal as the subsequent proposals are based on the previous ones
            log_debug(logger, f'No valid proposal found after 1000 tries for current events: {cur_events}')
            new_events = get_wgd_single_solution(cn_profile, total_cn=total_cn)
            break

    logger.debug(f'Transition: {cur_transition}')
    return new_events


def _create_mcmc_proposal_nowgd(best_events, iteration, swap_event_based_on_score=True, event_distances=None):
    best_events = np.array(best_events)
    event_proposal = best_events.copy()
    
    # event_distances can be None if all of the proposals failed LOH filter
    if swap_event_based_on_score and iteration % 4 != 3 and event_distances is not None: # every 4th turn, choose completely random
        p = 10**event_distances
        p /= p.sum()
    else:
        p = np.ones(len(best_events)) / len(best_events)
    
    ind1, ind2 = np.random.choice(range(len(best_events)), 2, replace=False, p=p)
    event_proposal[(ind1, ind2), 1] = best_events[(ind2, ind1), 1]

    return event_proposal


def calc_event_distances_mcmc_wrapper(
    event_tuples, knn_train_data, knn_graph_k, cur_chrom, cur_segment_breakpoints_starts,
    cur_segment_breakpoints_ends, cn_profile, centro_start, centro_end, cur_chrom_length, has_wgd=False,
    log10_distances=True, wgd_split=True, distance_lookup=None):

    # using @cache is too much of a hassle because the arguments are not hashable
    event_tuples_hash = (tuple([tuple(sorted(pre_post)) if len(pre_post) > 0 else () for pre_post in event_tuples])
                         if distance_lookup is not None else None)
    if event_tuples_hash is not None and event_tuples_hash in distance_lookup:
        return distance_lookup[event_tuples_hash]

    event_dist_data = get_event_dist_data_from_mcmc_proposal(
        event_tuples, cur_chrom, cur_segment_breakpoints_starts, cur_segment_breakpoints_ends,
        cn_profile, centro_start, centro_end, cur_chrom_length, has_wgd)

    event_distances = calc_event_distances(
        knn_train_data, event_dist_data, ks=knn_graph_k, ignore_empty_train=False, block_same_id=False,
        clip_k=False, log10_distances=log10_distances, wgd_split=wgd_split, assert_finite=True, single_width_bin=True)

    if event_tuples_hash is not None:
        distance_lookup[event_tuples_hash] = (event_dist_data, event_distances)

    return event_dist_data, event_distances


def create_diff_and_check_loh(event_proposal, cn_profile, has_wgd=False, total_cn=False,
                              use_cache=True, check_all_solutions=False, single_time_limit=1,
                              skip_loh_check=True):
    if not use_cache:
        raise DeprecationWarning('use_cache must be True')
    
    if has_wgd:
        event_proposal_ = (sorted((tuple(x) for x in event_proposal[0])), sorted([tuple(x) for x in event_proposal[1]]))
        event_proposal_ = tuple(tuple(pre_post) for pre_post in event_proposal_)
    else:
        event_proposal_ = tuple(sorted([tuple(x) for x in event_proposal]))
    cn_profile_ = tuple(cn_profile)

    diffs_filtered = _create_diff_and_check_loh_impl(
        event_proposal_, cn_profile_, has_wgd, check_all_solutions, single_time_limit, total_cn=total_cn,
        skip_loh_check=skip_loh_check)

    return diffs_filtered


@cache
def _create_diff_and_check_loh_impl(event_proposal, cn_profile, has_wgd=False, check_all_solutions=False,
                                    single_time_limit=1, total_cn=False, skip_loh_check=True):
    cn_profile = np.array(cn_profile)
    if has_wgd:
        diffs = get_events_diff_from_coords_wgd(
            [event_proposal], cn_profile, lexsort_diffs=True, filter_missed_lohs=False, fail_if_empty=False)
    else:
        diffs = get_events_diff_from_coords(
            [event_proposal], cn_profile, lexsort_diffs=True, filter_missed_lohs=not skip_loh_check,
            fail_if_empty=False)
    if len(diffs) == 0:
        return []
    assert diffs is not None, 'no diffs found'

    # can happen if there aren't any loss events at LOH positions
    if len(diffs) != 1:
        return []

    if 0 in cn_profile and not skip_loh_check:
        logger.debug('LOH detected, checking for LOH-viability')
        if has_wgd:
            # quick check to save time
            if np.logical_and(
                (diffs[0][0][:, cn_profile == 0] != -1).all(axis=0),
                (diffs[0][1][:, cn_profile == 0] != -1).all(axis=0)
            ).any():
                return []

    #         cur_diffs_hash = (hash(cur_diffs[0][0].tobytes()), hash(cur_diffs[0][1].tobytes()))
    #         loh_passed = loh_lookup.get(cur_diffs_hash)
            diffs_final = loh_filters_for_graph_result_diffs_wgd(
                diffs, cn_profile, return_all_solutions=check_all_solutions,
                shuffle_diffs=True, total_cn=total_cn)
        else:
            diffs_final = loh_filters_for_graph_result_diffs(
                diffs, cn_profile, return_all_solutions=check_all_solutions,
                shuffle_diffs=True, total_cn=total_cn)

        # if mulitple diffs are created, choose the one with the fewest number of affected segments
        if check_all_solutions and len(diffs_final) >= 1:
            min_events_index = np.argmin([np.sum(np.abs(x)) for x in diffs_final])
            diffs_final = [diffs_final[min_events_index]]
    else:
        diffs_final = diffs

    return diffs_final


@CALC_NEW()
def run_multiple_mcmc(n_duplicates, **kwargs):
    all_duplications_scores = []
    best_duplicate_score = np.inf
    best_duplication_sv_overlaps = 0
    best_results = None
    for _duplicate in range(n_duplicates):
        all_scores, all_sv_overlaps, all_loh_filter_passed, all_best_events, is_accepted_iteration = mcmc_event_selection(**kwargs)
        duplication_score = np.min(np.array(all_scores)[is_accepted_iteration])
        duplication_sv_overlaps = np.max(np.array(all_sv_overlaps)[is_accepted_iteration])
        all_duplications_scores.append(duplication_score)
        if (duplication_sv_overlaps >= best_duplication_sv_overlaps and duplication_score < best_duplicate_score):
            best_duplicate_score = duplication_score
            best_duplication_sv_overlaps = duplication_sv_overlaps
            best_results = (all_scores, all_sv_overlaps, all_loh_filter_passed, all_best_events, is_accepted_iteration)
        
    return best_results, all_duplications_scores


def plot_mcmc_scores(mcmc_result=None,
                     all_scores=None, is_accepted_iteration=None, all_sv_overlaps=None, scores_full=None, sv_overlaps_full=None,
                     ax=None, show_best_score=True, show_all_best_scores=True, title=None, show_svs=True):
        
    if mcmc_result is not None:
        all_scores = mcmc_result.scores
        is_accepted_iteration = mcmc_result.is_accepted_iteration
        all_sv_overlaps = mcmc_result.sv_overlaps
        scores_full = mcmc_result.scores_full
        sv_overlaps_full = mcmc_result.sv_overlaps_full
    else:
        if all_scores is None or is_accepted_iteration is None:
            raise ValueError('either mcmc_result or all_scores and is_accepted_iteration must be provided')
    if ax is None:
        fig, ax = plt.subplots(figsize=(15, 5))
    else:
        fig = ax.figure
    plt.plot(is_accepted_iteration, all_scores, 'o-')
    # plt.yscale('log')

    if show_best_score:
        plt.axvline(np.array(is_accepted_iteration)[np.array(all_scores) == np.min(all_scores)][0], color='red', label='best score', lw=2)
    if show_all_best_scores:
        for x in np.array(is_accepted_iteration)[np.array(all_scores) == np.min(all_scores)]:
            plt.axvline(x, color='grey', ls='--', alpha=0.5)

    if scores_full is not None:
        ax.plot(np.arange(len(scores_full))[np.array([x is not None for x in scores_full])],
                np.array(scores_full)[np.array([x is not None for x in scores_full])], '--', color='C0', alpha=0.25)

    if all_sv_overlaps is not None and show_svs:
        ax2 = ax.twinx()
        if len(all_sv_overlaps) > 0:
            ax2.plot(is_accepted_iteration, all_sv_overlaps, 'x-', color='orange');
        if sv_overlaps_full is not None:
            ax2.plot(np.arange(len(sv_overlaps_full))[np.array([x is not None for x in sv_overlaps_full])],
                    np.array(sv_overlaps_full)[np.array([x is not None for x in sv_overlaps_full])], '--', color='C1', alpha=0.25)

    if title is not None:
        plt.title(title, fontsize=20, va='bottom')
    
    return fig, ax


def create_best_events_df_from_mcmc(mcmc_result, chrom_segments, chrom_data=None, has_wgd=False):

    cur_id = mcmc_result.cur_id
    best_diffs = mcmc_result.best_diffs
    if has_wgd:
        wgd_status = np.array(['pre']*len(best_diffs[0]) + ['post']*len(best_diffs[1]))
        best_diffs = np.concatenate([pre_post for pre_post in best_diffs if len(pre_post) > 0])
    else:
        wgd_status = np.array(['nowgd']*len(best_diffs))

    cur_chrom_segments = chrom_segments.query('id == @cur_id')

    best_events_df = pd.DataFrame(index=np.arange(len(best_diffs)))
    best_events_df['diff'] = [''.join(np.abs(diff).astype(str)) for diff in best_diffs]
    best_events_df['type'] = ['gain' if x.max()==1 else 'loss' for x in best_diffs]
    best_events_df = create_full_df_from_diff_df(best_events_df, cur_id, cur_chrom_segments)
    best_events_df['wgd'] = wgd_status
    best_events_df['n_paths'] = -1

    if chrom_data is not None:
        assert chrom_data.has_wgd == has_wgd
        if (chrom_data.n_events - mcmc_result.sv_overlaps[-1]) <= 1:
            # for noWGD, the SVs should be pre-selected
            assert has_wgd or len(mcmc_result.sv_overlaps) == 1, f"sv_overlaps: {mcmc_result.sv_overlaps}"
            best_events_df['solved'] = 'sv'
        else:
            best_events_df['solved'] = 'mcmc'
        best_events_df['events_per_chrom'] = chrom_data.n_events
    else:
        best_events_df['events_per_chrom'] = -1
        best_events_df['solved'] = 'mcmc'

    assert len(best_events_df) == len(best_diffs)

    return best_events_df


def _create_proposal_from_cur_events(cur_events, remove_pre=None, add_pre=None,
                                     remove_post=None, add_post=None):
    '''Shortcut function that takes a wgd event (i.e. tuple with pre- and post-WGD events) and 
    creates a new proposal with the provided changes'''
    # event_proposal_pre = deepcopy(cur_events[0])
    # event_proposal_post = deepcopy(cur_events[1])
    event_proposal_pre, event_proposal_post = _deepcopy_fast(cur_events, has_wgd=True)
    if remove_pre is not None:
        for event in remove_pre:
             event_proposal_pre = [event for event in event_proposal_pre if event not in remove_pre or remove_pre.remove(event)]
    if remove_post is not None:
        for event in remove_post:
            event_proposal_post = [event for event in event_proposal_post if event not in remove_post or remove_post.remove(event)]
    if add_pre is not None:
        event_proposal_pre.extend(add_pre)
    if add_post is not None:
        event_proposal_post.extend(add_post)
    return [event_proposal_pre, event_proposal_post]


def _removed_pre_gain_passes_loh_filter(cur_events, cur_pre_gain, cn_profile, which, new_pre_gain=None, total_cn=False):
    '''If a gain is shortened this could potentially create a LOH since a pre-loss might now not have a counterpart'''
    
    pre_events = cur_events[0].copy()

    pre_events.remove(cur_pre_gain)
    if which == 'remove_bp':
        start, end = cur_pre_gain[0], (cur_pre_gain[1]+1)
    elif which == 'shorten_pre_gain':
        assert new_pre_gain is not None, 'new_pre_gain must be provided'
        pre_events.append(new_pre_gain)
        if cur_pre_gain[0] == new_pre_gain[0]:
            start, end = new_pre_gain[1], cur_pre_gain[1]
        elif cur_pre_gain[1] == new_pre_gain[1]:
            start, end = cur_pre_gain[0], new_pre_gain[0]
        else:
            raise ValueError('Invalid new_pre_gain')
    else:
        raise ValueError('Invalid parameter "which" for _removed_pre_gain_passes_loh_filter')

    cur_diff_pre = get_events_diff_from_coords(
        [pre_events], cn_profile, lexsort_diffs=True, filter_missed_lohs=False)[0]
    cur_diff_pre = cur_diff_pre[:, start:end]
    if np.any(cur_diff_pre == -1):
        cur_diff_pre_loh_filtered = loh_filters_for_graph_result_diffs(
            [cur_diff_pre], cn_profile[start:end], total_cn=total_cn,
            single_time_limit=None, return_all_solutions=False, shuffle_diffs=False)
        if len(cur_diff_pre_loh_filtered) == 0:
            return False
    
    return True


def _added_pre_loss_passes_loh_filter(cur_events, cur_loss, cn_profile, which, new_pre_loss=None, total_cn=False):
    '''Since pre loss was added/extended there needs to be a counterpart pre gain to prevent unwanted LOHs'''
    
    pre_events = cur_events[0].copy()

    if which == 'add_bp':
        start, end = cur_loss[1], (cur_loss[0])
        pre_events.append(cur_loss)
    elif which == 'extend_pre_loss':
        pre_events.remove(cur_loss)
        assert new_pre_loss is not None, 'new_pre_loss must be provided'
        pre_events.append(new_pre_loss)
        if cur_loss[0] == new_pre_loss[0]:
            start, end = new_pre_loss[1], cur_loss[1]
        elif cur_loss[1] == new_pre_loss[1]:
            start, end = cur_loss[0], new_pre_loss[0]
        else:
            raise ValueError('Invalid new_pre_loss')
    else:
        raise ValueError('Invalid parameter "which" for _added_pre_loss_passes_loh_filter')

    cur_diff_pre = get_events_diff_from_coords(
        [pre_events], cn_profile, lexsort_diffs=True, filter_missed_lohs=False)[0]
    cur_diff_pre = cur_diff_pre[:, start:end]

    # if there is no gain to counter-balance the loss, it would create a faulty LOH, no need to run the loh filters
    if np.logical_and((cur_diff_pre != 1).all(axis=0), cn_profile[start:end] != 0).any():
        return False

    cur_diff_pre_loh_filtered = loh_filters_for_graph_result_diffs(
        [cur_diff_pre], cn_profile[start:end], total_cn=total_cn,
        single_time_limit=None, return_all_solutions=False, shuffle_diffs=False)
    if len(cur_diff_pre_loh_filtered) == 0:
        return False
    
    return True


def _select_events_with_same_start_or_end(all_events, start_end):
    '''Finds all pairs of events that share a start or end'''
    assert start_end in ['start', 'end']
    start_end_i = 0 if start_end == 'start' else 1
    # Group tuples by their second element
    grouped_by_second = defaultdict(list)
    for tup in all_events:
        grouped_by_second[tup[start_end_i]].append(tup)
    # Filter groups to keep only those with more than one tuple
    filtered_groups = {k: v for k, v in grouped_by_second.items() if len(v) > 1}
    return filtered_groups


def proposal_wgd_simple_swap(cur_events, swap_event_based_on_score=False, event_distances=None):
    '''Performs a simple swap on the pre- or post-WGD events. A simple swap takes two random events
    (both from either pre- or post-WGD) and exchanges their ends

    Examples:
        2464: (1 3) (2 4) // (2 3) -> (1 4) (2 3) // (2 3)
        1232: // (1 3) (2 4) ->  // (1 4) (2 3)
    '''
    # event_distances can be None if all of the proposals failed LOH filter
    if swap_event_based_on_score and event_distances is not None:
        event_distances_pre = event_distances[:len(cur_events[0])]
        event_distances_post = event_distances[len(cur_events[0]):]
        p = [
            [10**x for x in event_distances_pre],
            [10**x for x in event_distances_post]]
        p = [x / np.sum(x) for x in p]
    else:
        p = [np.ones(len(cur_events[0])) / len(cur_events[0]), np.ones(len(cur_events[1])) / len(cur_events[1])]

    pre_post = [label for label, events in zip(['pre', 'post'], cur_events) if len(events) >= 2]
    if len(pre_post) == 0:
        return None
    pre_post = random.choice(pre_post)
    pre_post_ind = 0 if pre_post == 'pre' else 1
    new_events = _deepcopy_fast(cur_events, has_wgd=True)

    # ind1, ind2 = random.sample(range(len(new_events[pre_post_ind])), 2)
    ind1, ind2 = np.random.choice(range(len(new_events[pre_post_ind])), size=2, p=p[pre_post_ind])
    cur = (new_events[pre_post_ind][ind1][0], new_events[pre_post_ind][ind2][1])
    new_events[pre_post_ind][ind2] = (new_events[pre_post_ind][ind2][0], new_events[pre_post_ind][ind1][1])
    new_events[pre_post_ind][ind1] = cur

    if any([event[0]==event[1] for event in new_events[0]] + [event[0]==event[1] for event in new_events[1]]):
        return None
    
    # no change
    if (sorted((cur_events[pre_post_ind][ind1], cur_events[pre_post_ind][ind2])) == 
        sorted((new_events[pre_post_ind][ind1], new_events[pre_post_ind][ind2]))):
        return None

    return new_events


def proposal_wgd_add_bp(cur_events, cn_profile, total_cn=False):
    '''
    Turns a single post-WGD event into a pre-WGD event and changes a 2nd post-WGD event
    that shares either the start or end with it.
    This introduces a breakpoint that is now present in both starts and ends (hence the name).

    Examples:
        43: // (0 1) (0 2) -> (0 2) // (2 1) (2 gains with same start, bp 2 was added)
        43: // (0 1) (0 2) -> (0 1) // (1 2) (2 gains with same start, bp 1 was added)
        13: // (1 0) (1 2) -> (1 2) // (2 0) (gain and loss with same start, bp 1 was added)
        13: // (1 0) (1 2) -> (1 0) // (0 2) XXX does not work because of LOH
        4324: (0 4) // (3 1) (3 2) -> (0 4) (3 2) // (2 1) (2 losses with same start, bp 2 was added, note that (0 4) pre-gain balances new pre-loss)
        4324: (0 4) // (3 1) (3 2) -> (0 4) (3 1) // (2 3) (2 losses with same start, bp 3 was added, note that (0 4) pre-gain balances new pre-loss)
    
    Also works for the mirrored case:
        34: // (0 2) (1 2) -> (0 2) // (1 0) (2 gains with same start, bp 2 was added)
        etc...
    '''

    doubled_starts = [(e, 'start') for e, c in Counter([event[0] for event in cur_events[1]]).items() if c>=2]
    doubled_ends = [(e, 'end') for e, c in Counter([event[1] for event in cur_events[1]]).items() if c>=2]
    if len(doubled_starts) + len(doubled_ends) == 0:
        return None

    # select a doubled bp
    cur_bp, start_end = random.choice(doubled_starts + doubled_ends)
    start_end_i = 0 if start_end == 'start' else 1

    # select two events with this bp
    cur_events_with_bp = [event for event in cur_events[1] if event[start_end_i] == cur_bp]
    if len(cur_events_with_bp) < 2:
        return None
    cur_event_to_double, other_event_with_bp = random.sample(cur_events_with_bp, 2)

    if cur_event_to_double[0] > cur_event_to_double[1]:
        if not _added_pre_loss_passes_loh_filter(cur_events, cur_event_to_double, cn_profile, which='add_bp', total_cn=total_cn):
            return None

    event_proposal = _create_proposal_from_cur_events(
        cur_events, add_pre=[cur_event_to_double], remove_post=[cur_event_to_double, other_event_with_bp],
        add_post=[
            (cur_event_to_double[1], other_event_with_bp[1]) if start_end == 'start' else (other_event_with_bp[0], cur_event_to_double[0])]
    )

    return event_proposal


def proposal_wgd_remove_bp(cur_events, cn_profile, total_cn=False):
    '''Reverse of proposal_add_bp

    Turns a single pre-WGD event into a post-WGD event and changes a 2nd post-WGD event that shares an added bp
    (ie the pre starts there and the post ends there or vice verse).
    This removes a breakpoint that is present in both starts and ends (hence the name).

    Examples (reverse of proposal_add_bp):
        43: (0 2) // (2 1) -> // (0 1) (0 2) (gain and loss both at bp 2, which was removed)
        43: (0 1) // (1 2) -> // (0 1) (0 2) (2 gains both at bp 1, which was removed)
        13: (1 2) // (2 0) -> // (1 0) (1 2) (gain and loss both at bp 1, which was removed)
        4324: (0 4) (3 2) // (2 1) -> (0 4) // (3 1) (3 2) (gain and loss both at bp 2, which was removeds)
        4324: (0 4) (3 1) // (2 3) -> (0 4) // (3 1) (3 2) (gain and loss both at bp 3, which was removed)
    
    Also works for the mirrored case:
        34: (0 2) // (1 0) -> // (0 2) (1 2) (gain and loss both at bp 2, which was removed)
        etc...
    '''
    # find doubled bps
    cur_starts = set([e[0] for e in cur_events[0]]) | set([e[0] for e in cur_events[1]])
    cur_ends = set([e[1] for e in cur_events[0]]) | set([e[1] for e in cur_events[1]])
    doubled_bps = cur_starts & cur_ends
    if len(doubled_bps) == 0:
        return None

    # select one pre gain/loss that starts/ends at a doubled bp
    cur_pre_gains_start = [(event, 'gain', 'start') for event in cur_events[0] if event[0] in doubled_bps and event[0] < event[1]]
    cur_pre_gains_end = [(event, 'gain', 'end') for event in cur_events[0] if event[1] in doubled_bps and event[0] < event[1]]
    cur_pre_losses_start = [(event, 'loss', 'start') for event in cur_events[0] if event[0] in doubled_bps and event[0] > event[1]]
    cur_pre_losses_end = [(event, 'loss', 'end') for event in cur_events[0] if event[1] in doubled_bps and event[0] > event[1]]
    if len(cur_pre_gains_start) + len(cur_pre_gains_end) + len(cur_pre_losses_start) + len(cur_pre_losses_end) == 0:
        return None

    cur_pre_event, gain_loss, start_end = random.choice(
        cur_pre_gains_start + cur_pre_gains_end + cur_pre_losses_start + cur_pre_losses_end)
    start_end_i = 0 if start_end == 'start' else 1
    start_end_j = 1 if start_end == 'start' else 0
    cur_doubled_bp = cur_pre_event[start_end_i]

    if gain_loss=="gain" and not _removed_pre_gain_passes_loh_filter(cur_events, cur_pre_event, cn_profile, which='remove_bp', total_cn=total_cn):
        return None

    # find a post event that ends/starts at doubled bp
    cur_post_events = [event for event in cur_events[1] if event[start_end_j] == cur_doubled_bp]
    if len(cur_post_events) == 0:
        return None
    cur_post_event = random.choice(cur_post_events)

    event_proposal = _create_proposal_from_cur_events(
        cur_events, remove_pre=[cur_pre_event], remove_post=[cur_post_event],
        add_post=[cur_pre_event,
                  (cur_pre_event[0], cur_post_event[1]) if start_end == 'end' else (cur_post_event[0], cur_pre_event[1])]
        )
    return event_proposal


def proposal_wgd_extend_shorten_pre_gain(cur_events, cn_profile, return_selected_option=False, total_cn=False):
    '''Extend or shorten a pre-gain event and at the same time change one or two post-WGD events.
    There are 6 options in total, named A-F. A, C and E extend the gain while B, D and F shorten it.
    The idea is always that the pre-gain is either extended or shortened and the change in CN is balanced
    by changing one or two post-WGD events. An extension of a pre-gain requires a balancing copy-number of -2
    which can be achieved by either moving in 2 losses, moving in 1 loss and 1 gain or flipping one gain to a
    loss (moving out 2 gains technically works but in practice there are no cases where this is possible).

    A / B, C / D and E / F are reverse operations.
    Doubled bp = bp that is present in both starts and ends

    The six options are:
    A:
        Extend pre-gain and flip a gain (turn to loss) that shares a doubled bp.
        The doubled bp will change in this process.
        Examples:
            43: (0 1) // (1 2) -> (0 2) // (2 1) (bp 1 was doubled, afterwards bp 2 is doubled)

    B:
        Shorten pre-gain and flip a loss (turn to gain) that shares a doubled bp.
        The doubled bp will change in this process.

        Examples:
            43: (0 2) // (2 1) -> (0 1) // (1 2) (bp 2 was doubled, afterwards bp 1 is doubled)

    C:
        Extend pre-gain and extend two losses inwards from either side
        Examples:
            1321: (1 2) // (2 0) (4 3) -> (1 3) // (3 0) (4 2)

    D:
        Shorten pre-gain and shorten two losses outwards to either side
        Examples:
            1321: (1 3) // (3 0) (4 2) -> (1 2) // (2 0) (4 3)

    E:
        Extend pre-gain and moves two events with the same start/end inwards (to the left/right)
        from right/left side. The resulting two post-events will still share (a now altered) start/end.
        Note that gains can turn to losses.
        Examples:
            4234: (0 1) // (2 4) (3 4) -> (0 4) // (2 1) (3 1) (here two gains turns into losses,
                they share the same end before (4) and after (1) the operation)
            4201: (0 1) // (3 2) (4 2) -> (0 2) // (3 1) (4 1) (here the two losses stay losses
                they share the same end before (2) and after (1) the operation)

    F:
        Shortens a pre-gain and moves two events with the same start/end outwards (to the right/left)
        from the left/right side. The resulting two post-events will still share (a now altered) start/end.
        Note that losses can turn to gains.
        Examples:
            4234: (0 4) // (2 1) (3 1) -> (0 1) // (2 4) (3 4) (here two losses turns into gains,
                they share the same end before (1) and after (4) the operation)
            4201: (0 2) // (3 1) (4 1) -> (0 1) // (3 2) (4 2) (here the two losses stay losses
                they share the same end before (1) and after (2) the operation)       

    All of the examples also work for the mirrored cases.
    Here is the option A example mirrored:
        34: (1 2) // (0 1) -> (0 2) // (1 0)
    '''
    # select a random pre-gain
    all_pre_gains = [event for event in cur_events[0] if event[0] < event[1]]
    if len(all_pre_gains) == 0:
        if return_selected_option:
            return None, '0'
        return None
    cur_pre_gain = random.choice(all_pre_gains)

    all_post_gains = [event for event in cur_events[1] if event[0] < event[1]]
    all_post_losses = [event for event in cur_events[1] if event[0] > event[1]]

    # for A/B
    all_post_gains_start_at_pre_end = [event for event in all_post_gains if event[0] == cur_pre_gain[1]]
    all_post_gains_end_at_pre_start = [event for event in all_post_gains if event[1] == cur_pre_gain[0]]
    all_post_losses_start_at_pre_end_and_inside_pre = [event for event in all_post_losses if event[0] == cur_pre_gain[1] and event[1] > cur_pre_gain[0]]
    all_post_losses_end_at_pre_start_and_inside_pre = [event for event in all_post_losses if event[1] == cur_pre_gain[0] and event[0] < cur_pre_gain[1]]
    # for C/D
    all_post_losses_start_at_pre_end = [event for event in all_post_losses if event[0] == cur_pre_gain[1]]
    all_post_losses_end_at_pre_start = [event for event in all_post_losses if event[1] == cur_pre_gain[0]]
    all_post_losses_right = [event for event in all_post_losses if event[1] > cur_pre_gain[1]]
    all_post_losses_left = [event for event in all_post_losses if event[0] < cur_pre_gain[0]]
    all_post_losses_right_partly_inside_pre = [event for event in all_post_losses if event[1] < cur_pre_gain[1] and event[1] > cur_pre_gain[0] and event[0] > cur_pre_gain[1]]
    all_post_losses_left_partly_inside_pre = [event for event in all_post_losses if event[0] > cur_pre_gain[0] and event[0] < cur_pre_gain[1] and event[1] < cur_pre_gain[0]]
    # for E/F
    all_post_events_start_left = [event for event in cur_events[1] if event[0] < cur_pre_gain[0]]
    all_post_events_end_right = [event for event in cur_events[1] if event[1] > cur_pre_gain[1]]
    all_post_events_start_right_and_inside = [event for event in cur_events[1] if event[0] > cur_pre_gain[0] and event[0] < cur_pre_gain[1]]
    all_post_events_end_left_and_inside = [event for event in cur_events[1] if event[1] < cur_pre_gain[1] and event[1] > cur_pre_gain[0]]
    all_post_events_end_right_same_end = _select_events_with_same_start_or_end(all_post_events_end_right, start_end='end')
    all_post_events_start_left_same_start = _select_events_with_same_start_or_end(all_post_events_start_left, start_end='start')
    all_post_events_start_right_and_inside_same_start = _select_events_with_same_start_or_end(all_post_events_start_right_and_inside, start_end='start')
    all_post_events_end_left_and_inside_same_end = _select_events_with_same_start_or_end(all_post_events_end_left_and_inside, start_end='end')

    options = [
        ('A', len(all_post_gains_start_at_pre_end) > 0 or len(all_post_gains_end_at_pre_start) > 0),
        ('B', len(all_post_losses_start_at_pre_end_and_inside_pre) > 0 or len(all_post_losses_end_at_pre_start_and_inside_pre) > 0),
        ('C', ((len(all_post_losses_start_at_pre_end) > 0 and len(all_post_losses_right) > 0) or 
               (len(all_post_losses_end_at_pre_start) > 0 and len(all_post_losses_left) > 0))),
        ('D', (len(all_post_losses_start_at_pre_end) > 0 and len(all_post_losses_right_partly_inside_pre) > 0) or
              (len(all_post_losses_end_at_pre_start) > 0 and len(all_post_losses_left_partly_inside_pre) > 0)),
        ('E', len(all_post_events_end_right_same_end) > 0 or len(all_post_events_start_left_same_start) > 0),
        ('F', len(all_post_events_start_right_and_inside_same_start) > 0 or len(all_post_events_end_left_and_inside_same_end) > 0),

    ]
    available_options = [option for option, condition in options if condition]
    if len(available_options) == 0:
        if return_selected_option:
            return None, '0'
        return None
    selected_option = random.choice(available_options)

    ## OPTION A: extend pre-gain and flip a gain (turn to loss) that starts at doubled bp
    if selected_option == 'A':
        cur_post_gain, pre_start_end = random.choice([(x, 'end') for x in all_post_gains_start_at_pre_end] + [(x, 'start') for x in all_post_gains_end_at_pre_start])

        event_proposal = _create_proposal_from_cur_events(
            cur_events, remove_pre=[cur_pre_gain],
            add_pre=[(cur_post_gain[0], cur_pre_gain[1]) if pre_start_end == 'start' else (cur_pre_gain[0], cur_post_gain[1])],
            remove_post=[cur_post_gain], add_post=[(cur_post_gain[1], cur_post_gain[0])]
        )

    # OPTION B: shorten pre-gain and flip a loss (turn to gain) that ends at doubled bp but starts after pre-gain starts
    elif selected_option == 'B':
        cur_post_loss, pre_start_end = random.choice([(x, 'start') for x in all_post_losses_end_at_pre_start_and_inside_pre] + [(x, 'end') for x in all_post_losses_start_at_pre_end_and_inside_pre])
        new_pre_gain = (cur_post_loss[0], cur_pre_gain[1]) if pre_start_end == 'start' else (cur_pre_gain[0], cur_post_loss[1])

        if not _removed_pre_gain_passes_loh_filter(cur_events, cur_pre_gain, cn_profile, which='shorten_pre_gain', new_pre_gain=new_pre_gain, total_cn=total_cn):
            if return_selected_option:
                return None, selected_option
            return None

        event_proposal = _create_proposal_from_cur_events(
            cur_events, remove_pre=[cur_pre_gain], add_pre=[new_pre_gain],
            remove_post=[cur_post_loss], add_post=[(cur_post_loss[1], cur_post_loss[0])]
        )

    # OPTION C: extend pre-gain and extend two losses inwards from either side
    elif selected_option == 'C':
        pre_start_end_prob = np.array([min(len(all_post_losses_end_at_pre_start), len(all_post_losses_left)), 
                                       min(len(all_post_losses_start_at_pre_end), len(all_post_losses_right))])
        pre_start_end = np.random.choice(['start', 'end'], p=pre_start_end_prob/np.sum(pre_start_end_prob))

        if pre_start_end == 'start':
            cur_post_loss_left = random.choice(all_post_losses_left)
            cur_post_loss_right = random.choice(all_post_losses_end_at_pre_start)
            new_pre_gain = (cur_post_loss_left[0], cur_pre_gain[1])
        else:
            cur_post_loss_left = random.choice(all_post_losses_start_at_pre_end)
            cur_post_loss_right = random.choice(all_post_losses_right)
            new_pre_gain = (cur_pre_gain[0], cur_post_loss_right[1])

        event_proposal = _create_proposal_from_cur_events(
            cur_events, remove_pre=[cur_pre_gain], add_pre=[new_pre_gain],
            remove_post=[cur_post_loss_left, cur_post_loss_right],
            add_post=[(cur_post_loss_right[1], cur_post_loss_left[1]), (cur_post_loss_right[0], cur_post_loss_left[0])]
        )

    # OPTION D: shorten pre-gain and shorten two losses outwards (opposite of C)
    elif selected_option == 'D':

        pre_start_end_prob = np.array([min(len(all_post_losses_end_at_pre_start), len(all_post_losses_left_partly_inside_pre)), 
                                       min(len(all_post_losses_start_at_pre_end), len(all_post_losses_right_partly_inside_pre))])
        pre_start_end = np.random.choice(['start', 'end'], p=pre_start_end_prob/np.sum(pre_start_end_prob))

        if pre_start_end == 'start':
            cur_post_loss_left = random.choice(all_post_losses_left_partly_inside_pre)
            cur_post_loss_right = random.choice(all_post_losses_end_at_pre_start)
            new_pre_gain = (cur_post_loss_left[0], cur_pre_gain[1])
        else:
            cur_post_loss_left = random.choice(all_post_losses_start_at_pre_end)
            cur_post_loss_right = random.choice(all_post_losses_right_partly_inside_pre)
            new_pre_gain = (cur_pre_gain[0], cur_post_loss_right[1])

        if not _removed_pre_gain_passes_loh_filter(cur_events, cur_pre_gain, cn_profile, which='shorten_pre_gain', new_pre_gain=new_pre_gain, total_cn=total_cn):
            if return_selected_option:
                return None, selected_option
            return None

        event_proposal = _create_proposal_from_cur_events(
            cur_events, remove_pre=[cur_pre_gain], add_pre=[new_pre_gain],
            remove_post=[cur_post_loss_left, cur_post_loss_right],
            add_post=[(cur_post_loss_right[1], cur_post_loss_left[1]), (cur_post_loss_right[0], cur_post_loss_left[0])]
        )

    # OPTION E: extend pre-gain and change two events with the same start to the left/right from right/left side (gains will turn to losses)
    elif selected_option == 'E':

        # Randomly select a group and sample two tuples from the selected group
        selected_group, left_right = random.choice(
            [(x, 'left') for x in all_post_events_start_left_same_start.values()] +
            [(x, 'right') for x in all_post_events_end_right_same_end.values()])
        cur_post_events = random.sample(selected_group, 2)
        assert ((left_right == "right" and cur_post_events[0][1] == cur_post_events[1][1]) or
                (left_right == "left" and cur_post_events[0][0] == cur_post_events[1][0]))

        event_proposal = _create_proposal_from_cur_events(
            cur_events, remove_pre=[cur_pre_gain], remove_post=cur_post_events,
            add_pre=[(cur_pre_gain[0], cur_post_events[0][1]) if left_right == 'right' else (cur_post_events[0][0], cur_pre_gain[1])],
            add_post=([(cur_post_events[0][0], cur_pre_gain[1]), (cur_post_events[1][0], cur_pre_gain[1])] if left_right == 'right' else
                      [(cur_pre_gain[0], cur_post_events[0][1]), (cur_pre_gain[0], cur_post_events[1][1])])
        )

    # OPTION F: shorten pre-gain and shorten two losses with the same end to the right from left side (opposite of E)
    elif selected_option == 'F':

        # Randomly select a group and sample two tuples from the selected group
        selected_group, left_right = random.choice(
            [(x, 'left') for x in all_post_events_end_left_and_inside_same_end.values()] +
            [(x, 'right') for x in all_post_events_start_right_and_inside_same_start.values()])
        cur_post_losses = random.sample(selected_group, 2)
        assert ((left_right == "right" and cur_post_losses[0][0] == cur_post_losses[1][0]) or
                (left_right == "left" and cur_post_losses[0][1] == cur_post_losses[1][1]))
        new_pre_gain = (cur_post_losses[0][0], cur_pre_gain[1]) if left_right == 'right' else (cur_pre_gain[0], cur_post_losses[0][1])

        if not _removed_pre_gain_passes_loh_filter(cur_events, cur_pre_gain, cn_profile, which='shorten_pre_gain', new_pre_gain=new_pre_gain, total_cn=total_cn):
            if return_selected_option:
                return None, selected_option
            return None

        event_proposal = _create_proposal_from_cur_events(
            cur_events, remove_pre=[cur_pre_gain], remove_post=cur_post_losses, add_pre=[new_pre_gain],
            add_post=([(cur_pre_gain[0], cur_post_losses[0][1]), (cur_pre_gain[0], cur_post_losses[1][1])] if left_right == 'right' else
                      [(cur_post_losses[0][0], cur_pre_gain[1]), (cur_post_losses[1][0], cur_pre_gain[1])]))
    else:
        raise ValueError('selected option not available')
    
    assert len(event_proposal[0]) + len(event_proposal[1]) == len(cur_events[0]) + len(cur_events[1])
    if return_selected_option:
        return event_proposal, selected_option
    else:
        return event_proposal
    

def proposal_wgd_extend_shorten_pre_loss(cur_events, cn_profile, return_selected_option=False, total_cn=False):

    # select a random pre-loss
    all_pre_losses = [event for event in cur_events[0] if event[0] > event[1]]
    if len(all_pre_losses) == 0 or len(cur_events[1]) == 0:
        if return_selected_option:
            return None, '0'
        return None
    cur_pre_loss = random.choice(all_pre_losses)

    all_post_gains = [event for event in cur_events[1] if event[0] < event[1]]
    all_post_losses = [event for event in cur_events[1] if event[0] > event[1]]

    # for A/B
    all_post_losses_start_at_pre_end = [event for event in all_post_losses if event[0] == cur_pre_loss[1]]
    all_post_losses_end_at_pre_start = [event for event in all_post_losses if event[1] == cur_pre_loss[0]]
    all_post_gains_start_at_pre_end_and_inside_pre = [event for event in all_post_gains if event[0] == cur_pre_loss[1] and event[1] < cur_pre_loss[0]]
    all_post_gains_end_at_pre_start_and_inside_pre = [event for event in all_post_gains if event[1] == cur_pre_loss[0] and event[0] > cur_pre_loss[1]]
    # for C/D
    all_post_gains_start_at_pre_end = [event for event in all_post_gains if event[0] == cur_pre_loss[1]]
    all_post_gains_end_at_pre_start = [event for event in all_post_gains if event[1] == cur_pre_loss[0]]
    all_post_gains_right = [event for event in all_post_gains if event[0] > cur_pre_loss[0]]
    all_post_gains_left = [event for event in all_post_gains if event[1] < cur_pre_loss[1]]
    all_post_gains_left_partly_inside_pre = [event for event in all_post_gains if event[1] > cur_pre_loss[1] and event[1] < cur_pre_loss[0] and event[0] < cur_pre_loss[1]]
    all_post_gains_right_partly_inside_pre = [event for event in all_post_gains if event[0] < cur_pre_loss[0] and event[0] > cur_pre_loss[1] and event[1] > cur_pre_loss[0]]
    # for E/F
    all_post_events_start_right = [event for event in cur_events[1] if event[0] > cur_pre_loss[0]]
    all_post_events_end_left = [event for event in cur_events[1] if event[1] < cur_pre_loss[1]]
    all_post_events_start_right_same_start = _select_events_with_same_start_or_end(all_post_events_start_right, start_end='start')
    all_post_events_end_left_same_end = _select_events_with_same_start_or_end(all_post_events_end_left, start_end='end')
    all_post_events_start_left_and_inside = [event for event in cur_events[1] if event[0] > cur_pre_loss[1] and event[0] < cur_pre_loss[0]]
    all_post_events_end_right_and_inside = [event for event in cur_events[1] if event[1] > cur_pre_loss[1] and event[1] < cur_pre_loss[0]]
    all_post_events_start_left_and_inside_same_start = _select_events_with_same_start_or_end(all_post_events_start_left_and_inside, start_end='start')
    all_post_events_end_right_and_inside_same_end = _select_events_with_same_start_or_end(all_post_events_end_right_and_inside, start_end='end')

    options = [
        ('A', len(all_post_losses_start_at_pre_end) > 0 or len(all_post_losses_end_at_pre_start) > 0),
        ('B', len(all_post_gains_start_at_pre_end_and_inside_pre) > 0 or len(all_post_gains_end_at_pre_start_and_inside_pre) > 0),
        ('C', ((len(all_post_gains_start_at_pre_end) > 0 and len(all_post_gains_left) > 0) or 
               (len(all_post_gains_end_at_pre_start) > 0 and len(all_post_gains_right) > 0))),
        ('D', (len(all_post_gains_start_at_pre_end) > 0 and len(all_post_gains_left_partly_inside_pre) > 0) or
              (len(all_post_gains_end_at_pre_start) > 0 and len(all_post_gains_right_partly_inside_pre) > 0)),
        ('E', len(all_post_events_start_right_same_start) > 0 or len(all_post_events_end_left_same_end) > 0),
        ('F', len(all_post_events_start_left_and_inside_same_start) > 0 or len(all_post_events_end_right_and_inside_same_end) > 0),
    ]

    available_options = [option for option, condition in options if condition]
    if len(available_options) == 0:
        if return_selected_option:
            return None, '0'
        return None
    selected_option = random.choice(available_options)

    ## OPTION A: extend pre-loss and flip a loss (turn to gain) that starts at doubled bp
    if selected_option == 'A':
        cur_post_loss, pre_start_end = random.choice([(x, 'end') for x in all_post_losses_start_at_pre_end] + [(x, 'start') for x in all_post_losses_end_at_pre_start])
        new_pre_loss = (cur_post_loss[0], cur_pre_loss[1]) if pre_start_end == 'start' else (cur_pre_loss[0], cur_post_loss[1])
        if not _added_pre_loss_passes_loh_filter(cur_events, cur_pre_loss, cn_profile, which='extend_pre_loss',
                                                 new_pre_loss=new_pre_loss, total_cn=total_cn):
            if return_selected_option:
                return None, 'A'
            return None

        event_proposal = _create_proposal_from_cur_events(
            cur_events, remove_pre=[cur_pre_loss],
            add_pre=[new_pre_loss],
            remove_post=[cur_post_loss], add_post=[(cur_post_loss[1], cur_post_loss[0])]
        )

    # OPTION B: shorten pre-loss and flip a gain (turn to loss) that ends at doubled bp but starts after pre-loss starts
    elif selected_option == 'B':
        cur_post_gain, pre_start_end = random.choice([(x, 'start') for x in all_post_gains_end_at_pre_start_and_inside_pre] + [(x, 'end') for x in all_post_gains_start_at_pre_end_and_inside_pre])
        new_pre_loss = (cur_post_gain[0], cur_pre_loss[1]) if pre_start_end == 'start' else (cur_pre_loss[0], cur_post_gain[1])

        event_proposal = _create_proposal_from_cur_events(
            cur_events, remove_pre=[cur_pre_loss], add_pre=[new_pre_loss],
            remove_post=[cur_post_gain], add_post=[(cur_post_gain[1], cur_post_gain[0])]
        )

    # OPTION C: extend pre-loss and extend two gains inwards from either side
    elif selected_option == 'C':
        pre_start_end_prob = np.array([min(len(all_post_gains_end_at_pre_start), len(all_post_gains_right)), 
                                       min(len(all_post_gains_start_at_pre_end), len(all_post_gains_left))])
        pre_start_end = np.random.choice(['pre-start', 'pre-end'], p=pre_start_end_prob/np.sum(pre_start_end_prob))

        if pre_start_end == 'pre-start':
            cur_post_gain_left = random.choice(all_post_gains_end_at_pre_start)
            cur_post_gain_right = random.choice(all_post_gains_right)
            new_pre_loss = (cur_post_gain_right[0], cur_pre_loss[1])
        else:
            cur_post_gain_left = random.choice(all_post_gains_left)
            cur_post_gain_right = random.choice(all_post_gains_start_at_pre_end)
            new_pre_loss = (cur_pre_loss[0], cur_post_gain_left[1])

        if not _added_pre_loss_passes_loh_filter(cur_events, cur_pre_loss, cn_profile, which='extend_pre_loss',
                                                 new_pre_loss=new_pre_loss, total_cn=total_cn):
            if return_selected_option:
                return None, 'C'
            return None

        event_proposal = _create_proposal_from_cur_events(
            cur_events, remove_pre=[cur_pre_loss], add_pre=[new_pre_loss],
            remove_post=[cur_post_gain_left, cur_post_gain_right],
            add_post=[(cur_post_gain_left[0], cur_post_gain_right[0]), (cur_post_gain_left[1], cur_post_gain_right[1])]
        )

    # OPTION D: shorten pre-loss and shorten two gains outwards (opposite of C)
    elif selected_option == 'D':
        pre_start_end_prob = np.array([min(len(all_post_gains_end_at_pre_start), len(all_post_gains_right_partly_inside_pre)), 
                                       min(len(all_post_gains_start_at_pre_end), len(all_post_gains_left_partly_inside_pre))])
        pre_start_end = np.random.choice(['pre-start', 'pre-end'], p=pre_start_end_prob/np.sum(pre_start_end_prob))

        if pre_start_end == 'pre-start':
            cur_post_gain_left = random.choice(all_post_gains_end_at_pre_start)
            cur_post_gain_right = random.choice(all_post_gains_right_partly_inside_pre)
            new_pre_loss = (cur_post_gain_right[0], cur_pre_loss[1])
        else:
            cur_post_gain_left = random.choice(all_post_gains_left_partly_inside_pre)
            cur_post_gain_right = random.choice(all_post_gains_start_at_pre_end)
            new_pre_loss = (cur_pre_loss[0], cur_post_gain_left[1])

        event_proposal = _create_proposal_from_cur_events(
            cur_events, remove_pre=[cur_pre_loss], add_pre=[new_pre_loss],
            remove_post=[cur_post_gain_left, cur_post_gain_right],
            add_post=[(cur_post_gain_left[0], cur_post_gain_right[0]), (cur_post_gain_left[1], cur_post_gain_right[1])]
        )

    # OPTION E: extend pre-loss and change two events with the same start to the left/right from right/left side (losses will turn to gains)
    elif selected_option == 'E':
        # Randomly select a group and sample two tuples from the selected group
        selected_group, left_right = random.choice(
            [(x, 'left') for x in all_post_events_end_left_same_end.values()] +
            [(x, 'right') for x in all_post_events_start_right_same_start.values()])
        cur_post_events = random.sample(selected_group, 2)
        assert ((left_right == "right" and cur_post_events[0][0] == cur_post_events[1][0]) or
                (left_right == "left" and cur_post_events[0][1] == cur_post_events[1][1]))
        new_pre_loss = (cur_post_events[0][0], cur_pre_loss[1]) if left_right == 'right' else (cur_pre_loss[0], cur_post_events[0][1])

        if not _added_pre_loss_passes_loh_filter(cur_events, cur_pre_loss, cn_profile, which='extend_pre_loss',
                                                 new_pre_loss=new_pre_loss, total_cn=total_cn):
            if return_selected_option:
                return None, 'E'
            return None

        event_proposal = _create_proposal_from_cur_events(
            cur_events, remove_pre=[cur_pre_loss], remove_post=cur_post_events, add_pre=[new_pre_loss],
            add_post=([(cur_pre_loss[0], cur_post_events[0][1]), (cur_pre_loss[0], cur_post_events[1][1])] if left_right == 'right' else
                      [(cur_post_events[0][0], cur_pre_loss[1]), (cur_post_events[1][0], cur_pre_loss[1])])
        )

    # OPTION F: shorten pre-loss and shorten two gains with the same end to the right from left side (opposite of E)
    elif selected_option == 'F':
        # Randomly select a group and sample two tuples from the selected group
        selected_group, left_right = random.choice(
            [(x, 'left') for x in all_post_events_start_left_and_inside_same_start.values()] +
            [(x, 'right') for x in all_post_events_end_right_and_inside_same_end.values()])
        cur_post_gains = random.sample(selected_group, 2)
        assert ((left_right == "right" and cur_post_gains[0][1] == cur_post_gains[1][1]) or
                (left_right == "left" and cur_post_gains[0][0] == cur_post_gains[1][0]))

        event_proposal = _create_proposal_from_cur_events(
            cur_events, remove_pre=[cur_pre_loss], remove_post=cur_post_gains,
            add_pre=[(cur_pre_loss[0], cur_post_gains[0][1]) if left_right == 'right' else (cur_post_gains[0][0], cur_pre_loss[1])],
            add_post=([(cur_post_gains[0][0], cur_pre_loss[1]), (cur_post_gains[1][0], cur_pre_loss[1])] if left_right == 'right' else
                      [(cur_pre_loss[0], cur_post_gains[0][1]), (cur_pre_loss[0], cur_post_gains[1][1])]))
    else:
        raise ValueError('selected option not available')


    assert len(event_proposal[0]) + len(event_proposal[1]) == len(cur_events[0]) + len(cur_events[1])
    if return_selected_option:
        return event_proposal, selected_option
    else:
        return event_proposal


def proposal_wgd_switch_loh_loss(cur_events, cn_profile, return_pre_post=False):
    loh_pos = np.where(cn_profile == 0)[0]
    if len(loh_pos) == 0:
        if return_pre_post:
            return None, '0'
        else:
            return None
    potential_loh_losses = [(i+1, i) for i in loh_pos]
    all_loh_losses = (
        [(x, 'pre') for x in cur_events[0] if x in potential_loh_losses] +
        [(x, 'post') for x in cur_events[1] if x in potential_loh_losses]
    )
    if len(all_loh_losses) == 0:
        if return_pre_post:
            return None, '0'
        else:
            return None
    cur_loh_loss, pre_post = random.choice(all_loh_losses)

    event_proposal = _create_proposal_from_cur_events(
        cur_events,
        remove_pre=[cur_loh_loss] if pre_post=='pre' else [],
        remove_post=[cur_loh_loss] if pre_post=='post' else [],
        add_pre=[cur_loh_loss] if pre_post=='post' else [],
        add_post=[cur_loh_loss] if pre_post=='pre' else []
        )
    if return_pre_post:
        return event_proposal, pre_post
    else:
        return event_proposal


def _proposal_is_unchanged(event_proposal, best_events, has_wgd):
    if has_wgd:
        if len(event_proposal[0]) != len(best_events[0]) or len(event_proposal[1]) != len(best_events[1]):
            return False
        return ((sorted(event_proposal[0]) == sorted(best_events[0])) and
                (sorted(event_proposal[1]) == sorted(best_events[1])))
    else:
        return (event_proposal == best_events).all()


def _get_events_from_diff(overall_best_diffs, has_wgd):
    if has_wgd:
        best_events_pre = _get_events_from_diff(overall_best_diffs[0], has_wgd=False)
        best_events_post = _get_events_from_diff(overall_best_diffs[1], has_wgd=False)
        overall_best_events = [best_events_pre, best_events_post]
    else:
        overall_best_events = np.stack([
            np.argmax(np.abs(overall_best_diffs), axis=1),
            overall_best_diffs.shape[1] - np.argmax(np.abs(overall_best_diffs)[:, ::-1], axis=1)]).T
        overall_best_events[overall_best_diffs.min(axis=1) == -1] = overall_best_events[overall_best_diffs.min(axis=1) == -1][:, ::-1]
    return overall_best_events
