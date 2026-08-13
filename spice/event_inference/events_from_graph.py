import itertools
from copy import deepcopy
from functools import cache, lru_cache, reduce
from collections import Counter, namedtuple
from time import time

import numpy as np
import pandas as pd
from ortools.sat.python import cp_model
import fstlib

from spice import config
from spice.utils import create_full_df_from_diff_df, chrom_id_from_id
from spice.logging import get_logger, log_debug
from spice.random_state import np_rng
from spice.event_inference.fst_assets import get_diploid_fsa, T_forced_WGD
from spice.event_inference.fsts import fsa_from_string
from spice.event_inference.data_structures import Diff, FullPaths

sv_matching_threshold = config['params']['sv_matching_threshold']
diploid_fsa = get_diploid_fsa(total_copy_numbers=False)
diploid_fsa_total_cn = get_diploid_fsa(total_copy_numbers=True)

logger = get_logger(__name__)


def _make_solver_deterministic(solver):
    """Pin a CP-SAT solver to a reproducible search.

    Two things make the default solver non-reproducible: it seeds its own search randomly, and it
    runs a portfolio of strategies in parallel workers, so which solution comes back first (and in
    which order `enumerate_all_solutions` yields them) is a race. One worker plus a seed drawn from
    this thread's stream fixes both. Note this does not save a run that hits `max_time_in_seconds`:
    a wall-clock cutoff is non-deterministic by construction.
    """
    solver.parameters.random_seed = int(np_rng().randint(0, 2 ** 31 - 1))
    solver.parameters.num_workers = 1


def full_paths_from_graph_with_sv(cur_id, is_wgd, sv_data, chrom_segments, chrom,
                                  time_limit=60, path_limit=None, time_limit_loh_filters=None, total_cn=False,
                                  sv_matching_threshold=sv_matching_threshold, use_cache=True,
                                  skip_loh_checks=False, all_loh_solutions=True, **kwargs):
    cur_sample, cur_chrom, cur_allele = cur_id.split(':')


    chrom_id = chrom_id_from_id(cur_id)
    if sv_data is not None:
        cur_sv_data = sv_data.query('chrom_id == @chrom_id').copy()
    else:
        cur_sv_data = None
    cur_chrom_segments = chrom_segments.query('id == @cur_id').copy()
    assert len(cur_chrom_segments) > 0, f'No segments found for {cur_id}'
    cn_profile = cur_chrom_segments['cn'].values
    log_debug(logger, f'Creating full paths ({"without" if sv_data is None else "with"} SV overlaps) for {cur_id} {"WGD" if is_wgd else "noWGD"}. Nr events: {chrom.n_events}. CN profile: {cn_profile}')
    log_debug(logger, 'Using cache' if use_cache else 'Not using cache')
    
    if is_wgd:
        diffs, sv_selected_events = _full_paths_from_graph_with_sv_wgd(
            cn_profile=cn_profile,
            cur_sv_data=cur_sv_data,
            cur_chrom_segments=cur_chrom_segments,
            use_cache=use_cache,
            all_loh_solutions=all_loh_solutions,
            sv_matching_threshold=sv_matching_threshold,
            total_cn=total_cn,
            skip_loh_checks=skip_loh_checks,
            **kwargs)
    else:
        diffs, sv_selected_events = _full_paths_from_graph_with_sv_nowgd(
            cn_profile=cn_profile,
            cur_sv_data=cur_sv_data,
            cur_chrom_segments=cur_chrom_segments,
            use_cache=use_cache,
            time_limit=time_limit,
            path_limit=path_limit,
            all_loh_solutions=all_loh_solutions,
            skip_loh_checks=skip_loh_checks,
            time_limit_loh_filters=time_limit_loh_filters,
            sv_matching_threshold=sv_matching_threshold,
            total_cn=total_cn,
            **kwargs)
    # sorted(), not raw set order: Diff carries string fields, and CPython randomises string hashing
    # per process, so `enumerate(set(...))` handed out a different index to each event on every run.
    # Those indices are what the solution Counters, the pickled FullPaths and ultimately the row
    # order of final_events.tsv are built from -- the one thing a fixed seed could not pin down.
    unique_events = {i: d for i, d in enumerate(sorted(set(item for sublist in diffs for item in sublist)))}
    unique_events_reversed = {v: k for k, v in unique_events.items()}
    diffs = [[unique_events_reversed[event] for event in diff] for diff in diffs]
    solutions = [Counter(diff) for diff in diffs]
    # this is necessary because LOHs can create duplicate solutions (e.g. for profile 010)
    # sorted for the same reason: the de-duplicating set is iterated, so its order would otherwise
    # decide the order of the solution list.
    unique_solutions = [Counter({k: v for k, v in x})
                        for x in sorted({frozenset(c.items()) for c in solutions}, key=sorted)]
    log_debug(logger, f"Found {len(unique_events)} unique events")
    log_debug(logger, f"Found {len(solutions)} solutions of which {len(unique_solutions)} are unique")
    assert all([solution.total() == (chrom.n_events) for solution in unique_solutions]), f"expected nr of events: {chrom.n_events}. nr of events per solution: {[solution.total() for solution in unique_solutions]}"

    if any([event.diff.find('1')==-1 for event in unique_events.values()]):
        raise ValueError('Invalid empty events found. This usually means that the number of events '
                         'was calculated incorrectly for WGD samples.')

    if len(unique_solutions) == 1:
        if sv_selected_events is not None and (chrom.n_events - len(sv_selected_events)) <= 1 and chrom.n_events > 1:
            solved = 'sv'
        else:
            solved = 'unamb'
    else:
        solved = 'full'

    solved_chrom = FullPaths(
        id=cur_id, sample=cur_sample, chrom=cur_chrom, allele=cur_allele,
        cn_profile=cn_profile,
        solutions=unique_solutions, events=unique_events,
        n_solutions=len(unique_solutions), n_events=chrom.n_events,
        is_wgd=is_wgd, solved=solved)

    return solved_chrom


def _full_paths_implementation_nowgd(
        sv_selected_events, starts, ends, starts_,
        cn_profile, use_cache=True, time_limit=60, path_limit=None):

    if len(starts) > 0:
        paths = get_events_from_graph_step(
            starts, ends, use_cache=use_cache, time_limit=time_limit, path_limit=path_limit)
        assert [len(p) == len(paths[0]) for p in paths], f"Paths have different number of events: {[len(p) for p in paths]}"
        assert len(paths[0]) + (0 if sv_selected_events is None else len(sv_selected_events)) == len(starts_), f'Number of paths do not match: {len(paths[0])} + {0 if sv_selected_events is None else len(sv_selected_events)} vs {len(starts_)}'
    else:
        paths = []
        assert sv_selected_events is not None, f'len(start) == 0 but no sv_selected_events. starts = {starts}'
        assert len(sv_selected_events) == len(starts_), f'Number of SV selected events do not match: {len(sv_selected_events)} vs {len(starts_)}'

    if paths is None:
        raise NotImplementedError('No solutions (paths) found')

    diffs_ = diffs = get_events_diff_from_coords(paths, cn_profile, lexsort_diffs=True, filter_missed_lohs=False, fail_if_empty=False)
    assert diffs is not None, 'No viable diffs found'
    if sv_selected_events is not None:
        diffs_sv_solved = get_events_diff_from_coords([sv_selected_events], cn_profile, lexsort_diffs=True, filter_missed_lohs=False)
        assert diffs_sv_solved is not None, 'No viable diffs found'
        diffs_sv_solved = diffs_sv_solved[0]
        if len(diffs_) > 0:
            diffs = [np.concatenate([diffs_sv_solved, diff]) for diff in diffs_]
        else:
            diffs = [diffs_sv_solved]
        
    assert [len(d) == len(diffs[0]) for d in diffs], f"Diffs have different number of events: {[len(d) for d in diffs]}"
    assert len(diffs[0]) == len(starts_), f"Number of diffs do not match: {len(diffs[0])} ({len(diffs_[0])} + {len(diffs_sv_solved) if diffs_sv_solved is not None else 0}) vs {len(starts_)}"

    # <= -1 is needed because for WGD it can be -2
    diffs = [diff for diff in diffs if (diff[:, cn_profile==0] <= -1).any(axis=0).all()]
    log_debug(logger, f"{len(diffs)} viable solutions (diffs) found")

    return diffs


def _full_paths_from_graph_with_sv_nowgd(
        cn_profile, cur_sv_data, cur_chrom_segments, skip_loh_checks=False,
        use_cache=True, time_limit=60, path_limit=None, time_limit_loh_filters=60, total_cn=False,
        sv_matching_threshold=sv_matching_threshold, all_loh_solutions=True):

    starts_, ends_ = get_starts_and_ends(cn_profile, prior_profile=None, loh_adjust=True, total_cn=total_cn)
    log_debug(logger, f'CN profile: {cn_profile}')
    log_debug(logger, f'Starts: {starts_}')
    log_debug(logger, f'Ends: {ends_}')

    log_debug(logger, f'Current ID has {len(starts_)} events and {len(cur_sv_data) if cur_sv_data is not None else 0} SVs')
    if cur_sv_data is not None:
        sv_selected_events, starts, ends = connect_start_ends_using_svs(
            cur_chrom_segments, cur_sv_data, starts_, ends_, sv_matching_threshold=sv_matching_threshold)
        log_debug(logger, f'After SV matching, {len(sv_selected_events) if sv_selected_events is not None else 0} events are selected. {len(starts)} events remain')
        log_debug(logger, f'Selected SV events: {sv_selected_events if sv_selected_events is not None else "None"}')
        assert len(starts) == len(ends), f'Number of starts and ends do not match: {len(starts)} vs {len(ends)}'
        assert sv_selected_events is None or len(sv_selected_events) + len(starts) == len(starts_), f'Number of SV selected events do not match: {len(sv_selected_events)} + {len(starts)} vs {len(starts_)}'
    else:
        starts = starts_
        ends = ends_
        sv_selected_events = None

    diffs = _full_paths_implementation_nowgd(
        sv_selected_events, starts, ends, starts_, cn_profile, use_cache, time_limit, path_limit)
    if len(diffs) == 0 and sv_selected_events is not None:
        logger.warning(f'No viable solutions (diffs) found with SVs. Trying without SVs')
        sv_selected_events = None
        diffs = _full_paths_implementation_nowgd(
            None, starts_, ends_, starts_, cn_profile, use_cache, time_limit, path_limit)  

    if not skip_loh_checks and 0 in cn_profile:
        diffs = loh_filters_for_graph_result_diffs(
            diffs, cn_profile, total_cn=total_cn,
            return_all_solutions=all_loh_solutions, shuffle_diffs=True)

        log_debug(logger, f"{len(diffs)} solutions after LOH filters")
    if len(diffs) == 0:
        raise NotImplementedError('No viable solutions (diffs) found')

    diffs = [[Diff(diff=''.join(map(str, np.abs(x))), is_gain=x.max()==1, wgd="nowgd") for x in diff] for diff in diffs]

    return diffs, sv_selected_events


def _full_paths_from_graph_with_sv_wgd(
        cn_profile, cur_sv_data, cur_chrom_segments, total_cn=False, skip_loh_checks=False,
        use_cache=True, all_loh_solutions=True,
        sv_matching_threshold=sv_matching_threshold, **kwargs):
    
    paths = get_events_from_graph_wgd(
        cn_profile, use_cache=use_cache, adjust_loh_wgd=True, assert_correct=True, total_cn=total_cn,
        remove_duplicates=True, **kwargs)
    
    if cur_sv_data is not None and cur_chrom_segments is not None:
        sv_selected_paths, sv_selected_events = get_sv_selected_paths_wgd(
            paths, cur_sv_data, cur_chrom_segments, sv_matching_threshold=sv_matching_threshold)
        log_debug(logger, f'SVs selected {len(sv_selected_paths)} paths{". not using SVs here" if len(sv_selected_paths) == 0 else ""}')
        if len(sv_selected_paths) > 0:
            paths = sv_selected_paths
    else:
        sv_selected_events = None

    diffs = get_events_diff_from_coords_wgd(paths, cn_profile, lexsort_diffs=True, filter_missed_lohs=False)
    if not skip_loh_checks and 0 in cn_profile:
        log_debug(logger, f"Performing LOH filters (returning {'all solutions' if all_loh_solutions else 'a single solution'})")
        diffs = loh_filters_for_graph_result_diffs_wgd(
            diffs, cn_profile, total_cn=total_cn,
            return_all_solutions=all_loh_solutions, shuffle_diffs=True)
        logger.debug(f"{len(diffs)} solutions after LOH filters")

    diffs = [[Diff(diff=''.join(map(str, np.abs(x))), is_gain=x.max()==1, wgd="pre") for x in diff[0]] +
             [Diff(diff=''.join(map(str, np.abs(x))), is_gain=x.max()==1, wgd="post") for x in diff[1]]
             for diff in diffs]
    assert all([len(x) == len(diffs[0]) for x in diffs])

    return diffs, sv_selected_events


def raw_events_from_FullPaths(full_paths, wgd=False):
    full_path_events = [[full_paths.events[x] for x in list(sol.elements())] for sol in full_paths.solutions]
    if wgd:
        full_path_events = [
            [[(x.diff.find('1'), (x.diff.rfind('1')+1)) if x.is_gain else
                (x.diff.rfind('1')+1, x.diff.find('1')) for x in sol if x.wgd=='pre'],
            [(x.diff.find('1'), (x.diff.rfind('1')+1)) if x.is_gain else
                (x.diff.rfind('1')+1, x.diff.find('1')) for x in sol if x.wgd=='post']]
            for sol in full_path_events]
    else:
        full_path_events = [
            [(x.diff.find('1'), (x.diff.rfind('1')+1)) if x.is_gain else
                (x.diff.rfind('1')+1, x.diff.find('1')) for x in sol]
            for sol in full_path_events]

    return full_path_events


def connect_start_ends_using_svs(cur_chrom_segments, cur_sv_data, starts, ends, sv_matching_threshold=sv_matching_threshold):

    unique_starts = np.unique(starts)
    unique_ends = np.unique(ends)
    cur_segment_breakpoints_starts = np.append(cur_chrom_segments['start'].values, cur_chrom_segments['end'].values[-1])
    cur_segment_breakpoints_ends =  np.append(cur_chrom_segments['start'].values[0], cur_chrom_segments['end'].values)

    if 'start_for_overlap' not in cur_sv_data.columns or 'end_for_overlap' not in cur_sv_data.columns:
        cur_sv_data['start_for_overlap'] = cur_sv_data['start']
        cur_sv_data['end_for_overlap'] = cur_sv_data['end']

        cur_sv_data.loc[cur_sv_data['svclass'] == 'DEL', 'start_for_overlap'] = cur_sv_data.loc[cur_sv_data['svclass'] == 'DEL', 'end']
        cur_sv_data.loc[cur_sv_data['svclass'] == 'DEL', 'end_for_overlap'] = cur_sv_data.loc[cur_sv_data['svclass'] == 'DEL', 'start']

    sv_start_matches = np.abs(cur_segment_breakpoints_starts[unique_starts][:, None] - cur_sv_data['start_for_overlap'].values) < sv_matching_threshold
    sv_end_matches = np.abs(cur_segment_breakpoints_ends[unique_ends][:, None] - cur_sv_data['end_for_overlap'].values) < sv_matching_threshold
    # invalid_svs = (sv_start_matches.sum(axis=0) == 0) | (sv_end_matches.sum(axis=0) == 0)
    valid_svs = np.logical_and((sv_start_matches.sum(axis=0) == 1), (sv_end_matches.sum(axis=0) == 1))

    if not valid_svs.any():
        return None, starts, ends

    if valid_svs.any():
        sv_selected_events = np.stack([
            unique_starts[sv_start_matches[:, valid_svs].argmax(axis=0)],
            unique_ends[sv_end_matches[:, valid_svs].argmax(axis=0)]]
            ).T
        assert sv_selected_events.shape[0] == valid_svs.sum() and sv_selected_events.shape[1] == 2, f"sv_selected_events: {sv_selected_events.shape}"
        correct_gain_loss = (np.diff(sv_selected_events, axis=1) > 0)[:, 0] == (cur_sv_data['svclass'][valid_svs].values=='DUP')
        sv_selected_events = sv_selected_events[correct_gain_loss]
        starts_ = np.array(list((Counter(starts) - Counter(sv_selected_events[:, 0])).elements())).copy()
        ends_ = np.array(list((Counter(ends) - Counter(sv_selected_events[:, 1])).elements())).copy()
    else:
        sv_selected_events = None
        starts_ = starts.copy()
        ends_ = ends.copy()

    return sv_selected_events, starts_, ends_


def get_starts_and_ends_for_wgd(cn_profile, adjust_loh_wgd=True, total_cn=False):
    if adjust_loh_wgd and 0 in cn_profile:

        profile_adjust_nowgd = adjust_profile_for_loh(cn_profile, wgd=False, total_cn=total_cn)
        profile_adjust_wgd = adjust_profile_for_loh(cn_profile, wgd=True, total_cn=total_cn)

        differing_loh_adjust = np.where(profile_adjust_nowgd != profile_adjust_wgd)[0]
        assert all(cn_profile[differing_loh_adjust] == 0)

        initial_start, initial_end = get_starts_and_ends(
            cn_profile,
            prior_profile=np.ones_like(cn_profile) * (4 if total_cn else 2),
            loh_adjust=True, wgd=False)
        all_starts, all_ends = [initial_start], [initial_end]
        loh_combinations = sum([list(itertools.combinations(differing_loh_adjust, i + 1)) for i in range(len(differing_loh_adjust))], [])
        for loh_comb in loh_combinations:
            cur_start = np.append(initial_start.copy(), np.array(loh_comb)+1)
            cur_end = np.append(initial_end.copy(), np.array(loh_comb))
            all_starts.append(cur_start)
            all_ends.append(cur_end)    

    else:
        starts, ends = get_starts_and_ends(
            cn_profile,
            prior_profile=np.ones_like(cn_profile) * (4 if total_cn else 2),
            loh_adjust=False, wgd=adjust_loh_wgd)
        all_starts, all_ends = [starts], [ends]
    return all_starts, all_ends


def get_events_from_graph_wgd(cn_profile, n_events=None, use_cache=True, adjust_loh_wgd=True, total_cn=False,
                              assert_correct=True, assert_unique=True, remove_duplicates=True):

    n_events = max(1, int(float(fstlib.score(T_forced_WGD, diploid_fsa_total_cn if total_cn else diploid_fsa,
                                             fsa_from_string(''.join(cn_profile.astype(str)))))) - 1)
    if total_cn and (cn_profile==0).all():
        n_events = 2
    # max is required in case of a whole-chrom LOH

    log_debug(logger, f"cn profile: {cn_profile}")
    log_debug(logger, f'nr of events (excluding WGD) = {n_events}')

    all_paths = []
    all_starts, all_ends = get_starts_and_ends_for_wgd(
        cn_profile, adjust_loh_wgd=adjust_loh_wgd, total_cn=total_cn)

    log_debug(logger, f"{'LOH in profile' if 0 in cn_profile else 'No LOH in profile'}. Initial starts: {all_starts[0]} and ends: {all_ends[0]}. Running {len(all_starts)} different LOH adjustments\n")

    for starts, ends in zip(all_starts, all_ends):
        log_debug(logger, f"Current LOH adjustment: {len(starts) - len(all_starts[0])} LOH adj added. Added starts: {np.sort(list((Counter(starts) - Counter(all_starts[0])).elements()))} and ends: {np.sort(list((Counter(ends) - Counter(all_ends[0])).elements()))}")

        base_paths = get_events_for_cur_start_ends_wgd(
            starts.copy(), ends.copy(), n_events, cn_profile, use_cache=use_cache, total_cn=total_cn)
        if len(base_paths) == 0:
            log_debug(logger, 'no base paths found, skipping extra bps to add\n')
            continue

        bps_to_add = get_wgd_bps_to_add(base_paths, starts, ends, loh=(0 in cn_profile))
        log_debug(logger, f'{len(base_paths)} base_paths')
        # log_debug(logger, f'base paths: {base_paths}')
        log_debug(logger, f'{len(bps_to_add)} initial bps to add: {bps_to_add}')

        new_paths = get_new_wgd_paths_by_adding_bps(
            starts, ends, cn_profile, n_events, bps_to_add, use_cache=use_cache, total_cn=total_cn)
        paths = base_paths + new_paths

        log_debug(logger, f'{len(new_paths)} new_paths')
        # log_debug(logger, f'new paths: {new_paths}')
        log_debug(logger, f'Found {len(paths)} paths in total\n')
        all_paths.extend(paths)

    log_debug(logger, f'Found {len(all_paths)} paths for all {len(all_starts)} LOH adjustments')

    if remove_duplicates:
        # remove duplicate paths
        paths_ = list(set([(tuple(sorted(x[0])), tuple(sorted(x[1]))) for x in all_paths]))
        log_debug(logger, f'Removed {len(all_paths) - len(paths_)} duplicate paths')
        all_paths = [(list(x[0]), list(x[1])) for x in paths_]
    elif assert_unique:
        # assert that all paths are unique
        paths_ = list(set([(tuple(sorted(x[0])), tuple(sorted(x[1]))) for x in all_paths]))
        assert len(paths_) == len(all_paths), f'found non-unique paths ({len(paths_)} vs {len(all_paths)}): {[k for k, v in Counter([(tuple(sorted(x[0])), tuple(sorted(x[1]))) for x in all_paths]).items() if v>1]}'

    if assert_correct:  
        log_debug(logger, f'Running final checks for {len(all_paths)} paths')
        # assert that there are paths
        assert len(all_paths) > 0, 'No paths found'

        # assert that all paths have the correct length
        assert all([len(x[0]) + len(x[1]) == n_events for x in all_paths]), f'not all paths have length {n_events}'

        # assert that diffs actually create the correct profile -> does not work for LOH
        paths_ = [(2*x[0] + x[1]) for x in all_paths]
        diffs = get_events_diff_from_coords(paths_, cn_profile, filter_missed_lohs=False, fail_if_empty=False)
        diffs_sum = np.clip([np.sum(d, axis=0) + (4 if total_cn else 2) for d in diffs], a_min=0, a_max=None)
        assert all([np.all((d == cn_profile)[cn_profile!=0]) for d in diffs_sum]), f'not all diffs create the correct profile'
        log_debug(logger, 'All checks passed')


    return all_paths


def get_events_from_graph_step(starts, ends, use_cache=False, time_limit=None, path_limit=None):
    if len(starts) == 0:
        return []
    starts = list(zip(*np.unique(starts, return_counts=True)))
    if use_cache:
        return list(_get_events_from_graph_step_cache(tuple(starts), tuple(ends)))
    else:
        return _get_events_from_graph_step(starts, list(ends), time_limit=time_limit, path_limit=path_limit)[0]


@cache
def _get_events_from_graph_step_cache(starts, ends):
    '''Note: path and time limit cannot be inputs to the function otherwise caching does not work'''
    starts = list(starts)
    ends = list(ends)

    if len(starts) == 0 or len(ends) == 0:
        return [[]]

    if len(starts) == 1:
        # reached the bottom of the recursion
        assert starts[0][1] == len(ends)
        bottom_start_ends = [[(starts[0][0], e) for e in ends]]
        # log_debug(logger, f'_get_events_from_graph_step_cache RETURN BOTTOM level {len(starts)}: start {starts} and ends {ends} -> {bottom_start_ends}')
        return bottom_start_ends
    
    s = starts[0]
    new_starts = starts[1:].copy()
    res = []
    for chosen_ends in set(itertools.combinations([e for e in ends if e != s[0]], s[1])):
        new_ends = ends.copy()
        for e in chosen_ends:
            new_ends.remove(e)
        # log_debug(logger, f'_get_events_from_graph_step_cache: level {len(starts)} New function call: start {s} and ends {chosen_ends}. new starts = {new_starts}, new ends = {new_ends}')
        cur_res = _get_events_from_graph_step_cache(tuple(new_starts), tuple(new_ends))
        res.extend([[(s[0], e) for e in chosen_ends] + r for r in cur_res])

    # logger.debug(f'_get_events_from_graph_step_cache RETURN for level {len(starts)}: (starts = {starts},  ends = {ends}) -> {res}')
    return res


def _get_events_from_graph_step(starts, ends, i=0, res=None, cur_res=None, time_start=None,
                                block_same_start_end=True, time_limit=2*60, path_limit=None):
    
    if not block_same_start_end:
        raise NotImplementedError('block_same_start_end=False not implemented')

    if time_start is None:
        time_start = time()
    if res is None:
        res = []
    if cur_res is None:
        cur_res = []

    if path_limit is not None and len(res) >= path_limit:
        logger.warning(f"Reached path limit: {path_limit}")
        return None, None
    if time_limit is not None and (time() - time_start) > time_limit:
        logger.warning(f"Reached time limit: {time_limit}")
        return None, None

    if len(starts) == 0 or len(ends) == 0:
        res.append(cur_res)
        return res, cur_res
    
    s = starts[0]
    new_starts = starts[1:].copy()
    for chosen_ends in set(itertools.combinations([e for e in ends if e != s[0]], s[1])):
        new_ends = ends.copy()
        # if s[0] in chosen_ends:
        #     continue
        for e in chosen_ends:
            new_ends.remove(e)
            cur_res.append((s[0], e))
        res, cur_res = _get_events_from_graph_step(
            new_starts, new_ends, i+1, res, cur_res, time_start=time_start, time_limit=time_limit, 
            path_limit=path_limit)
        if res is None:
            return None, None
        cur_res = cur_res[:len(cur_res)-s[1]]

    return res, cur_res


def pad_profile(profile, pad_values=(1, 1)):
    # This is faster than np.pad
    return np.append(pad_values[0], np.append(profile, pad_values[1]))


def adjust_profile_for_loh(profile, wgd=False, total_cn=False):
    pad_value = (2 if wgd else 1) * (2 if total_cn else 1)
    profile = profile.copy()
    profile = np.pad(profile.copy(), 1, mode='constant', constant_values=pad_value)

    loh_pos = profile == 0
    # loh_pos[0] = False
    # loh_pos[-1] = False
    loh_adjust = np.stack([np.roll(profile, 1)[loh_pos],
                           np.roll(profile, -1)[loh_pos]], axis=1).min(axis=1) - pad_value
    profile[loh_pos] = np.clip(loh_adjust, a_min=(-1 if wgd else 0), a_max=None)
    # profile[loh_pos] = np.clip(loh_adjust, a_min=(-1 if wgd else 0) * (2 if total_cn else 1), a_max=None)
    profile = profile[1:-1]

    return profile


def get_starts_and_ends(profile, prior_profile=None, loh_adjust=True, wgd=False, total_cn=False):
    breakpoints = np.arange(len(profile)+1)

    if prior_profile is None:
        prior_profile = np.ones_like(profile) * (2 if total_cn else 1)

    if loh_adjust and 0 in profile:
        profile = adjust_profile_for_loh(profile.copy(), wgd=wgd, total_cn=total_cn)

    cn_changes = pad_profile(profile - prior_profile, (0, 0))

    starts = np.clip(np.diff(cn_changes), a_min=0, a_max=None)
    starts = np.repeat(breakpoints, starts)
    ends = np.clip(np.diff(cn_changes[::-1])[::-1], a_min=0, a_max=None)
    ends = np.repeat(breakpoints, ends)

    return starts, ends


def create_random_start_end_pairs(starts, ends, n_paths, pre_selected_events=None):
    if pre_selected_events is not None:

        n_events = len(starts)
        starts, ends = starts.copy(), ends.copy()

        pre_selected_starts, pre_selected_ends = np.array(list(zip(*pre_selected_events)))
        vals, counts = np.unique(pre_selected_starts, return_counts=True)
        for val, count in zip(vals, counts):
            starts = np.delete(starts, np.where(starts==val)[0][:count])
        assert len(starts) == n_events - len(pre_selected_starts), f'pre_selected_starts not in starts: {pre_selected_starts}'

        vals, counts = np.unique(pre_selected_ends, return_counts=True)
        for val, count in zip(vals, counts):
            ends = np.delete(ends, np.where(ends==val)[0][:count])
        assert len(ends) == n_events - len(pre_selected_ends), f'pre_selected_ends not in ends: {pre_selected_ends}'
    else:
        pre_selected_events = []

    random_ends = ends[np.argsort(np_rng().rand(n_paths, len(ends)), axis=1)]
    events = [pre_selected_events + [(s, e) for s, e in zip(starts, cur_ends)] for cur_ends in random_ends]

    return events


def get_events_diff_from_coords(event_coords, profile, lexsort_diffs=False,
                                filter_missed_lohs=True, fail_if_empty=True):
    """
    Treats event_coords as breakpoints and not segments! So for a profile of length N, there are
    N+1 breakpoints.
    """
    if event_coords is None:
        return None
    
    loh_pos = profile == 0

    diffs = []
    for cur_solution in event_coords:
        cur_diff = np.zeros((len(cur_solution), len(profile)), dtype=int)
        for n, event in enumerate(cur_solution):
            if event[0] <= event[1]: # gain
                cur_diff[n, event[0]:event[1]] = 1
            else: # loss
                cur_diff[n, event[1]:event[0]] = -1

        if lexsort_diffs:
            cur_diff = cur_diff[np.lexsort(cur_diff.T[::-1])][::-1]

        # filter out that don't delete LOHs
        if filter_missed_lohs and not (cur_diff[:, loh_pos] == -1).any(axis=0).all():
            continue
        diffs.append(cur_diff)
    if fail_if_empty and len(diffs) == 0:
        raise ValueError('No diffs found')

    return diffs


def get_events_diff_from_coords_wgd(event_coords, profile, lexsort_diffs=True, 
                                    filter_missed_lohs=False, fail_if_empty=True):
    diffs_pre = get_events_diff_from_coords(
        [path[0] for path in event_coords], profile, lexsort_diffs=lexsort_diffs,
        filter_missed_lohs=filter_missed_lohs, fail_if_empty=fail_if_empty)
    diffs_post = get_events_diff_from_coords(
        [path[1] for path in event_coords], profile, lexsort_diffs=lexsort_diffs,
        filter_missed_lohs=filter_missed_lohs, fail_if_empty=fail_if_empty)
    assert filter_missed_lohs or len(diffs_pre) == len(diffs_post), (len(diffs_pre), len(diffs_post))
    diffs = list(zip(diffs_pre, diffs_post))

    return diffs


class CpSolverSolutionArray(cp_model.CpSolverSolutionCallback):
    """Print intermediate solutions."""

    def __init__(self, variables, silent=False):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__variables = variables
        self.__solution_count = 0
        self.silent = silent
        self.all_solutions = []

    def on_solution_callback(self):
        self.__solution_count += 1
        self.all_solutions.append([self.Value(v) for v in self.__variables])
        if not self.silent:
            for v in self.__variables:
                print(f"{v}={self.Value(v)}", end=" ")
            print()

    def solution_count(self):
        return self.__solution_count


def loh_filters_for_graph_result_diffs(diffs, profile, single_time_limit=None, total_cn=False,
                                       return_all_solutions=True, shuffle_diffs=True):

    if len(diffs) == 0:
        logger.warning('Empty diffs passed into loh_filters_for_graph_result_diffs. Returning empty list.')
        return []
    
    all_final_diffs = []
    log_debug(logger, f"Performing LOH filters (returning {'all solutions' if return_all_solutions else 'a single solution'})")
    log_debug(logger, f"Profile: {profile}")
    log_debug(logger, f"number of initial diffs: {len(diffs)}")

    n_events = len(diffs[0])
    lohs = np.array(profile) == 0

    for n, cur_diff in enumerate(diffs):

        cur_diff = cur_diff.copy()
        if shuffle_diffs:
            cur_diff = cur_diff[np_rng().choice(np.arange(len(cur_diff)), len(cur_diff), replace=False)]

        model = cp_model.CpModel()
        n_events = len(cur_diff)
        order = [model.NewIntVar(0, n_events - 1, f"event_{i}") for i in range(n_events)]
        model.AddAllDifferent(order);

        unique_diff_cols = set()
        for diff_col, is_loh in zip(cur_diff.T, lohs):
            cur_hash = (is_loh, tuple(diff_col))
            if cur_hash in unique_diff_cols:
                continue
            else:
                unique_diff_cols.add(cur_hash)

            # loh/non-loh is guaranteed by all possible orders so no need to include
            if (is_loh and (((diff_col == 1).sum() + (1 if total_cn else 0)) < (diff_col == -1).sum()) or
                (not is_loh and ((diff_col == -1).sum() < (2 if total_cn else 1)))):
                continue

            losses = np.where(diff_col == -1)[0]
            gains = np.where(diff_col == 1)[0]
            n_losses = len(losses)
            n_gains = len(gains)

            # logger.debug("LOH" if is_loh else "non-LOH") # should be commented out to save time
            # logger.debug(f'n gains: {n_gains} / n losses: {n_losses}') # should be commented out to save time
            # for LOH make sure that at least 1 loss is before the gains (2 losses in case of total CN)
            if is_loh:
                n_losses_early = 2 if total_cn else 1
                n_losses_gains_free = n_losses - n_losses_early
                n_gains_late = n_gains - n_losses_gains_free
                # Here is where the OR statement starts!
                cur_or = []
                n = 0
                for fixed_loss_i, fixed_loss_is in enumerate(itertools.combinations(range(n_losses), n_losses_early)):
                    fixed_losses = [losses[i] for i in fixed_loss_is]
                    free_losses = [l for l in losses if l not in fixed_losses]
                    for fixed_gains_i in itertools.combinations(range(n_gains), n_gains_late):
                        fixed_gains = [gains[i] for i in fixed_gains_i]
                        free_gains_base = [gains[i] for i in range(n_gains) if i not in fixed_gains_i]
                        for free_gains in itertools.permutations(free_gains_base):
                            cur_bool_or = model.NewBoolVar("loh_bool:" + str(fixed_loss_i) + ":" + '-'.join([str(i) for i in free_gains]))
                            # logger.debug(cur_bool_or) # should be commented out to save time
                            cur_and = []
                            # First add all the fixed ones (i.e. gains that have to happen after the fixed loss)
                            for i in range(n_gains_late):
                                cur_bool_early_loss = model.NewBoolVar("loh_bool:{n}")
                                n += 1 
                                # logger.debug(f"{order[fixed_loss]} < {order[fixed_gains[i]]}") # should be commented out to save time
                                for fixed_loss in fixed_losses:
                                    model.Add(order[fixed_loss] < order[fixed_gains[i]]).OnlyEnforceIf(cur_bool_early_loss)
                                cur_and.append(cur_bool_early_loss)

                            # Now add the free gains are either behind the fixed loss are appear before but paired with a loss
                            # logger.debug('---') # should be commented out to save time
                            for i in range(n_losses_gains_free):
                                cur_bool_pair = model.NewBoolVar("loh_bool:{n}")
                                cur_bool_before_after = model.NewBoolVar("loh_bool_before_after:{n}")
                                n += 1 
                                # logger.debug(f"{order[free_gains[i]]} > {order[fixed_loss]} OR ({order[free_gains[i]]} < {order[fixed_loss]} AND {order[free_losses[i]]} < {order[fixed_loss]})") # should be commented out to save time
                                for fixed_loss in fixed_losses:
                                    model.Add(order[free_gains[i]] > order[fixed_loss]).OnlyEnforceIf([cur_bool_pair, cur_bool_before_after])
                                    model.Add(order[free_gains[i]] < order[fixed_loss]).OnlyEnforceIf([cur_bool_pair, cur_bool_before_after.Not()])
                                    model.Add(order[free_losses[i]] < order[fixed_loss]).OnlyEnforceIf([cur_bool_pair, cur_bool_before_after.Not()])
                                cur_and.append(cur_bool_pair)

                        # logger.debug('#######') # should be commented out to save time
                        model.AddBoolAnd(cur_and).OnlyEnforceIf(cur_bool_or)
                        cur_or.append(cur_bool_or)
                # logger.debug(f'len cur or: {len(cur_or)}') # should be commented out to save time
                model.AddBoolOr(cur_or)

                # logger.debug('/////////////') # should be commented out to save time
                        
            # for non-LOH make sure that every loss that could create a LOH (post-wgd value - nr of losses + 1) is countered by a gain
            # in case of total_cn, one extra loss can actually be free
            else:
                n_gains_early = n_losses - (1 if total_cn else 0)
                # Here is where the OR statement starts!
                cur_or = []
                n = 0
                for cur_losses in itertools.permutations(losses):
                    for cur_gains in itertools.combinations(gains, n_gains_early):
                        cur_bool_or = model.NewBoolVar("non-loh_bool:" + '-'.join([str(i) for i in cur_losses]) + ":" + '-'.join([str(i) for i in cur_gains]))
                        # logger.debug(cur_bool_or) # should be commented out to save time
                        cur_and = []
                        for i in range(n_gains_early):
                            cur_bool_early_gain = model.NewBoolVar("non-loh_bool:{n}")
                            n += 1 
                            # logger.debug(f"{order[cur_losses[i]]} > {order[cur_gains[i]]}") # should be commented out to save time
                            model.Add(order[cur_losses[i]] > order[cur_gains[i]]).OnlyEnforceIf(cur_bool_early_gain)
                            cur_and.append(cur_bool_early_gain)

                        # logger.debug('#######') # should be commented out to save time
                        model.AddBoolAnd(cur_and).OnlyEnforceIf(cur_bool_or)
                        cur_or.append(cur_bool_or)
                # logger.debug(f'len cur or: {len(cur_or)}') # should be commented out to save time
                model.AddBoolOr(cur_or)
                    
                # logger.debug('/////////////') # should be commented out to save time

        solver = cp_model.CpSolver()
        _make_solver_deterministic(solver)
        if single_time_limit is not None:
            solver.parameters.max_time_in_seconds = single_time_limit
        solver_solutions = CpSolverSolutionArray(order, silent=True)
        if return_all_solutions:
            solver.parameters.enumerate_all_solutions = True
        else:
            solver.parameters.enumerate_all_solutions = False
            solver.solution_limit = 1
        status = solver.Solve(model, solver_solutions)

        if len(solver_solutions.all_solutions) == 0:
            logger.debug(f'no loh solution found for solution {n}')
            continue

        cp_solutions = np.array(solver_solutions.all_solutions)
        if not return_all_solutions:
            cp_solutions = cp_solutions[np_rng().choice(range(len(cp_solutions)), 1)]

        unique_results = set()
        for cur_solution_ in cp_solutions:
            # this is required to transform the order into valid indices
            cur_solution = np.argsort(cur_solution_)

            # make sure that the profile is fulfilled
            assert not (profile==0).any() or (np.cumsum(cur_diff[cur_solution], axis=0)[:, profile==0] == (-2 if total_cn else -1)).any()
            assert (np.cumsum(cur_diff[cur_solution], axis=0)[:, profile!=0] != (-2 if total_cn else -1)).all()

            # this selects the first event after the LOH event
            cur_diff_mod = cur_diff[cur_solution].copy()
            loh_happend_mask = np.zeros_like(cur_diff_mod, dtype=bool)
            loh_happend_mask[:, profile==0] = np.roll(np.pad(np.cumsum(np.cumsum(cur_diff[cur_solution], axis=0)[:, profile==0] == (-2 if total_cn else -1), axis=0).astype(bool),
                                            ((0, 1), (0, 0)), constant_values=False), 1, axis=0)[:-1]
            cur_diff_mod[loh_happend_mask] = 0   

            cur_hash = tuple(sorted([''.join(map(str, x)) for x in cur_diff_mod]))
            if cur_hash not in unique_results:
                all_final_diffs.append(cur_diff_mod)
                unique_results.add(cur_hash)

        log_debug(logger, f"for solution {n} found {solver_solutions.solution_count()} loh-compatible solutions of which {len(unique_results)} were unique")   

    log_debug(logger, f"number of final diffs: {len(all_final_diffs)}")

    return all_final_diffs


def loh_filters_for_graph_result_diffs_wgd(
        diffs, profile, return_all_solutions=True, total_cn=False,
        shuffle_diffs=True):
    log_debug(logger, f"Performing LOH filters (returning {'all solutions' if return_all_solutions else 'a single solution'})")
    log_debug(logger, f"number of initial solutions: {len(diffs)}")

    all_final_diffs = []

    # needed in case profile is a tuple
    profile = np.array(profile)

    for n, cur_diff in enumerate(diffs):
        if shuffle_diffs:
            cur_diff = [cur_diff[0][np_rng().choice(np.arange(len(cur_diff[0])), len(cur_diff[0]), replace=False)],
                        cur_diff[1][np_rng().choice(np.arange(len(cur_diff[1])), len(cur_diff[1]), replace=False)]]

        model = cp_model.CpModel()
        n_pre = len(cur_diff[0])
        n_post = len(cur_diff[1])
        order_pre = [model.NewIntVar(0, n_pre - 1, f"event_{i}") for i in range(n_pre)]
        order_post = [model.NewIntVar(0, n_post - 1, f"event_{i}") for i in range(n_post)]
        if n_pre > 0:
            model.AddAllDifferent(order_pre);
        if n_post > 0:
            model.AddAllDifferent(order_post);

        # this neglects LOHs that happened pre-WGD but that's okay because if it's a LOH position then 
        # loh_fulfilled is True anyways and if it's not a LOH position then the value will actually be != 0
        post_wgd_state = 2*((2 if total_cn else 1)+cur_diff[0].sum(axis=0))
        post_wgd_state[post_wgd_state<0] = 0

        for diff_col_pre, diff_col_post, profile_value, post_wgd_value in zip(cur_diff[0].T, cur_diff[1].T, profile, post_wgd_state):
            is_loh = profile_value == 0
            target_elements_pre = []
            target_elements_post = []
            loh_fulfilled = []
            for i in range(n_pre):
                target_element = model.NewIntVar(-1, 1, f"target_{i}")
                model.AddElement(order_pre[i], list(diff_col_pre), target_element)
                target_elements_pre.append(target_element)
                if is_loh:
                    current_loh_fulfilled = model.NewBoolVar(f"loh_fulfilled_{i}")
                    model.Add(sum(target_elements_pre) == (-2 if total_cn else -1)).OnlyEnforceIf(current_loh_fulfilled)
                    model.Add(sum(target_elements_pre) != (-2 if total_cn else -1)).OnlyEnforceIf(current_loh_fulfilled.Not())
                    loh_fulfilled.append(current_loh_fulfilled)
                else:
                    model.Add(sum(target_elements_pre) > (-2 if total_cn else -1))
            for i in range(n_post):
                target_element = model.NewIntVar(-1, 1, f"target_{i}")
                model.AddElement(order_post[i], list(diff_col_post), target_element)
                target_elements_post.append(target_element)
                if is_loh:
                    current_loh_fulfilled = model.NewBoolVar(f"loh_fulfilled_{i}")
                    model.Add(post_wgd_value + sum(target_elements_post) == 0).OnlyEnforceIf(current_loh_fulfilled)
                    model.Add(post_wgd_value + sum(target_elements_post) != 0).OnlyEnforceIf(current_loh_fulfilled.Not())
                    loh_fulfilled.append(current_loh_fulfilled)
                else:
                    model.Add((post_wgd_value + sum(target_elements_post)) > 0)
            if is_loh:
                model.AddAtLeastOne(loh_fulfilled)

        solver = cp_model.CpSolver()
        _make_solver_deterministic(solver)
        # if single_time_limit is not None:
        #     solver.parameters.max_time_in_seconds = single_time_limit
        solver_solutions = CpSolverSolutionArray(order_pre + order_post, silent=True)
        if return_all_solutions:
            solver.parameters.enumerate_all_solutions = True
        else:
            solver.parameters.enumerate_all_solutions = False
            solver.solution_limit = 1
        status = solver.Solve(model, solver_solutions)
        cp_solutions = np.array(solver_solutions.all_solutions)

        unique_results = set()
        for cur_solution in cp_solutions:

            cur_diff_pre_solved = cur_diff[0][cur_solution[:n_pre]]
            cur_diff_post_solved = cur_diff[1][cur_solution[n_pre:]]
            post_wgd_state = 2*((2 if total_cn else 1)+np.sum(cur_diff_pre_solved, axis=0))
            cur_diff_mod_pre = cur_diff_pre_solved.copy()
            cur_diff_mod_post = cur_diff_post_solved.copy()

            # this selects the first event after the LOH event
            loh_happend_mask_pre = np.zeros_like(cur_diff_mod_pre, dtype=bool)
            loh_happend_mask_pre[:, profile==0] = np.roll(np.pad(np.cumsum(np.cumsum(cur_diff_pre_solved, axis=0)[:, profile==0] == (-2 if total_cn else -1), axis=0).astype(bool),
                                                    ((0, 1), (0, 0)), constant_values=False),
                                                        1, axis=0)[:-1]
            cur_diff_mod_pre[loh_happend_mask_pre] = 0

            # this selects the first event after the LOH event
            loh_happend_mask_post = np.zeros_like(cur_diff_mod_post, dtype=bool)
            loh_happend_mask_post[:, profile==0] = np.roll(np.pad(np.cumsum(post_wgd_state[profile==0] + np.cumsum(cur_diff_post_solved, axis=0)[:, profile==0] <= 0, axis=0).astype(bool),
                                                    ((0, 1), (0, 0)), constant_values=False),
                                                    1, axis=0)[:-1]
            cur_diff_mod_post[(post_wgd_state == 0) | loh_happend_mask_post] = 0

            cur_hash = (tuple(sorted([''.join(map(str, x)) for x in cur_diff_mod_pre])),
                            tuple(sorted([''.join(map(str, x)) for x in cur_diff_mod_post])))
            if cur_hash not in unique_results:
                all_final_diffs.append((cur_diff_mod_pre, cur_diff_mod_post))
                unique_results.add(cur_hash)

        log_debug(logger, f"For solution {n}: found {solver_solutions.solution_count()} loh-compatible solutions of which {len(unique_results)} were unique")   

    return all_final_diffs


def check_loh_for_full_paths(full_path, solutions=None):
    if solutions is not None and isinstance(solutions, int):
        solutions = [solutions]
    if full_path.is_wgd:
        # cur_diffs = (
        #     [[np.stack([np.fromiter(full_path.events[event].diff, dtype=int) * (1 if full_path.events[event].is_gain else -1)
        #         for event, count in sol.items() for _ in range(count) if full_path.events[event].wgd == 'pre']),
        #     np.stack([np.fromiter(full_path.events[event].diff, dtype=int) * (1 if full_path.events[event].is_gain else -1)
        #         for event, count in sol.items() for _ in range(count) if full_path.events[event].wgd == 'post'])]
        #     for sol_i, sol in enumerate(full_path.solutions) if solutions is None or sol_i in solutions])
        cur_diffs = (
            [[[np.fromiter(full_path.events[event].diff, dtype=int) * (1 if full_path.events[event].is_gain else -1)
                for event, count in sol.items() for _ in range(count) if full_path.events[event].wgd == 'pre'],
              [np.fromiter(full_path.events[event].diff, dtype=int) * (1 if full_path.events[event].is_gain else -1)
                for event, count in sol.items() for _ in range(count) if full_path.events[event].wgd == 'post']]
            for sol_i, sol in enumerate(full_path.solutions) if solutions is None or sol_i in solutions]
            )
        cur_diffs = [
            [np.stack(cur_diff_pair[0]) if len(cur_diff_pair[0]) > 0 else np.empty((0, len(full_path.cn_profile)), dtype=np.int64),
             np.stack(cur_diff_pair[1]) if len(cur_diff_pair[1]) > 0 else np.empty((0, len(full_path.cn_profile)), dtype=np.int64)]
            for cur_diff_pair in cur_diffs
        ]
        # log_debug(logger, cur_diffs)
        filtered_diffs = loh_filters_for_graph_result_diffs_wgd(
            cur_diffs, full_path.cn_profile, return_all_solutions=False)
    else:
        cur_diffs = (
            [np.stack([np.fromiter(full_path.events[event].diff, dtype=int) * (1 if full_path.events[event].is_gain else -1)
            for event, count in sol.items() for _ in range(count)])
            for sol_i, sol in enumerate(full_path.solutions) if solutions is None or sol_i in solutions])

        # log_debug(logger, cur_diffs)
        filtered_diffs = loh_filters_for_graph_result_diffs(cur_diffs, full_path.cn_profile, return_all_solutions=False)
    return filtered_diffs


def diff_to_start_end(events_df):
    '''Takes events_df and return start/end tuples for all events per chain_nr'''
    all_events = []
    for chain_nr in events_df['chain_nr'].unique():
        cur_events = events_df.query('chain_nr == @chain_nr')

        events = np.stack([
            cur_events['diff'].map(lambda x: x.find('1')).values,
            cur_events['diff'].map(lambda x: x.rfind('1')).values+1,
        ]).T
        events[cur_events['type'].values == 'loss'] = events[cur_events['type'].values == 'loss'][:, ::-1]
        repeats = np.min(np.stack([
            cur_events['type'].map({'gain': 2, 'loss': 2}).values,
            # cur_events['type'].map({'gain': 2, 'loss': 1}).values,
            cur_events['wgd'].map({'pre': 2, 'post': 1}).values
        ]), axis=0)
        events = np.repeat(events, repeats, axis=0)

        all_events.append(events)
    return all_events


def check_pre_post_wgd_loh_viability_simple(which, loh_pos, pre_wgd_diffs_sum=None, pre_wgd_diff=None,
                                            post_wgd_diffs=None, post_wgd_state=None, total_cn=False):
    '''
    Checks that events are LOH-valid.
    Checks that paths do not create LOHs in non-LOH positions and for post-WGD that all paths have
    at least one deletion in LOH positions. I don't know if the check for unwanted LOHs in non-LOH
    positions for post makes sense but it also doesn't hurt.
    
    Note that this does not check if the events can be placed in a 
    reasonable order. For this cp_model from ortools.sat.python has to be used.        
    '''

    if which == 'pre':
        # check that non-LOH positions are not deleted
        loh_valid = [all(diff_sum[~loh_pos] >= (-1 if total_cn else 0)) for diff_sum in pre_wgd_diffs_sum]
    elif which == 'post':
        remaining_loh_pos = loh_pos * (pre_wgd_diff != -1).all(axis=0)
        total_diffs = [(post_wgd_state + diff) for diff in post_wgd_diffs] # note that post_wgd_state already takes care of total_cn
        total_diffs_sum = [diff.sum(axis=0) for diff in total_diffs]
        # has to be >= 2 because post-WGD state can either be 0 or >=2 but not 1.
        loh_valid = [all(total_diff_sum[~loh_pos] >= 0) and ((post_diff[:, remaining_loh_pos] == -1).sum(axis=0) >= (4 if total_cn else 2)).all() if remaining_loh_pos.any() else True 
                    for total_diff_sum, post_diff in zip(total_diffs_sum, post_wgd_diffs)]
    else:
        raise ValueError(f'which must be pre or post, not {which}')
    return loh_valid


def get_events_for_cur_start_ends_wgd(starts, ends, n_events, cn_profile, total_cn=False, use_cache=True):

    starts_counter = Counter(starts)
    ends_counter = Counter(ends)

    logger.debug('Start of get_events_for_cur_start_ends_wgd')
    logger.debug(f'Current starts and ends (after post-WGD SV events are removed) are: {starts} / {ends}')

    loh_pos = cn_profile == 0
    n_required_pre_wgd_events = len(starts) - int(n_events)
    n_post_wgd_events = int(n_events) - n_required_pre_wgd_events
    logger.debug(f'Requiring {n_required_pre_wgd_events} pre-WGD events (dist = {n_events} and {len(starts)} starts/ends and {n_post_wgd_events} post-WGD events)')

    pre_wgd_starts = Counter({s: c//2 for s, c in starts_counter.items() if c>=2})
    pre_wgd_ends = Counter({s: c//2 for s, c in ends_counter.items() if c>=2})

    paths = []
    # combinations is used here because there might be more potential starts/ends than required events
    for cur_pre_start, cur_pre_end in itertools.product(
            set(itertools.combinations(pre_wgd_starts.elements(), r=n_required_pre_wgd_events)),
            set(itertools.combinations(pre_wgd_ends.elements(), r=n_required_pre_wgd_events))):

        if len(cur_pre_start) > 0:
            pre_wgd_paths = get_events_from_graph_step(
                cur_pre_start, cur_pre_end, use_cache=use_cache)
        else:
            pre_wgd_paths = [[]]
            
        if len(pre_wgd_paths) == 0 : # all events are post-WGD
            pre_wgd_diffs = [np.zeros_like(cn_profile)]
            post_wgd_states = [2*(np.ones_like(cn_profile) * (2 if total_cn else 1))]
            pre_wgd_loh_valid = [True]
        else:
            pre_wgd_diffs = get_events_diff_from_coords(pre_wgd_paths, cn_profile, filter_missed_lohs=False, fail_if_empty=False)
            pre_wgd_diffs_sum = [diff.sum(axis=0) for diff in pre_wgd_diffs]
            post_wgd_states = [2*(diff.sum(axis=0) + (2 if total_cn else 1)) for diff in pre_wgd_diffs]
            pre_wgd_loh_valid = check_pre_post_wgd_loh_viability_simple(
                'pre', loh_pos, pre_wgd_diffs_sum=pre_wgd_diffs_sum, total_cn=total_cn)

        for pre_wgd_path, pre_wgd_diff, post_wgd_state, valid in zip(pre_wgd_paths, pre_wgd_diffs, post_wgd_states, pre_wgd_loh_valid):
            if not valid:
                continue
            cur_post_ends = list(ends.copy())
            cur_post_starts = list(starts.copy())
            if len(pre_wgd_path) > 0:
                [cur_post_ends.remove(event[1]) for event in pre_wgd_path for _ in range(2)]
                cur_post_starts = list((starts_counter - Counter(2*[event[0] for event in pre_wgd_path])).elements())
            
            post_wgd_paths = get_events_from_graph_step(cur_post_starts, cur_post_ends, use_cache=use_cache)
            # this can happen if a start can only connect to an end that has the same value
            post_wgd_paths = [path for path in post_wgd_paths if len(path) == n_post_wgd_events]

            if len(post_wgd_paths) == 0:
                # in case no valid post-WGD paths are found, continue with the next pre-WGD path
                if len(pre_wgd_path) < n_events:
                    continue
                # if all events are pre-WGD, create this dummy diffs and paths
                post_wgd_diffs = [np.zeros((1, len(cn_profile)))]
                post_wgd_paths = [[]]
            else:
                post_wgd_diffs = get_events_diff_from_coords(post_wgd_paths, cn_profile, lexsort_diffs=True, filter_missed_lohs=False)
    
            post_wgd_valid = check_pre_post_wgd_loh_viability_simple(
                'post', loh_pos, pre_wgd_diff=pre_wgd_diff, post_wgd_diffs=post_wgd_diffs,
                post_wgd_state=post_wgd_state, total_cn=total_cn)
            
            paths.append([(pre_wgd_path, path) for path, valid in zip(post_wgd_paths, post_wgd_valid) if valid])
            paths = [path for path in paths if len(path) > 0] # remove empty paths

    # slightly faster than paths.extend
    paths = sum(paths, [])

    return paths


def get_wgd_bps_to_add(paths, starts, ends, loh=True):
    starts_counter = Counter(starts)
    ends_counter = Counter(ends)

    # first detect all starts and ends that are part of pre-WGD events in every path and exclude them from available starts and ends
    min_required_pre_wgd_starts_ends = [
        reduce(Counter.__and__, all_start_end, default) for all_start_end, default in zip(list(zip(*[[Counter(2*path_start_end) 
                                                                                for path_start_end in list(zip(*path[0] if len(path[0]) > 0 else []))] 
                                                                                for path in paths])), [starts_counter, ends_counter])]
    min_required_pre_wgd_starts, min_required_pre_wgd_ends = min_required_pre_wgd_starts_ends if len(min_required_pre_wgd_starts_ends) == 2 else (Counter(), Counter())
    available_starts = starts_counter - min_required_pre_wgd_starts
    available_ends = ends_counter - min_required_pre_wgd_ends

    # Next for available starts/ends with duplicity >= 2, find all ends/starts with odd duplicity.
    # If there is no LOH, only select those that are larger/smaller than them (so that they form a gain together)
    bps_to_add = set()
    if loh:
        for a, b in zip((available_starts, available_ends), (available_ends, available_starts)):
            doubled_parter_bps = np.array([k for k, v in a.items() if v >= 2])
            potential_ks = np.array([k for k, v in b.items() if v%2 == 1]) # odd numbered
            bps_to_add |= set(potential_ks[[any(k != doubled_parter_bps) for k in potential_ks]])
    else:
        for a, b, which in zip((available_starts, available_ends), (available_ends, available_starts), ('larger', 'smaller')):
            doubled_parter_bps = np.array([k for k, v in a.items() if v >= 2])
            potential_ks = np.array([k for k, v in b.items() if v%2 == 1]) # odd numbered
            bps_to_add |= set(potential_ks[[any(k > doubled_parter_bps if which == 'larger' else k < doubled_parter_bps) for k in potential_ks]])

    return list(bps_to_add)


def get_new_wgd_paths_by_adding_bps(starts, ends, cn_profile, n_events, bps_to_add, use_cache=True, total_cn=False):
    new_paths = []
    i = 0
    while i < len(bps_to_add):
        cur_bp = bps_to_add[i]
        i += 1
        if not hasattr(cur_bp, '__len__'):
            cur_bp = [cur_bp]

        cur_starts = np.append(starts, list(cur_bp))
        cur_ends = np.append(ends, list(cur_bp))
        cur_paths = get_events_for_cur_start_ends_wgd(
            cur_starts, cur_ends, n_events, cn_profile, use_cache=use_cache, total_cn=total_cn)
        if len(cur_paths) == 0:
            log_debug(logger, f'added bp {cur_bp} but not used')
            continue
        new_paths.extend(cur_paths)

        cur_bps_to_add = get_wgd_bps_to_add(cur_paths, cur_starts, cur_ends, loh=(0 in cn_profile))
        old_len_bps = len(bps_to_add) # just for debugging, might delete later
        if len(cur_bps_to_add) > 0:
            new_bps = set([tuple(sorted([*cur_bp, x])) for x in cur_bps_to_add])
            bps_to_add.extend([x for x in new_bps if x not in set(bps_to_add)])

        log_debug(logger, f'for added bp(s) {cur_bp} found {len(cur_paths)} new paths and {len(cur_bps_to_add)} potential bps to add ({cur_bps_to_add}) of which {len(bps_to_add)-old_len_bps} were new')
        # log_debug(logger, f'New paths: {cur_paths}')

    return new_paths


def get_sv_selected_paths_wgd(paths, cur_sv_data, cur_chrom_segments, sv_matching_threshold=sv_matching_threshold):

    all_starts = np.unique([event[0] for path in paths for pre_post in path for event in pre_post if len(pre_post) > 0])
    all_ends = np.unique([event[1] for path in paths for pre_post in path for event in pre_post if len(pre_post) > 0])

    sv_selected_events, _, _ = connect_start_ends_using_svs(
        cur_chrom_segments, cur_sv_data, all_starts, all_ends, sv_matching_threshold=sv_matching_threshold)
    if sv_selected_events is None:
        return paths, None
    sv_selected_events_tuples = [tuple(x) for x in sv_selected_events]
    paths_flat = [sorted(2*x[0] + x[1]) for x in paths]
    sv_overlap = np.array([[x in path_flat for x in sv_selected_events_tuples] for path_flat in paths_flat])
    sv_overlap_sum = sv_overlap[:, sv_overlap.any(axis=0)].sum(axis=1)
    sv_selection = sv_overlap_sum == sv_overlap_sum.max()
    sv_selected_paths = [path for path, valid in zip(paths, sv_selection) if valid]
    sv_selected_events = [event for event, valid in zip(sv_selected_events, sv_overlap.any(axis=0)) if valid]

    return sv_selected_paths, sv_selected_events
    

def is_same_paths(test, ground_truth, can_be_subset=False, wgd=False):
    '''Tests that test paths are the same as ground_truth paths'''

    if wgd:
        # Double pre-WGD events
        test = [(2*x[0] + x[1]) for x in test]

    test = [sorted(sublist) for sublist in test]
    ground_truth = [sorted(sublist) for sublist in ground_truth]
    if can_be_subset:
        return all([x in test for x in ground_truth])
    else:
        return sorted(test) == sorted(ground_truth)


def get_wgd_single_solution(cn_profile, max_n_iterations=25, total_cn=False):
    '''used in mcmc to get a starting solution'''

    # detect potential early LOH events and remove them
    lohs = np.where(cn_profile == 0)[0]
    n_events = max(1, int(float(fstlib.score(T_forced_WGD, diploid_fsa_total_cn if total_cn else diploid_fsa, fsa_from_string(''.join(cn_profile.astype(str)))))) - 1)
    if total_cn and (cn_profile==0).all():
        n_events = 2
    if n_events > config['params']['dist_limit']:
        return None
    cn_profile_pad = np.pad(cn_profile, (1, 1), constant_values=2)
    cn_profile = cn_profile.copy()
    early_loh_events = []
    for i in lohs:
        cn_profile_ = cn_profile.copy()
        cn_profile_[i] = min(cn_profile_pad[i+1-1], cn_profile_pad[i+1+1])
        cur_n_events = max(1, int(float(fstlib.score(
            T_forced_WGD, diploid_fsa_total_cn if total_cn else diploid_fsa, fsa_from_string(''.join(cn_profile_.astype(str)))))) - 1)
        if total_cn:
            # for total CN it can either have one (eg "101") or two ("303") pre-WGD, single-segment losses
            if cn_profile_[i] == 1 and cur_n_events + 1 == n_events:
                cn_profile[i] = cn_profile_[i]
                n_events -= 1
                early_loh_events.append((i+1, i))
            elif cn_profile_[i] > 1 and cur_n_events + 2 == n_events:
                cn_profile[i] = cn_profile_[i]
                n_events -= 2
                early_loh_events.extend([(i+1, i), (i+1, i)])
        else:
            if cur_n_events + 1 == n_events:
                cn_profile[i] = cn_profile_[i]
                n_events -= 1
                early_loh_events.append((i+1, i))
    if len(early_loh_events) > 0:
        logger.debug(f'Found simple LOH events. Adjusted profile: {cn_profile}. Added {len(early_loh_events)} early LOH events: {early_loh_events}')
    if 0 not in cn_profile:
        # either non-LOH or all LOHs are simple, i.e. they can be solved with the above method
        select_first_solution = True
    else:
        select_first_solution = False

    all_starts, all_ends = get_starts_and_ends_for_wgd(cn_profile, adjust_loh_wgd=True, total_cn=total_cn)
    # reverse them so that the most complex LOH adjustments are done first
    all_starts, all_ends = all_starts[::-1], all_ends[::-1]
    logger.debug(f'{len(all_starts)} starts/ends combinations due to LOH adjust')
    for iteration_loh_adjust, (starts, ends) in enumerate(zip(all_starts, all_ends)):
        if iteration_loh_adjust > max_n_iterations*10:
            break
        base_paths = __get_events_for_cur_start_ends_wgd_single_solution(
            starts.copy(), ends.copy(), n_events, cn_profile,
            select_first_solution=select_first_solution, total_cn=total_cn,
            max_n_iterations=max_n_iterations)
        if base_paths is not None and len(base_paths) == 2:
            assert len(base_paths[0]) + len(base_paths[1]) == n_events, (len(base_paths[0]) + len(base_paths[1]), n_events)
            base_paths = [early_loh_events + base_paths[0], base_paths[1]]
            return base_paths

    return None


def __get_events_for_cur_start_ends_wgd_single_solution(
        starts, ends, n_events, cn_profile, select_first_solution=False,
        max_n_iterations=25, total_cn=False):
    '''modified from get_events_for_cur_start_ends_wgd'''

    starts_counter = Counter(starts)
    ends_counter = Counter(ends)

    logger.debug('Start of get_events_for_cur_start_ends_wgd')
    logger.debug(f'Current starts and ends (after post-WGD SV events are removed) are: {starts} / {ends}')

    loh_pos = cn_profile == 0
    n_required_pre_wgd_events = len(starts) - int(n_events)
    n_post_wgd_events = int(n_events) - n_required_pre_wgd_events
    logger.debug(f'Requiring {n_required_pre_wgd_events} pre-WGD events (dist = {n_events} and {len(starts)} starts/ends and {n_post_wgd_events} post-WGD events)')

    pre_wgd_starts = Counter({s: c//2 for s, c in starts_counter.items() if c>=2})
    pre_wgd_ends = Counter({s: c//2 for s, c in ends_counter.items() if c>=2})

    paths = []
    # combinations is used here because there might be more potential starts/ends than required events
    for iteration_pre_selection, (cur_pre_start, cur_pre_end) in enumerate(itertools.product(
            set(itertools.combinations(pre_wgd_starts.elements(), r=n_required_pre_wgd_events)),
            set(itertools.combinations(pre_wgd_ends.elements(), r=n_required_pre_wgd_events)))):
        # if iteration_pre_selection > max_n_iterations*10:
        if iteration_pre_selection > (max_n_iterations**2 if select_first_solution else max_n_iterations*10):
            break

        log_debug(logger, f'Current pre-WGD start and end: {cur_pre_start} / {cur_pre_end}')
        if len(cur_pre_start) > 0:
            if select_first_solution:
                pre_wgd_paths = [[(s, e) for s, e in zip(cur_pre_start, cur_pre_end)]]
            else:
                pre_wgd_paths = set()
                for iteration_pre_events, cur_ends in enumerate(itertools.permutations(cur_pre_end)):
                    if iteration_pre_events > max_n_iterations:
                        break
                    cur_pre_wgd_path = tuple(sorted([(s, e) for s, e in zip(cur_pre_start, cur_ends)]))
                    if cur_pre_wgd_path in pre_wgd_paths:
                        continue
                    pre_wgd_paths.add(cur_pre_wgd_path)
        else:
            pre_wgd_paths = [[]]
        pre_wgd_paths = [list(path) for path in pre_wgd_paths if len(path) == len(cur_pre_start)]
        logger.debug(f'Created {len(pre_wgd_paths)} valid pre-WGD paths')
            
        if len(pre_wgd_paths) == 0 : # all events are post-WGD
            pre_wgd_diffs = [np.zeros_like(cn_profile)]
            post_wgd_states = [2*(np.ones_like(cn_profile) * (2 if total_cn else 1))]
            pre_wgd_loh_valid = [True]
        else:
            pre_wgd_diffs = get_events_diff_from_coords(pre_wgd_paths, cn_profile, filter_missed_lohs=False, fail_if_empty=False)
            pre_wgd_diffs_sum = [diff.sum(axis=0) for diff in pre_wgd_diffs]
            post_wgd_states = [2*(diff.sum(axis=0) + (2 if total_cn else 1)) for diff in pre_wgd_diffs]
            pre_wgd_loh_valid = check_pre_post_wgd_loh_viability_simple(
                'pre', loh_pos, pre_wgd_diffs_sum=pre_wgd_diffs_sum, total_cn=total_cn)

        log_debug(logger, f'Found {sum(pre_wgd_loh_valid)} valid pre-WGD paths')
        for pre_wgd_path, pre_wgd_diff, post_wgd_state, valid in zip(pre_wgd_paths, pre_wgd_diffs, post_wgd_states, pre_wgd_loh_valid):
            if not valid:
                continue
            log_debug(logger, f'Current pre-WGD path: {pre_wgd_path}')
            cur_post_ends = list(ends.copy())
            cur_post_starts = list(starts.copy())
            if len(pre_wgd_path) > 0:
                [cur_post_ends.remove(event[1]) for event in pre_wgd_path for _ in range(2)]
                cur_post_starts = list((starts_counter - Counter(2*[event[0] for event in pre_wgd_path])).elements())

            if select_first_solution:
                cur_solution = [pre_wgd_path, [(s, e) for s, e in zip(cur_post_starts, cur_post_ends)]]
                return cur_solution
            else:
                post_wgd_paths = set()
                for iteration_post_events, cur_ends in enumerate(itertools.permutations(cur_post_ends)):
                    if iteration_post_events > max_n_iterations:
                        break
                    cur_post_wgd_path = tuple(sorted([(s, e) for s, e in zip(cur_post_starts, cur_ends)]))
                    if cur_post_wgd_path in post_wgd_paths:
                        continue
                    post_wgd_paths.add(cur_post_wgd_path)
            # this can happen if a start can only connect to an end that has the same value
            post_wgd_paths = [path for path in post_wgd_paths if len(path) == n_post_wgd_events]
            log_debug(logger, f'Created {len(post_wgd_paths)} valid post-WGD paths')

            if len(post_wgd_paths) == 0:
                # in case no valid post-WGD paths are found, continue with the next pre-WGD path
                if len(pre_wgd_path) < n_events:
                    continue
                # if all events are pre-WGD, create this dummy diffs and paths
                post_wgd_diffs = [np.zeros((1, len(cn_profile)))]
                post_wgd_paths = [[]]
            else:
                post_wgd_diffs = get_events_diff_from_coords(post_wgd_paths, cn_profile, lexsort_diffs=True, filter_missed_lohs=False)
    
            post_wgd_valid = check_pre_post_wgd_loh_viability_simple(
                'post', loh_pos, pre_wgd_diff=pre_wgd_diff, post_wgd_diffs=post_wgd_diffs,
                post_wgd_state=post_wgd_state, total_cn=total_cn)
            
            paths.append([(pre_wgd_path, path) for path, valid in zip(post_wgd_paths, post_wgd_valid) if valid])
            paths = [path for path in paths if len(path) > 0] # remove empty paths
            if len(paths) > 0:
                return paths[0][0]


def _deepcopy_fast(cur_events, has_wgd=False):
    if has_wgd:
        return [list(cur_events[0]), list(cur_events[1])]
    else:
        return cur_events.copy()


def create_events_df_from_single_path_solution(full_paths, cur_id, chrom_segments=None,
                                               create_full=True, calc_telomere_bound=False):
    unique_events_df = pd.DataFrame(full_paths.events.values())
    final_events_df = unique_events_df.iloc[np.array(list(full_paths.solutions[0].elements()))].copy()
    final_events_df['events_per_chrom'] = full_paths.n_events
    final_events_df['n_paths'] = int(full_paths.n_solutions)
    final_events_df['solved'] = (
        full_paths.solved if full_paths.solved is not None else 'unambiguous' if full_paths.n_solutions 
        == 1 else '')
    final_events_df['id'] = cur_id
    final_events_df[['sample', 'chrom', 'allele']] = cur_id.split(':')
    if create_full:
        assert chrom_segments is not None, "chrom_segments must be provided if create_full is True"
        final_events_df = create_full_df_from_diff_df(
            final_events_df, cur_id, chrom_segments.query('id == @cur_id').copy(),
            calc_telomere_bound=calc_telomere_bound)
        if not calc_telomere_bound:
            final_events_df[['telomere_bound', 'whole_chrom', 'whole_arm']] = None

    return final_events_df
