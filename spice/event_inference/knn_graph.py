import sys
import pickle
import itertools
from collections import namedtuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from spice import config, directories
from spice.logging import log_debug
from spice.utils import (
    CALC_NEW, get_logger, open_pickle, save_pickle, create_full_df_from_diff_df)
from spice.data_loaders import load_chrom_lengths

# Use importlib.resources for accessing package data (works with installed packages)
if sys.version_info >= (3, 9):
    from importlib.resources import files
else:
    try:
        from importlib_resources import files
    except ImportError:
        files = None

logger = get_logger('spice.knn_graph', load_config=True)

EventDistData = namedtuple('EventDistData', 
                               ['chrom', 'starts', 'ends', 'widths', 'type', 'is_telomere_bound',
                                'is_whole_chrom', 'is_whole_arm', 'wgd', 'chrom_lengths'])


def get_event_dist_data_from_df(events_df):
    chrom_lengths = load_chrom_lengths()
    event_dist_data = EventDistData(
        chrom=events_df['chrom'].values,
        starts=events_df['start'].values,
        ends=events_df['end'].values,
        widths=events_df['width'].values,
        type=events_df['type'].values,
        is_telomere_bound=events_df['telomere_bound'].values,
        is_whole_chrom=events_df['whole_chrom'].values,
        is_whole_arm=events_df['whole_arm'].values,
        wgd=events_df['wgd'].values,
        chrom_lengths=chrom_lengths.loc[events_df['chrom'].values].values
    )

    return event_dist_data


def calc_event_distances(train_events, test_events, wgd_split=True, block_same_id=True,
                         assert_finite=True, ks=[1, 2, 3, 4, 5, 10, 25, 100, 250, 500],
                         show_progress=False, ignore_empty_train=False, clip_k=False,
                         log10_distances=True, single_width_bin=True):

    single_k = not hasattr(ks, '__iter__')
    if single_k:
        ks = [ks]

    ks = np.array(ks)
    n_ks = len(ks)
    max_k = max(ks)

    if isinstance(test_events, pd.DataFrame):
        events_data = get_event_dist_data_from_df(test_events)
    elif isinstance(test_events, EventDistData):
        events_data = test_events
    else:
        raise ValueError(f'test_events must be a DataFrame or EventDistData. Current is of type: {type(test_events)}')
    
    assert np.isin(events_data.type, ['gain', 'loss']).all()

    event_distances = np.inf * np.ones((len(events_data.starts), n_ks))

    pre_post_wgds = ['nowgd', 'pre', 'post'] if wgd_split else ['all']

    for cur_type, telomere_bound, pre_post_wgd in itertools.product(['gain', 'loss'], [True, False], pre_post_wgds):
        # log_debug(logger, f"Calculating distances for {cur_type} {'tel' if telomere_bound else 'int'} {'all' if pre_post_wgd == 'all' else pre_post_wgd} events")
        if pre_post_wgd == "all":

            cur_mask_test = np.logical_and.reduce([
                (events_data.type == cur_type),
                (events_data.is_telomere_bound == telomere_bound),
                (~events_data.is_whole_chrom),
                (~events_data.is_whole_arm)
            ])
        else:
            cur_mask_test = np.logical_and.reduce([
                (events_data.type == cur_type),
                (events_data.is_telomere_bound == telomere_bound),
                (~events_data.is_whole_chrom),
                (~events_data.is_whole_arm),
                (events_data.wgd == pre_post_wgd)
            ])
        if cur_mask_test.sum() == 0:
            continue
        cur_train_events = train_events[(cur_type, telomere_bound, pre_post_wgd)]
        cur_n_train = len(cur_train_events[0])

        if cur_n_train == 0:
            if ignore_empty_train:
                continue
            else:
                raise ValueError(f'no train events for {cur_type} {"telomere-bound" if telomere_bound else "internal"} {pre_post_wgd} -> set "ignore_empty_train" to True to skip')

        cur_ks = ks.copy()
        if cur_n_train < max_k:
            if clip_k:
                cur_ks[cur_ks > cur_n_train] = cur_n_train
                if len(cur_ks) == 0:
                    if ignore_empty_train:
                        continue
                    else:
                        raise ValueError(f'not enough train events for {cur_type} {"telomere-bound" if telomere_bound else "internal"} for current k -> set "ignore_empty_train" to skip')
            else:
                raise ValueError(f'not enough train events for {cur_type} {"telomere-bound" if telomere_bound else "internal"}: Found {cur_n_train} but requires {max_k} -> set "clip_k" to skip')

        if single_width_bin:
            cur_n_bins = 1
        else:
            cur_n_bins_ = int(np.floor(cur_n_train / max_k))

            width_bins = pd.qcut(
                np.append(
                    cur_train_events[0],
                    [events_data.widths[cur_mask_test].min(), events_data.widths[cur_mask_test].max()]
                    # [test_events.loc[cur_mask_test]['width'].values.min(), test_events.loc[cur_mask_test]['width'].values.max()]
                ), q=cur_n_bins_, duplicates='drop').categories

            # update in case some bin-edges were duplicate and thus dropped
            cur_n_bins = len(width_bins)
            cur_train_width_bins = pd.cut(cur_train_events[0], bins=width_bins)
            cur_test_width_bins = pd.cut(events_data.widths[cur_mask_test], bins=width_bins)
            # cur_test_width_bins = pd.cut(test_events.loc[cur_mask_test]['width'].values, bins=width_bins)
        
        log_debug(logger, f'{cur_type} {"telomere-bound" if telomere_bound else "internal"}: {cur_n_train} train, {cur_mask_test.sum()} test, {cur_n_bins} bins')
        for i in tqdm(range(cur_n_bins), disable=(not show_progress)):
            if single_width_bin:
                # log_debug(logger, f'{cur_type} {"telomere-bound" if telomere_bound else "internal"} bin {i+1} / 1 (single_width_bin=True): {cur_n_train} train, {cur_mask_test.sum()} test')
                cur_bin_mask_test = np.ones(cur_mask_test.sum(), dtype=bool)
                cur_bin_mask_train = np.ones(cur_n_train, dtype=bool)
            else:
                cur_bin_mask_test = cur_test_width_bins == width_bins[i]
                if cur_bin_mask_test.sum() == 0:
                    continue
                cur_bin_mask_train = np.isin(cur_train_width_bins, 
                                            width_bins[max(0, i - 1):min(cur_n_bins, i + 2)])
                # log_debug(logger, f'{cur_type} {"telomere-bound" if telomere_bound else "internal"} bin {i+1} / {cur_n_bins}: {cur_bin_mask_train.sum()} train, {cur_bin_mask_test.sum()} test')
                # log_debug(logger, f'test width bin: {width_bins[i]}, train width bins: {width_bins[max(0, i - 1):min(cur_n_bins, i + 2)]}')

            if isinstance(events_data.chrom_lengths, np.ndarray):
                chrom_length_norm = (events_data.chrom_lengths[cur_mask_test][cur_bin_mask_test][:, None] + 
                                     cur_train_events[1][cur_bin_mask_train])/2
            else:
                chrom_length_norm = ((events_data.chrom_lengths + cur_train_events[1][cur_bin_mask_train])/2)[None, :]

            cur_event_distances = np.abs(
                events_data.widths[cur_mask_test][cur_bin_mask_test][:, None] - 
                    cur_train_events[0][cur_bin_mask_train]) / chrom_length_norm

            # add infinity for events that are from same ID
            if block_same_id:
                if not isinstance(test_events, pd.DataFrame):
                    raise NotImplementedError('block_same_id only implemented for DataFrame test_events')
                # log_debug(logger, 'blocking same id events')
                cur_event_distances += max(99, cur_event_distances.max()) * (
                    test_events.loc[cur_mask_test, 'id'].values[cur_bin_mask_test][:, None] == 
                    cur_train_events[2][cur_bin_mask_train]).astype(int)
            # log_debug(logger, 'sorting and cumsumming distances')
            cur_event_distances = np.cumsum(np.sort(cur_event_distances, axis=1), axis=1)[:, cur_ks - 1]
            event_distances[np.arange(len(event_distances))[cur_mask_test][cur_bin_mask_test], :len(cur_ks)] = cur_event_distances

    event_distances[events_data.is_whole_chrom, :] = 0
    event_distances[events_data.is_whole_arm, :] = 0

    if assert_finite:
        assert clip_k or np.isfinite(event_distances).all(), 'some events have no finite distances'
        assert np.isfinite(event_distances).any(axis=1).all(), 'some events have no finite distances'

    if log10_distances:
        event_distances_ = -6 * np.ones_like(event_distances)
        event_distances_[event_distances != 0] = np.log10(event_distances[event_distances != 0])
        # There are cases where there is only whole-arm/chrom solutions (e.g. wgd "13" with bp at centromere)
        # prefer whole-chrom solutions over whole-arm solutions and post-WGD over pre-WGD
        event_distances_[np.logical_and(events_data.is_whole_chrom, events_data.wgd=='nowgd'), :] = -6.01
        event_distances_[np.logical_and(events_data.is_whole_arm, events_data.wgd=='nowgd'), :] = -6
        event_distances_[np.logical_and(events_data.is_whole_chrom, events_data.wgd=='pre'), :] = -6.001
        event_distances_[np.logical_and(events_data.is_whole_chrom, events_data.wgd=='post'), :] = -6.01
        event_distances_[np.logical_and(events_data.is_whole_arm, events_data.wgd=='pre'), :] = -6
        event_distances_[np.logical_and(events_data.is_whole_arm, events_data.wgd=='post'), :] = -6.0001
        event_distances = event_distances_

    if single_k:
        event_distances = event_distances[:, 0]

    return event_distances


def solve_with_knn(full_paths, cur_chrom_segments, knn_train_data, k=250, wgd_split=True,
                   log10_distances=True, save_all_scores=None, ignore_empty_train=False,
                   single_width_bin=True, clip_k=True, perform_loh_checks=False):

    from spice.event_inference.events_from_graph import check_loh_for_full_paths  # lazy: needs ortools/fstlib/medicc

    # because of how calc_event_distances works now...
    if hasattr(k, '__iter__'):
        assert len(k) == 1, 'k must be a single integer'
    else:
        k = [k]

    log_debug(logger, f'Starting knn-graph with k={k} for {full_paths.id}')

    cur_id = full_paths.id

    if isinstance(cur_chrom_segments, str):
        log_debug(logger, f'loading chrom_segments from {cur_chrom_segments}')
        cur_chrom_segments = pd.read_csv(cur_chrom_segments, sep='\t', index_col=['sample_id', 'chrom', 'allele']).query('id == @cur_id')

    assert isinstance(knn_train_data, dict), 'knn_train_data must be a string or a dictionary'

    index, full_paths_data = zip(*[[k, v] for k, v in full_paths.events.items()])

    unique_events_df = pd.DataFrame(full_paths_data, index=index)
    unique_events_df = create_full_df_from_diff_df(unique_events_df, cur_id, cur_chrom_segments)
    unique_events_df['n_paths'] = int(full_paths.n_solutions)
    unique_events_df['events_per_chrom'] = full_paths.n_events
    unique_events_df['solved'] = 'knn'

    events_per_path = np.zeros((len(full_paths.solutions), len(full_paths.events)), dtype=int)
    for i, solution in enumerate(full_paths.solutions):
        events_per_path[i, np.array(list(solution.keys()))] = list(solution.values())
    assert np.all(events_per_path.sum(axis=1) == full_paths.n_events), f'{events_per_path.sum(axis=1)} != {full_paths.n_events}'

    log_debug(logger, f'Found {len(unique_events_df)} unique events. Calculating distances')
    unique_events_distances = calc_event_distances(knn_train_data, unique_events_df, block_same_id=False,
                            ks=k, show_progress=False, ignore_empty_train=ignore_empty_train, clip_k=clip_k,
                            wgd_split=wgd_split, log10_distances=log10_distances, single_width_bin=single_width_bin)
    log_debug(logger, 'Finished calculating distances. Selecting best solution')
    path_distances = (events_per_path * unique_events_distances[:, 0].T).sum(axis=1)

    # a bit hacky: penalize events that were gained and then lost (only required for WGD)
    gain_loss_diffs = (unique_events_df.groupby(['diff', 'type']).size().astype(bool)
                       .groupby(['diff']).sum().to_frame('gain_loss_event').query('gain_loss_event == 2').index.values)
    for diff in gain_loss_diffs:
        cur_gain_ind = unique_events_df.query('type=="gain" and diff==@diff').index.values
        cur_loss_ind = unique_events_df.query('type=="loss" and diff==@diff').index.values
        paths_with_same_gain_loss_event = np.logical_and(
            events_per_path[:, cur_gain_ind].sum(axis=1) > 0,
            events_per_path[:, cur_loss_ind].sum(axis=1) > 0)
        path_distances[paths_with_same_gain_loss_event] = np.max(path_distances)

    if perform_loh_checks and (0 in full_paths.cn_profile):
        log_debug(logger, 'Performing LOH checks')
        best_path_indices = np.argsort(path_distances)
        best_path_index = None
        for idx in best_path_indices:
            valid_indices = check_loh_for_full_paths(full_paths, solutions=[idx])
            if len(valid_indices) > 0:
                best_path_index = idx
                break
        if best_path_index is None:
            raise ValueError('No valid path found that passes LOH checks')
    else:
        best_path_index = np.argmin(path_distances)
        
    if (path_distances == path_distances[best_path_index]).sum() > 1:
        logger.warning(f'Multiple best paths ({(path_distances == path_distances[best_path_index]).sum()}) with min distance ({path_distances[best_path_index]}) found')
    final_events_indices = np.where(events_per_path[best_path_index] > 0)[0]
    final_events_indices = np.repeat(final_events_indices, events_per_path[best_path_index][final_events_indices])
    unique_events_df['knn_distance'] = unique_events_distances[:, 0]
    final_events_df = unique_events_df.iloc[final_events_indices].copy()
    assert len(final_events_df) == full_paths.n_events 

    if save_all_scores is not None and (isinstance(save_all_scores, bool) and save_all_scores):
        save_pickle((unique_events_df, events_per_path, path_distances), save_all_scores)
    
    return final_events_df


def load_knn_train():
    if files is None:
        raise FileNotFoundError("importlib.resources unavailable for knn_train data")
    try:
        knn_train_file = files('spice').joinpath('objects', 'train_events_sv_and_unamb.pickle').read_bytes()
        knn_train = pickle.loads(knn_train_file)
        
    except (TypeError, ImportError, AttributeError, FileNotFoundError) as exc:
        raise FileNotFoundError("Could not find knn_train data in spice/objects/") from exc
    assert isinstance(knn_train, dict), 'knn_train data should be a dictionary'

    return knn_train
