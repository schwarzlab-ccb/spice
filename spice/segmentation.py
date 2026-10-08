import pandas as pd
import numpy as np
from tqdm.auto import tqdm

from spice import data_loaders
from spice.utils import CALC_NEW, create_chrom_type_pos_indices
from spice.length_scales import DEFAULT_SEGMENT_SIZE_DICT
from spice.data_loaders import format_chromosomes, load_segmentation
from spice.event_analysis.final_events import classify_event_position

CHROMS = ['chr' + str(x) for x in range(1, 23)] + ['chrX', 'chrY']
CHROM_LENS = data_loaders.load_chrom_lengths()


def create_events_in_segmentation(final_events_df, bin_df=100e3, skip_tqdm=False):
    """
    Map events to segmentation bins.

    Returns a DataFrame indexed like `bin_df.index` with a single column
    `'events'` containing lists of event indices from `final_events_df` that
    overlap each segmentation bin.
    """
    if hasattr(bin_df, '__int__'):
        bin_df = load_segmentation(bin_df)
    else:
        assert isinstance(bin_df, pd.DataFrame)
    events_in_segmentation = pd.DataFrame(
        0,
        index=bin_df.index,
        columns=['events']).sort_index()

    for cur_chrom in tqdm(final_events_df['chrom'].unique(), disable=skip_tqdm):
        cur_events = final_events_df.query('chrom == @cur_chrom').copy()
        if len(cur_events) == 0:
            continue
        events_starts = cur_events['start'].values
        events_ends = cur_events['end'].values
        cur_seg = bin_df.loc[[cur_chrom]].reset_index().copy()
        cur_seg['end'] += 1

        cur_breakpoints = np.unique(cur_seg[['start', 'end']].values.flat)
        events_starts_smaller = events_starts < cur_breakpoints[1:, None]
        events_ends_larger = events_ends > cur_breakpoints[:-1, None]

        output = np.matmul(np.logical_and(events_starts_smaller, events_ends_larger), np.ones(len(cur_events), dtype=int))
        events_in_segmentation.loc[cur_chrom, 'events'] = output

    return events_in_segmentation


def create_events_in_segmentation_full(final_events_df, segmentation=100e3, show_tqdm=False):
    if hasattr(segmentation, '__int__'):
        segmentation = load_segmentation(segmentation)
    else:
        assert isinstance(segmentation, pd.DataFrame)
    if 'pos' not in final_events_df.columns:
        final_events_df['pos'] = classify_event_position(final_events_df)

    all_segmented_events = []
    for cur_type in ['gain', 'loss']:
        for cur_pos in ['whole_chrom', 'whole_arm', 'telomere_bound', 'internal']:
            cur_events = final_events_df.loc[(final_events_df['type'] == cur_type) & (final_events_df['pos'] == cur_pos)]
            cur_segmented_events = create_events_in_segmentation(cur_events, segmentation)
            all_segmented_events.append(cur_segmented_events)
    combined_segmented_events = pd.concat(all_segmented_events, axis=1)
    combined_segmented_events.columns = pd.MultiIndex.from_product(
        [['gain', 'loss'], ['whole_chrom', 'whole_arm', 'telomere_bound', 'internal']],
        names=['type', 'pos'])
    return combined_segmented_events


def get_events_at_position(signal, bin_df, cur_chrom, position):
    """
    Get the signal value at a specific genomic position.
    
    Args:
        signal: Array of signal values (typically from events_in_segmentation.loc[cur_chrom].sum(axis=1).values)
        bin_df: DataFrame defining the segmentation bins (with MultiIndex: chrom, start, end)
        chrom: Chromosome name (e.g., 'chr1')
        position: Genomic position (integer)
    
    Returns:
        float: Signal value at the given position, or 0 if position not found
    """
    if hasattr(bin_df, '__int__'):
        bin_df = load_segmentation(bin_df)
    else:
        assert isinstance(bin_df, pd.DataFrame)
    try:
        # Get all bins for the chromosome
        chrom_bins = bin_df.loc[[cur_chrom]].reset_index()
        
        # Find the bin containing this position
        mask = (chrom_bins['start'] <= position) & (chrom_bins['end'] >= position)
        bin_indices = np.where(mask)[0]
        
        if len(bin_indices) == 0:
            return 0.0
        elif len(bin_indices) == 1:
            return float(signal[bin_indices[0]])
        else:
            # If multiple bins match (shouldn't happen with non-overlapping bins),
            # return the sum
            return float(signal[bin_indices].sum())
    except KeyError:
        # Chromosome not in the dataframe
        return 0.0


def get_events_at_position_all_ls(data_per_length_scale, cur_chrom, position):
    events_at_position = []
    for length_scale, bin_df in DEFAULT_SEGMENT_SIZE_DICT.items():
        for cur_type in ['gain', 'loss']:
            signal = data_per_length_scale[(length_scale, cur_type)]['signals']
            value = get_events_at_position(signal, bin_df, cur_chrom, position)
            events_at_position.append(value)
    return events_at_position
