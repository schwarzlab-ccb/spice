"""Event filtering shared by observed-table construction and loci analysis.

Importing this module needs only static assembly tables. Observed-table construction
uses use_observed_centromeres=False to avoid depending on its own output.
"""
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd

from spice import data_loaders
from spice.length_scales import DEFAULT_LENGTH_SCALE_BOUNDARIES
from spice.logging import get_logger, log_debug
from spice.utils import calc_telomere_bound_whole_arm_whole_chrom
from spice.tsg_og.plateaus import categorize_events_by_plateau_overlap

logger = get_logger('loci_preprocessing')


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

    import spice
    config = spice.config
    if use_observed_centromeres:
        CENTROMERES_OBSERVED = data_loaders.load_centromeres(observed=True, extended=False)

    if final_events_df is None:
        log_debug(logger, "Loading final events dataframe from file")
        final_events_df = data_loaders.load_final_events()

    raw_length = len(final_events_df)

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

    # Remove events contained within the extended centromere plus 5 Mb on either side.
    log_debug(logger, 'Remove whole centromere events and events contained within the extended centromere plus 5 Mb padding')
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
    assert len(final_events_df) > 0, 'No events left after removing whole centromeres and events contained within the extended centromere plus 5 Mb padding. Please check the centromere definitions and event coordinates.'
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
    min_width = length_scale_boundaries['small'][0]
    max_width = length_scale_boundaries['large'][1]
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

    logger.info(f'Processed final events for loci routines: {len(final_events_df)} events across {final_events_df["sample"].nunique()} samples and {final_events_df["id"].nunique()} IDs (from {raw_length} raw events)')

    return final_events_df
