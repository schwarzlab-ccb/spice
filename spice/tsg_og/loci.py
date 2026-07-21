import os
from functools import reduce

import pandas as pd
import numpy as np
from scipy.signal._peak_finding_utils import PeakPropertyWarning
from scipy.ndimage import gaussian_filter1d
from scipy.signal import peak_widths as scipy_peak_widths
from scipy.signal import peak_prominences
from scipy.stats import false_discovery_control

from spice import data_loaders
from spice.logging import log_debug
from spice.utils import get_logger, suppress_warnings, assert_close, open_pickle, save_pickle, CALC_NEW
from spice.length_scales import DEFAULT_SEGMENT_SIZE_DICT, LENGTH_SCALE_NAMES, LS_I_DICT_REV
from spice.tsg_og.simulation import (
    convolution_simulation, SelectionPoints, combine_selection_points)
from spice.tsg_og.event_rate_per_loci import calc_total_events_per_loci

CENTROMERES = data_loaders.load_centromeres(extended=False, observed=False)
CHROMS = ['chr' + str(x) for x in range(1, 23)] + ['chrX', 'chrY']
CHROM_LENS = data_loaders.load_chrom_lengths()
logger = get_logger('loci')

def calc_overlaps(a, b):
    return np.logical_and(
        ~np.logical_or(
            a['start'].values[:, None] > b['end'].values,
            a['end'].values[:, None] < b['start'].values
        ),
        a['chrom'].values.astype(str)[:, None] == b['chrom'].values.astype(str)
    )


def calc_fraction_overlaps(a, b, norm_by='a'):
    """
    Calculate the fraction of overlap between intervals in a and b.

    Args:
        a (pd.DataFrame): DataFrame with columns ['chrom', 'start', 'end'].
        b (pd.DataFrame): DataFrame with columns ['chrom', 'start', 'end'].
        norm_by (str): 'a' to normalize by length of a, 'b' to normalize by length of b.

    Returns:
        np.ndarray: 2D array of shape (len(a), len(b)) with fractional overlaps.
    """
    overlaps = calc_overlaps(a, b)
    frac_matrix = np.zeros(overlaps.shape, dtype=float)

    for i in range(a.shape[0]):
        for j in range(b.shape[0]):
            if overlaps[i, j]:
                start = max(a.iloc[i]['start'], b.iloc[j]['start'])
                end = min(a.iloc[i]['end'], b.iloc[j]['end'])
                overlap_len = max(0, end - start)
                if norm_by == 'a':
                    denom = a.iloc[i]['end'] - a.iloc[i]['start']
                elif norm_by == 'b':
                    denom = b.iloc[j]['end'] - b.iloc[j]['start']
                else:
                    raise ValueError("norm_by must be 'a' or 'b'")
                frac_matrix[i, j] = overlap_len / denom if denom > 0 else 0.0
    return frac_matrix


def create_loci_df(all_selection_points, all_loci_widths=None, nr_stds_widths=2,
                    min_widths_is_small_kernel=True):

    if all_loci_widths is None:
        all_loci_widths = {cur_chrom: len(all_selection_points[cur_chrom][0]) * [[0]]
                           for cur_chrom in all_selection_points.keys()}

    loci_df = pd.DataFrame({
        'chrom': np.concatenate([[cur_chrom] * len(chrom_selection_points[0])
                            for cur_chrom, chrom_selection_points in all_selection_points.items()]),
        'pos': np.concatenate([[x[0].pos for x in chrom_selection_points[0]]
                            for chrom_selection_points in all_selection_points.values()]),
        'rank_on_chrom': np.concatenate([[i for i in range(len(cur_selection_points[0]))] for cur_selection_points in all_selection_points.values()]),
        'width': np.concatenate([[np.std(x) for x in chrom_loci_widths]
                                    for chrom_loci_widths in all_loci_widths.values()]),
    })
    if len(loci_df) == 0:
        raise ValueError('No loci found. Please check your data and the input parameters.')
    loci_df['chrom'] = data_loaders.format_chromosomes(loci_df['chrom'])
    loci_df['pos_total'] = (loci_df.join(CHROM_LENS.cumsum().shift(1, fill_value=0)
                                           .to_frame('chrom_offset'), on='chrom')
                             .eval('pos + chrom_offset').values)
    loci_df['width'] = nr_stds_widths * loci_df['width']
    if min_widths_is_small_kernel:
        # 476_173 is the mean kernel size across all chromosomes for the small length scale (and gain) 
        loci_df['width'] = np.maximum(loci_df['width'].values, 476_173/4)

    loci_df['start'] = loci_df['pos'] - loci_df['width'] / 2
    loci_df['end'] = loci_df['pos'] + loci_df['width'] / 2
    loci_df[[f'fitness_{LS_I_DICT_REV[i][0]}_{LS_I_DICT_REV[i][1]}' for i in range(8)]] = np.concatenate([np.stack(
        [[ls[0].fitness for ls in locus] for locus in list(zip(*chrom_selection_points))])
        for chrom_selection_points in all_selection_points.values()], axis=0)
    loci_df['type'] = np.where((loci_df[[f'fitness_{LS_I_DICT_REV[i][0]}_{LS_I_DICT_REV[i][1]}'
                                            for i in range(0, 8, 2)]] > 0).any(axis=1), 'OG', 'TSG')
    loci_df = loci_df.sort_values(by=['chrom', 'pos']).reset_index(drop=True)
    return loci_df


def overlap_with_cosmic_davoli(loci_df, cosmic_loci, davoli_loci):
    """
    This function creates a DataFrame that combines information from COSMIC and Davoli loci
    and determines overlaps with the provided loci DataFrame.

    Args:
        loci_df (pd.DataFrame): DataFrame containing loci with columns ['chrom', 'start', 'end'].
        cosmic_loci (pd.DataFrame): DataFrame containing COSMIC loci with columns ['Gene Symbol', 'chrom', 'start', 'end'].
        davoli_loci (pd.DataFrame): DataFrame containing Davoli loci with columns ['name', 'chrom', 'start', 'end'].

    Returns:
        pd.DataFrame: Updated loci_df with additional columns:
            - 'cosmic': Boolean indicating overlap with COSMIC loci.
            - 'davoli': Boolean indicating overlap with Davoli loci.
            - 'cosmic_genes': List of overlapping COSMIC genes.
            - 'davoli_genes': List of overlapping Davoli genes.
            - 'cosmic_davoli_genes': List of genes overlapping both COSMIC and Davoli.
            - 'n_cosmic_genes': Number of overlapping COSMIC genes.
            - 'n_davoli_genes': Number of overlapping Davoli genes.
    """
    cosmic_overlaps = calc_overlaps(cosmic_loci, loci_df)
    davoli_overlaps = calc_overlaps(davoli_loci, loci_df)
    loci_df['cosmic'] = cosmic_overlaps.any(axis=0)
    loci_df['davoli'] = davoli_overlaps.any(axis=0)

    loci_df['cosmic_genes'] = [[cosmic_loci['Gene Symbol'].values[y] for y in np.where(x)[0]] for x in cosmic_overlaps.T]
    loci_df['davoli_genes'] = [[davoli_loci['name'].values[y] for y in np.where(x)[0]] for x in davoli_overlaps.T]
    loci_df['cosmic_davoli_genes'] = loci_df.apply(lambda x: np.intersect1d(x['cosmic_genes'], x['davoli_genes']).tolist() if x['cosmic'] > 0 and x['davoli'] > 0 else [], axis=1)
    loci_df['n_cosmic_genes'] = loci_df['cosmic_genes'].map(len)
    loci_df['n_davoli_genes'] = loci_df['davoli_genes'].map(len)

    assert ((loci_df['n_cosmic_genes'] > 0) == loci_df['cosmic']).all()
    assert ((loci_df['n_davoli_genes'] > 0) == loci_df['davoli']).all()

    return loci_df


def overlap_with_gistic_biscut(loci_df, gistic_loci, biscut_loci):
    gistic_overlaps = calc_overlaps(gistic_loci, loci_df)
    biscut_overlaps = calc_overlaps(biscut_loci, loci_df)
    loci_df['gistic'] = gistic_overlaps.any(axis=0)
    loci_df['biscut'] = biscut_overlaps.any(axis=0)

    loci_df['gistic_loci'] = [np.where(x)[0] for x in gistic_overlaps.T]
    loci_df['biscut_loci'] = [np.where(x)[0] for x in biscut_overlaps.T]
    loci_df['gistic_genes'] = loci_df['gistic_loci'].map(
        lambda x: sum([[] if isinstance(gistic_loci['genes'].values[y], float) else
                    gistic_loci['genes'].values[y] for y in x], []))
    loci_df['biscut_genes'] = loci_df['biscut_loci'].map(
        lambda x: sum([[] if isinstance(biscut_loci['genes'].values[y], float) else
                    biscut_loci['genes'].values[y] for y in x], []))
    loci_df['gistic_biscut_genes'] = loci_df.apply(lambda x: np.intersect1d(x['gistic_genes'], x['biscut_genes']).tolist() if x['gistic'] > 0 and x['biscut'] > 0 else [], axis=1)
    loci_df['n_gistic_loci'] = loci_df['gistic_loci'].map(len)
    loci_df['n_biscut_loci'] = loci_df['biscut_loci'].map(len)
    loci_df['n_gistic_genes'] = loci_df['gistic_genes'].map(len)
    loci_df['n_biscut_genes'] = loci_df['biscut_genes'].map(len)

    assert loci_df.query('not gistic')['n_gistic_genes'].max() == 0
    assert loci_df.query('not biscut')['n_biscut_genes'].max() == 0

    return loci_df


def calculate_events_per_loci_df(loci_df, all_selection_points=None, final_events_df=None, rates_and_events_per_loci=None):

    loci_df['added_events'] = 0.
    for cur_chrom in CHROMS[:-1]:
        if rates_and_events_per_loci is not None:
            total_events_per_loci = rates_and_events_per_loci[cur_chrom][1]
        else:
            assert all_selection_points is not None and final_events_df is not None
            if len(final_events_df.query('chrom == @cur_chrom and pos == "internal"')) == 0:
                continue
            total_events_per_loci = calc_total_events_per_loci(
                cur_chrom,
                final_events_df=final_events_df,
                cur_selection_points=all_selection_points[cur_chrom],
            )
        cur_added_events = [x[:-1] for x in total_events_per_loci.values() if not isinstance(x, int)]
        if len(cur_added_events) == 0:
            continue
        cur_added_events = np.sum(np.stack(cur_added_events), axis=0)
        cur_index = loci_df.query('chrom == @cur_chrom').sort_values('rank_on_chrom').index
        loci_df.loc[cur_index, 'added_events'] = cur_added_events

    return loci_df


@suppress_warnings(PeakPropertyWarning)
def calc_prominence(cur_chrom, data_per_length_scale, cur_loci_df=None, selection_points=None, calc_on='conv',
                    loci_widths=None, size_neighborhood=10, segment_size_dict=DEFAULT_SEGMENT_SIZE_DICT):

    assert calc_on in ['data', 'conv'], 'calc_on must be either "data" or "conv"'

    if cur_loci_df is None:
        assert selection_points is not None and loci_widths is not None, 'If cur_loci_df is None, final_selection_points and final_loci_widths must be provided'
        cur_loci_df = create_loci_df(all_selection_points={cur_chrom: selection_points},
                                       all_loci_widths={cur_chrom: loci_widths})
    cur_ind = np.where(cur_loci_df['chrom'] == cur_chrom)[0]
    # has to order by pos here so that limiting by neighboring loci works
    cur_loci_df = cur_loci_df.loc[cur_ind].sort_values('pos')

    ls_i = -1
    for cur_length_scale in ['small', 'mid1', 'mid2', 'large']:
        for cur_type in ['gain', 'loss']:
            ls_i += 1

            cur_kernel_size = data_per_length_scale[(cur_length_scale, cur_type)]['loci_width']

            if calc_on == 'data':
                data_for_prominence = data_per_length_scale[(cur_length_scale, cur_type)]['signals']
                data_for_prominence = gaussian_filter1d(data_for_prominence, size_neighborhood)
            elif calc_on == 'conv':
                data = data_per_length_scale[(cur_length_scale, cur_type)]
                if selection_points is None:
                    cur_selection_points = selection_points_from_loci_df(cur_loci_df, cur_chrom, ls_i)
                else:
                    cur_selection_points = selection_points[ls_i]

                data_for_prominence = convolution_simulation(
                    cur_chrom=cur_chrom, selection_points=combine_selection_points(cur_selection_points),
                    cur_widths=data['cur_widths'], kernel=data['kernel'], chrom_size=None,
                    kernel_edge=data.get('kernel_edge', None), cur_length_scale=data['length_scale'], cur_signal=data['signals'],
                    segment_size=segment_size_dict[data['length_scale']], centromere_values=data['centromere_values'],
                    height_multiplier=data.get('height_multiplier', None))

            cur_loci_df = process_locus_prominence(
                cur_loci_df=cur_loci_df,
                data_for_prominence=data_for_prominence,
                cur_length_scale=cur_length_scale,
                cur_type=cur_type,
                cur_kernel_size=cur_kernel_size,
                size_neighborhood=size_neighborhood,
                segment_size_dict=segment_size_dict,
            )

    # set prominence to zero if the fitness is zero
    cur = cur_loci_df[[col for col in cur_loci_df.columns if col.startswith('prominence_OG')]].values
    cur[cur_loci_df[[x for x in cur_loci_df.columns if x.startswith('fit_') and  x.endswith('gain')]].values <= 0] = 0
    cur_loci_df[[col for col in cur_loci_df.columns if col.startswith('prominence_OG')]] = cur
    cur = cur_loci_df[[col for col in cur_loci_df.columns if col.startswith('prominence_TSG')]].values
    cur[cur_loci_df[[x for x in cur_loci_df.columns if x.startswith('fit_') and  x.endswith('loss')]].values <= 0] = 0
    cur_loci_df[[col for col in cur_loci_df.columns if col.startswith('prominence_TSG')]] = cur
    
    # set max_prominence
    cur_loci_df['max_prominence'] = cur_loci_df[[col for col in cur_loci_df.columns if col.startswith('prominence_OG') ]].max(axis=1)
    cur_loci_df.loc[cur_loci_df['type'] == 'TSG', 'max_prominence'] = cur_loci_df.loc[cur_loci_df['type'] == 'TSG'][[col for col in cur_loci_df.columns if col.startswith('prominence_TSG') ]].max(axis=1)
    cur_loci_df['sum_prominence'] = cur_loci_df[[col for col in cur_loci_df.columns if col.startswith('prominence')]].sum(axis=1)

    # order by rank_on_chrom to fit with selection_points
    cur_loci_df = cur_loci_df.sort_values('rank_on_chrom')

    return cur_loci_df


@suppress_warnings(PeakPropertyWarning)
def process_locus_prominence(
    cur_loci_df, data_for_prominence, cur_length_scale, cur_type, cur_kernel_size, size_neighborhood, segment_size_dict
):
    """
    Process and calculate the prominence of loci for a given signal.

    Parameters:
    -----------
    cur_loci_df : pd.DataFrame
        DataFrame containing locus information (e.g., position, start, end).
    data_for_prominence : np.ndarray
        Signal array for the current length scale and type.
    cur_length_scale : str
        Current length scale (e.g., 'small', 'mid1', etc.).
    cur_type : str
        Current type (e.g., 'gain' or 'loss').
    cur_kernel_size : int
        Kernel size for limiting the locus width.
    size_neighborhood : int
        Size of the neighborhood for smoothing.
    segment_size_dict : dict
        Dictionary mapping length scales to segment sizes.

    Returns:
    --------
    pd.DataFrame
        Updated DataFrame with calculated prominence values for each locus.
    """

    assert np.all(cur_loci_df['pos'].values == np.sort(cur_loci_df['pos'].values)), \
        'Positions in cur_loci_df must be sorted'

    cur_loci_type = 'OG' if cur_type == 'gain' else 'TSG'
    cur_loci_df[f'prominence_{cur_loci_type}_{cur_length_scale}'] = 0.
    cur_loci_df_ud = cur_loci_df.query('type == @cur_loci_type')
    for i, ind in enumerate(cur_loci_df_ud.index):
        cur_pos_raw = int(cur_loci_df_ud.iloc[i]['pos'] / segment_size_dict[cur_length_scale])
        # Choose the std start / end values
        cur_start = int(cur_loci_df_ud.iloc[i]['start'] / segment_size_dict[cur_length_scale])
        cur_end = int(cur_loci_df_ud.iloc[i]['end'] / segment_size_dict[cur_length_scale])
        # Limit start / end by the kernel size
        cur_start = int(max(cur_start, cur_pos_raw - cur_kernel_size / 2))
        cur_end = int(min(cur_end, cur_pos_raw + cur_kernel_size / 2))
        # If the range is too small, set to 2*size_neighborhood
        if (cur_end - cur_start) < 2 * size_neighborhood:
            cur_start = cur_pos_raw - size_neighborhood
            cur_end = cur_pos_raw + size_neighborhood
        # Limit start / end by the neighboring loci
        if i > 0:
            cur_start = max(cur_start, int(cur_loci_df_ud.iloc[i - 1]['pos'] / segment_size_dict[cur_length_scale]) + size_neighborhood)
        if i < len(cur_loci_df_ud) - 1:
            cur_end = min(cur_end, int(cur_loci_df_ud.iloc[i + 1]['pos'] / segment_size_dict[cur_length_scale]) - size_neighborhood)
        
        # This can happen for larger LS if the two loci left and right are quite close
        if cur_start >= cur_end:
            # cur_loci_df.loc[ind, f'prominence_{cur_type}_{cur_length_scale}'] = 0
            # continue
            cur_start = cur_pos_raw - 1
            cur_end = cur_pos_raw + 2

        # If multiple loci are very close to the telomere this can happen
        if cur_start > len(data_for_prominence) - 2:
            cur_start = len(data_for_prominence) - 2
            cur_end = len(data_for_prominence) - 1
        if cur_end <= 0:
            cur_start = 0
            cur_end = 1

        assert cur_start < cur_end, f'Start ({cur_start}) must be less than end ({cur_end}) for locus {ind}'
        cur_pos = cur_start + np.argmax(data_for_prominence[max(0, cur_start):min(cur_end, len(data_for_prominence) - 1)])
        cur_pos = max(0, min(cur_pos, len(data_for_prominence) - 1))

        cur_prominence = peak_prominences(data_for_prominence, np.array([cur_pos]))[0]
        cur_loci_df.loc[ind, f'prominence_{cur_loci_type}_{cur_length_scale}'] = cur_prominence

        # Set prominence to zero if the locus width is wider than the kernel size
        actual_loci_width = scipy_peak_widths(
            data_for_prominence, np.array([cur_pos]), rel_height=0.25, prominence_data=None, wlen=None
        )[0][0]
        if actual_loci_width > cur_kernel_size * 2:
            cur_loci_df.loc[ind, f'prominence_{cur_loci_type}_{cur_length_scale}'] = 0.

    return cur_loci_df


def calc_locus_prominence_on_combined_conv(loci_df, all_simulated_conv_combined, all_data_per_length_scale, segment_size_dict,
                                          size_neighborhood=10):
    """
    Compute combined prominence for each locus in loci_df using process_locus_prominence.

    Returns a DataFrame with new columns: combined_prominence_OG, combined_prominence_TSG, combined_prominence.
    """
    cur_length_scale = 'small'

    all_loci = []
    for cur_chrom in CHROMS[:-1]:
        cur_loci_df = loci_df.query('chrom == @cur_chrom').sort_values('pos').copy()
        for cur_type in ['gain', 'loss']:
            data_for_prominence = all_simulated_conv_combined[cur_chrom][cur_type]
            cur_kernel_size = all_data_per_length_scale[cur_chrom][('large', cur_type)]['loci_width']

            cur_loci_df_ = process_locus_prominence(
                cur_loci_df=cur_loci_df,
                data_for_prominence=data_for_prominence,
                cur_length_scale=cur_length_scale,
                cur_loci_type=cur_type,
                cur_kernel_size=cur_kernel_size,
                size_neighborhood=size_neighborhood,
                segment_size_dict=segment_size_dict,
            )
            cur_loci_df[f'combined_prominence_{cur_type}'] = cur_loci_df_[f'prominence_{cur_type}_{cur_length_scale}']
        cur_loci_df['combined_prominence'] = cur_loci_df[[f'combined_prominence_{t}' for t in ['gain', 'loss']]].max(axis=1)
        all_loci.append(cur_loci_df)

    all_loci = pd.concat(all_loci, ignore_index=False)
    assert np.all((all_loci.sort_index()[['chrom', 'pos']].values == loci_df.sort_index()[['chrom', 'pos']].values))
    cols = ['combined_prominence', 'combined_prominence_OG', 'combined_prominence_TSG']
    loci_df[cols] = all_loci[cols]
    return loci_df


def prominence_overlap_check(loci_df, data_per_length_scale, threshold, segment_size_dict=DEFAULT_SEGMENT_SIZE_DICT):
    cur_fitness = loci_df[[f'fit_{i}' for i in range(8)]].values
    largest_nonzero_ls_i = 7 - np.argmax(cur_fitness[:, ::-1]!=0, axis=1)
    largest_nonzero_ls = [{
        0: 'mid2', 1: 'mid2', 2: 'mid2', 3: 'mid2', 4: 'mid1', 5: 'mid1', 6: 'mid1', 7: 'mid1'
        # 0: 'small', 1: 'small', 2: 'mid1', 3: 'mid1', 4: 'mid2', 5: 'mid2', 6: 'large', 7: 'large'
        }[x] for x in largest_nonzero_ls_i]
    largest_nonzero_ls_kernel_size = [
        data_per_length_scale[(x, 'gain')]['loci_width'] * segment_size_dict[x]
        for x in largest_nonzero_ls]
    loci_df['largest_nonzero_ls'] = largest_nonzero_ls
    loci_df['largest_nonzero_ls_kernel_size'] = largest_nonzero_ls_kernel_size

    loci_to_remove = np.where(loci_df['max_prominence'].values <= threshold)[0]
    loci_to_keep = np.where(loci_df['max_prominence'].values > threshold)[0]

    # Check if loci to remove are within other loci
    type_order = loci_df['type'].values
    locus_pos = loci_df['pos'].values
    locus_chrom = loci_df['chrom'].values.astype(str)
    locus_start = locus_pos - loci_df['largest_nonzero_ls_kernel_size'].values / 2
    locus_end = locus_pos + loci_df['largest_nonzero_ls_kernel_size'].values / 2
    loci_within_other_loci = reduce(np.logical_and, [
        locus_chrom[loci_to_remove][:, None] == locus_chrom[loci_to_keep],
        locus_start[loci_to_remove][:, None] < locus_pos[loci_to_keep],
        locus_end[loci_to_remove][:, None] > locus_pos[loci_to_keep],
        type_order[loci_to_remove][:, None] == type_order[loci_to_keep],
    ])

    # Check if loci to remove are within centromeres/telomeres
    centro_telo_pos = np.concatenate([CENTROMERES.values.T.flatten(), len(CHROMS) * [0], CHROM_LENS.values])
    centro_telo_chrom = np.concatenate([2 * list(CENTROMERES.index), CHROMS, CHROM_LENS.index])
    assert len(centro_telo_pos) == len(centro_telo_chrom)
    locus_start = locus_pos - data_per_length_scale[('mid1', 'gain')]['loci_width'] * segment_size_dict['mid1'] / 2
    locus_end = locus_pos + data_per_length_scale[('mid1', 'gain')]['loci_width'] * segment_size_dict['mid1'] / 2

    loci_within_centro_telo = reduce(np.logical_and, [
        locus_chrom[loci_to_remove][:, None] == centro_telo_chrom,
        locus_start[loci_to_remove][:, None] < centro_telo_pos,
        locus_end[loci_to_remove][:, None] > centro_telo_pos,
    ])

    # Combine those two checks
    loci_to_remove_with_replacing_locus = loci_to_remove[
        np.logical_or(loci_within_other_loci.any(axis=1), loci_within_centro_telo.any(axis=1))]
    loci_df['prominence_within_other_locus'] = False
    loci_df.loc[loci_to_remove_with_replacing_locus, 'prominence_within_other_locus'] = True

    return loci_df['prominence_within_other_locus'].values


def selection_points_from_loci_df(loci_df, cur_chrom, ls_i):
    return [SelectionPoints(loci=[x]) for x in loci_df.query('chrom == @cur_chrom')[['pos', f'fit_{ls_i}']].values]


def full_selection_points_from_loci_df(loci_df):
    all_sp = {}
    for cur_chrom in CHROMS[:-1]:
        cur_loci = loci_df.query('chrom == @cur_chrom').sort_values('rank_on_chrom')
        all_sp[cur_chrom] = [[
            SelectionPoints(loci=[x]) for x in cur_loci[['pos', f'fit_{ls_i}']].values]
            for ls_i in range(8)]
    return all_sp


def calc_overlap_pairs(loci_1, loci_2):

    cur_dist = np.abs(loci_1['pos'].values[:, None] - loci_2['pos'].values)
    cur_overlaps = calc_overlaps(loci_1, loci_2)
    cur_dist[~cur_overlaps] = np.inf

    cur_dist_ = cur_dist.copy()

    cur_pairs = []
    while cur_dist_.min() < np.inf:
        min_i, min_j = np.unravel_index(np.argmin(cur_dist_), cur_dist_.shape)
        cur_dist_[min_i, :] = np.inf
        cur_dist_[:, min_j] = np.inf
        cur_pairs.append((min_i, min_j))
    return np.array(cur_pairs)


def resim_null_for_chrom_type(cur_chrom, cur_type, data_per_length_scale, output_dir,
                              N_random, n_iterations_optim, mode='random', overwrite=False, n_jobs=1):
    """Load-or-compute (and cache) the resimulation null for one (chrom, type). The cache key is
    shared by every fitness p-value consumer -- assign_p_values (below) and the per-chromosome
    scatter that warms it -- so warming the caches in parallel lets the combine's assign_p_values
    reuse the nulls without re-simulating. `data_per_length_scale` is one chromosome's dict."""
    from spice.tsg_og.p_values import p_value_using_resim
    cache = os.path.join(
        output_dir, 'p_values',
        f'{cur_chrom}_{cur_type}_N_random_{N_random}_N_optim_{n_iterations_optim}_mode_{mode}.pickle')
    if not overwrite and os.path.exists(cache):
        logger.info(f"Loading p-values for {cur_chrom} ({cur_type}) from cache")
        return open_pickle(cache)
    logger.info(f"Calculating p-value distribution for {cur_chrom} ({cur_type})")
    results = p_value_using_resim(
        cur_chrom=cur_chrom, cur_up_down='up' if cur_type == 'OG' else 'down', N_test=N_random,
        data_per_length_scale=data_per_length_scale, n_iterations_optim=n_iterations_optim,
        mode=mode, n_jobs=n_jobs)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    save_pickle(results, cache)
    return results


def assign_p_values(
    loci_df,
    N_random=10_000,
    n_iterations_optim=1_000,
    output_dir=None,
    data_per_length_scale=None,
    overwrite=False,
    statistic='added_events',
    mode='random',
    n_jobs=1,
):
    """Assign p-values to loci, either loading from cache or calculating from scratch.

    Args:
        loci_df: DataFrame with loci to assign p-values to
        N_random: Number of random simulations for p-value calculation
        n_iterations_optim: Number of optimization iterations
        output_dir: Directory to cache/load p-value results
        data_per_length_scale: Data per length scale (required if overwrite=True)
        overwrite: If True, recalculate p-values from scratch. If False, load from cache.
        mode: Resimulation-locus selection mode passed to p_value_using_resim ('random' or 'top')
        n_jobs: joblib workers for any resim that isn't already cached (default 1)
    """
    from spice.tsg_og.p_values import (
        get_actual_p_values_from_results, get_actual_p_values_per_ls_from_results)


    log_debug(logger, f'Assigning p-values to loci for {len(data_per_length_scale)} chromosomes with N_random={N_random}, n_iterations_optim={n_iterations_optim}, mode={mode}, overwrite={overwrite}')

    assert statistic in ['added_events', 'fitness'], 'Parameter `statistic` has to be either "added_events" or "fitness"'

    loci_df['p_value_raw'] = 1
    if statistic == 'fitness':
        for ls in LENGTH_SCALE_NAMES:
            loci_df[f'p_value_raw_{ls}'] = 1


    for cur_chrom in data_per_length_scale.keys():
        for cur_type in ['OG', 'TSG']:
            p_value_results = resim_null_for_chrom_type(
                cur_chrom, cur_type, data_per_length_scale[cur_chrom], output_dir,
                N_random, n_iterations_optim, mode=mode, overwrite=overwrite, n_jobs=n_jobs)

            # Apply p-values to loci dataframe
            cur_loci = loci_df.query('chrom == @cur_chrom and type == @cur_type')
            p_values = get_actual_p_values_from_results(cur_loci, p_value_results, N_random, statistic=statistic)
            loci_df.loc[cur_loci.index, 'p_value_raw'] = p_values

            if statistic == 'fitness':
                p_values_per_ls = get_actual_p_values_per_ls_from_results(cur_loci, p_value_results, N_random)
                for i, ls in enumerate(LENGTH_SCALE_NAMES):
                    loci_df.loc[cur_loci.index, f'p_value_raw_{ls}'] = p_values_per_ls[:, i]

    loci_df['p_value'] = false_discovery_control(loci_df['p_value_raw'].values)
    if statistic == 'fitness':
        loci_df[[f'p_value_{ls}' for ls in LENGTH_SCALE_NAMES]] = np.reshape(false_discovery_control(
            loci_df[[f'p_value_raw_{ls}' for ls in LENGTH_SCALE_NAMES]].values, axis=None), (-1, 4))

    return loci_df
