'''
All the cnsistent functions are adapted from commit e61fbd3ca15cfc9d6459f873dc6ee05b5f45d0d2 (v0.4)
'''
import os
import contextlib

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from spice import data_loaders, config, directories
from spice.logging import get_logger, log_debug

logger = get_logger('preprocessing')


def fill_gaps_cnsistent(cns_df, print_info=False):
    # Iterate over the rows
    new_rows = []
    for i in range(len(cns_df) - 1):
        # Check if the next row has the same 'sample_id' and 'chrom'
        if (
            cns_df.at[i, "sample_id"] == cns_df.at[i + 1, "sample_id"]
            and cns_df.at[i, "chrom"] == cns_df.at[i + 1, "chrom"]
        ):
            # Calculate the range
            range_start = cns_df.at[i, "end"]
            range_end = cns_df.at[i + 1, "start"]
            # If the range is greater than 1, add a new row
            if range_end > range_start:
                new_rows.append(
                    {
                        "sample_id": cns_df.at[i, "sample_id"],
                        "chrom": cns_df.at[i, "chrom"],
                        "start": range_start,
                        "end": range_end,
                    }
                )

    if len(new_rows) == 0:
        if print_info:
            print(f"No gaps found.")
        return cns_df.copy()
    else:
        # Concatenate the cns_dfs
        if print_info:
            print(f"Filling {len(new_rows)} gaps.")
        new_cns_df = pd.DataFrame(new_rows, columns=cns_df.columns)
        res_df = pd.concat([cns_df, new_cns_df])
        res_df.sort_values(by=["sample_id", "chrom", "start"], inplace=True, ignore_index=True)
        return res_df


def add_tails_cnsistent(cns_df, chr_lengths, print_info=False):
    grouped = cns_df.groupby(["sample_id", "chrom"]).agg({"start": "min", "end": "max"})
    grouped = grouped.rename(columns={"start": "min_start", "end": "max_end"})
    grouped = grouped.reset_index()
    missing_ranges = []
    for _, row in grouped.iterrows():
        if row.min_start > 0:
            missing_ranges.append(
                {
                    "sample_id": row.sample_id,
                    "chrom": row.chrom,
                    "start": 0,
                    "end": row.min_start
                }
            )
        if row.max_end < chr_lengths[f"{row.chrom}"]:
            missing_ranges.append(
                    {
                        "sample_id": row.sample_id,
                        "chrom":  row.chrom,
                        "start": row.max_end,
                        "end": chr_lengths[str(row.chrom)]
                    }
            )

    if len(missing_ranges) == 0:
        if print_info:
            print(f"No missing ends found.")
        return cns_df.copy()
    else:
        if print_info:
            print(f"Adding {len(missing_ranges)} missing ends")
        new_cns_df = pd.DataFrame(missing_ranges, columns=cns_df.columns)
        res_df = pd.concat([cns_df, new_cns_df])
        res_df.sort_values(
            by=["sample_id", "chrom", "start"], inplace=True, ignore_index=True
        )
        return res_df


def fill_gaps_cnsistent_wrapper(data, print_info=False):
    '''adjusted from cns.process.pipelines.main_fill'''
    chrom_lengths_dict = data_loaders.load_chrom_lengths().to_dict()
    data = data.reset_index(drop=True)
    cna_tailed_df = add_tails_cnsistent(data, chrom_lengths_dict, print_info=print_info)
    cna_filled_df = fill_gaps_cnsistent(cna_tailed_df, print_info=print_info)
    assert cna_filled_df['chrom'].str.startswith('chr').all()

    return cna_filled_df


def _are_mergeable_mod(a, b, cn_columns=('major_cn', 'minor_cn'), start_end_must_overlap=False):
    '''modified from cnsistent'''
    return (
        a.sample_id == b.sample_id
        and a.chrom == b.chrom
        and ((not start_end_must_overlap) or (a.end == b.start))
        and all([(a[col] == b[col]) or (np.isnan(a[col]) and np.isnan(b[col])) for col in cn_columns])
    )


def merge_neighbours_mod(cna_df, cn_columns=['major_cn', 'minor_cn'], start_end_must_overlap=False):
    '''modified from cnsistent'''
    assert (cna_df.reset_index().index == cna_df.index).all(), 'index is messed up'
    res_df = cna_df.copy()
    idx_to_remove = []

    for i, (index, row) in enumerate(res_df.iterrows()):
        if i != 0 and _are_mergeable_mod(prev, row, cn_columns, start_end_must_overlap):
            idx_to_remove.append(i - 1)
            res_df.at[index, "start"] = prev.start
            row.start = prev.start  # update the comparison copy too
        prev = row

    # remove from cna_df where idx_to_remove is in the index
    res_df = res_df.drop(res_df.index[idx_to_remove])
    res_df.sort_values(
        by=["sample_id", "chrom", "start"], inplace=True, ignore_index=True
    )
    log_debug(logger, f'Merged {len(idx_to_remove)} segments')
    return res_df


def fill_telomere_nans(data, cn_columns=['major_cn', 'minor_cn']):

    nan_index = data.loc[data[cn_columns[0]].isna()].index.values
    if len(nan_index) == 0:
        return data

    data['sample_chrom_id'] = data['sample_id'] + ':' + data['chrom']

    nan_telomere_bound = pd.DataFrame(False, index=nan_index, columns=['left', 'right', 'any'])
    nan_index_l = np.setdiff1d(nan_index, 0)
    nan_index_r = np.setdiff1d(nan_index, len(data)-1)
    nan_telomere_bound.loc[nan_index_l, 'left'] = ~(data.loc[nan_index_l, 'sample_chrom_id'].values == data.loc[nan_index_l-1, 'sample_chrom_id'].values)
    nan_telomere_bound.loc[nan_index_r, 'right'] = ~(data.loc[nan_index_r, 'sample_chrom_id'].values == data.loc[nan_index_r+1, 'sample_chrom_id'].values)
    if 0 in nan_index:
        nan_telomere_bound.loc[0, 'left'] = True
    if len(data)-1 in nan_index:
        nan_telomere_bound.loc[len(data)-1, 'right'] = True
    nan_telomere_bound['any'] = nan_telomere_bound['left'] | nan_telomere_bound['right']
    assert not np.logical_and(nan_telomere_bound['left'], nan_telomere_bound['right']).any(), data.loc[nan_telomere_bound.query('left & right').index]

    nan_index_to_merge_left = data.loc[nan_telomere_bound.query('left').index].query('(end - start) < 1e6').index.values
    nan_index_to_merge_right = data.loc[nan_telomere_bound.query('right').index].query('(end - start) < 1e6').index.values
    for cn_col in cn_columns:
        data.loc[nan_index_to_merge_left, cn_col] = data.loc[nan_index_to_merge_left+1, cn_col].values
        data.loc[nan_index_to_merge_right, cn_col] = data.loc[nan_index_to_merge_right-1, cn_col].values

    data = data.drop(columns=['sample_chrom_id'])

    return data


def get_breaks_mod(cna):
    '''modified from cnsistent'''
    assert cna['chrom'].nunique() == 1
    return {cna['chrom'].values[0]: np.sort(np.unique((cna[['start', 'end']].values).flatten()))}


def infer_wgd_status(data, results_dir=None, plot=False, method='major_cn', total_cn=False):
    log_debug(logger, 'Inferring WGD status from copy number data')
    _data = data.copy()

    _data[['major_cn', 'minor_cn']] = np.sort(_data[['cn_a', 'cn_b']].values, axis=1)[:, ::-1]
    _data['width'] = _data['end'] - _data['start']
    if method == 'major_cn':
        if total_cn:
            raise ValueError('Method "major_cn" is not compatible with total_cn=True, since it relies on major_cn and minor_cn columns. Please set total_cn=False or choose a different method.')
        _data['major_cn_above_equal_2_weighted'] = _data.eval('(major_cn >= 2) * width')
        _data['major_minus_minor_cn'] = _data['major_cn'] - _data['minor_cn']
        _data['major_minus_minor_cn_weighted'] = _data['major_minus_minor_cn'] * _data['width']
        major_minor_cn = _data.groupby('sample_id')[['major_cn_above_equal_2_weighted', 'major_minus_minor_cn_weighted', 'width']].sum()
        major_minor_cn['fraction_major_cn_above_equal_2'] = major_minor_cn['major_cn_above_equal_2_weighted'] / major_minor_cn['width']
        major_minor_cn['wgd'] = major_minor_cn['fraction_major_cn_above_equal_2'] > 0.5
        wgd_status = major_minor_cn['wgd']

        if plot:
            import seaborn as sns
            assert results_dir is not None, 'results_dir needs to be provided if plot=True'
            major_minor_cn['mean_major_minus_minor_cn'] = major_minor_cn['major_minus_minor_cn_weighted'] / major_minor_cn['width']
            plt.figure(figsize=(8, 8))
            sns.scatterplot(data=major_minor_cn, x='fraction_major_cn_above_equal_2', y='mean_major_minus_minor_cn', hue='wgd')
            plt.legend(title='WGD status')
            plt.axvline(0.5, color='grey', linestyle='--')
            plt.xlabel('Fraction of genome with major CN >= 2')
            plt.ylabel('Mean (major CN - minor CN)')
            plt.savefig(os.path.join(results_dir, f'WGD_status_{method}.png'))
            plt.close()

    elif method == 'ploidy_loh':
        if total_cn:
            _data['weighted_cn'] = _data['total_cn'] * _data['width']
        else:
            _data['weighted_cn'] = (_data['major_cn'] + _data['minor_cn']) * _data['width']
        ploidy = (_data.groupby(['sample_id'])['weighted_cn'].sum() / _data.groupby(['sample_id'])['width'].sum())
        _data['is_loh'] = _data['minor_cn'] == 0
        _data['loh_width'] = _data['is_loh'] * _data['width']
        loh_fraction = (_data.groupby(['sample_id'])['loh_width'].sum() / _data.groupby(['sample_id'])['width'].sum())
        assert loh_fraction.min() >= 0 and loh_fraction.max() <= 1, loh_fraction.agg(['min', 'max'])
        ploidy_loh_fraction = pd.concat([ploidy.to_frame('ploidy'), loh_fraction.to_frame('loh_fraction')], axis=1)
        ploidy_loh_fraction['wgd'] = ploidy_loh_fraction.eval('(2.9 - 2 * loh_fraction) < ploidy')

        if plot and results_dir is not None:
            import seaborn as sns
            plt.figure(figsize=(8, 8))
            sns.scatterplot(data=ploidy_loh_fraction, x='loh_fraction', y='ploidy', hue='wgd')
            linex = np.linspace(loh_fraction.min(), loh_fraction.max())
            linea = -2
            linec = 2.9
            liney = linea * linex + linec
            plt.plot(linex, liney, '--', color='grey')
            plt.legend(title='WGD status')
            plt.xlabel('Fraction of LOH')
            plt.ylabel('Ploidy')
            plt.savefig(os.path.join(results_dir, f'WGD_status_{method}.png'))
            plt.close()

        wgd_status = ploidy_loh_fraction['wgd']
    else:
        raise ValueError(f'Unknown method {method} for WGD inference')
    return wgd_status


def get_or_infer_wgd_status(data, total_cn=False):
    """Return WGD status per sample either from file or by inference.

    - If `input_files.wgd_status` is provided and non-empty, load it (expects a
      TSV with index as sample_id and a boolean `wgd` column).
    - Otherwise, infer using `infer_wgd_status` with the method from
      `params.wgd_inference_method`.
    - Logs a summary of counts via the module's logger.
    """

    wgd_status_file = config['input_files'].get('wgd_status', None)
    if wgd_status_file is not None and wgd_status_file != 'None' and wgd_status_file != '':
        wgd_status = pd.read_csv(wgd_status_file, sep='\t', index_col=0, dtype={'wgd': bool})['wgd']
        missing_samples = set(data['sample_id'].unique()) - set(wgd_status.index)
        if len(missing_samples) > 0:
            raise ValueError(f'Samples missing in WGD status file: {missing_samples}')
    else:
        wgd_status = infer_wgd_status(
            data,
            results_dir=directories['results_dir'],
            plot=False,
            method=config['params'].get('wgd_inference_method', 'major_cn'),
            total_cn=total_cn,
        )
    log_debug(logger, f'Inferred WGD status for {len(wgd_status)} samples. Found {wgd_status.sum()} WGDs and {len(wgd_status)-wgd_status.sum()} non-WGDs.')
    log_debug(logger, f'WGD status: {wgd_status}')
    return wgd_status


def get_or_infer_xy_status(data):
    """Return XY (male) status per sample either from file or by inference.

    - If `input_files.xy_status` is provided and non-empty, load it (expects a TSV
      with index as sample_id and a boolean `xy` column where True indicates male/XY).
    - Otherwise, infer: a sample is XY if any segments exist on chromosome 'chrY'.
    - Logs a summary of counts via the module's logger.
    """
    xy_file = config['input_files'].get('xy_status', None)
    if xy_file is not None and xy_file != 'None' and xy_file != '':
        xy_status = pd.read_csv(xy_file, sep='\t', index_col=0, dtype={'xy': bool})['xy']
        missing_samples = set(data['sample_id'].unique()) - set(xy_status.index)
        if len(missing_samples) > 0:
            raise ValueError(f'Samples missing in XY status file: {missing_samples}')
    else:
        xy_status = pd.DataFrame({'xy': False},
                                 index=data['sample_id'].unique()).astype(bool)['xy']
        xy_samples = (data.query('chrom == "chrY"')
                      .groupby('sample_id')[['cn_a', 'cn_b']].max().max(axis=1).to_frame('max_cn')
                      .query('max_cn > 0').index
        )
        xy_status.loc[xy_samples] = True

    n_xy = int(xy_status.sum())
    log_debug(logger, f'XY status resolved for {len(xy_status)} samples. Found {n_xy} XY and {len(xy_status)-n_xy} XX.')
    log_debug(logger, f'XY status: {xy_status}')
    return xy_status


def main_aggregate_quiet(*args, **kwargs):
    '''Wrapper around consistent's main_aggregate to suppress the print statements'''
    try:
        from cns import main_aggregate
    except ModuleNotFoundError:
        raise ModuleNotFoundError('CNSistent is not installed. Please install using "pip install CNSistent" in order to use the preprocessing functions.')
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        return main_aggregate(*args, **kwargs)