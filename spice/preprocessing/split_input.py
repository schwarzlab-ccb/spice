import os
import sys
import pickle
from joblib import Parallel, delayed

import numpy as np
import pandas as pd
import fstlib

from spice.utils import save_pickle, add_filename_suffix

# Use importlib.resources for accessing package data (works with installed packages)
if sys.version_info >= (3, 9):
    from importlib.resources import files
else:
    try:
        from importlib_resources import files
    except ImportError:
        files = None
from spice.logging import log_debug, get_logger
from spice.data_loaders import resolve_copynumber_file, load_raw_copy_number_data, load_sv_data
from spice.event_inference.fst_assets import nowgd_fst, get_diploid_fsa, T_forced_WGD
from spice.event_inference.fsts import fsa_from_string
from spice.event_inference.events_from_graph import create_events_df_from_single_path_solution
from spice.event_inference.data_structures import ChromData
from spice.preprocessing.preprocessing import merge_neighbours_mod, get_or_infer_wgd_status, get_or_infer_xy_status
from spice import config


logger = get_logger('Split input')

def format_input(data, total_cn=False):
    log_debug(logger, 'Formatting input data')
    data['chrom'] = data['chrom'].astype(str).map(lambda x: x if x.startswith('chr') else 'chr'+x)

    # Cap copynumber at 8 and filter small segments
    if total_cn:
        n_capped = int((data['cn_a'] > 8).sum())
        log_debug(logger, f'Capping copy numbers at max value 8: {n_capped} entries are affected')
        data.loc[data['cn_a'] > 8, 'cn_a'] = 8
        if 'total_cn' in data.columns:
            data['total_cn'] = data['cn_a']
    else:
        log_debug(logger, f'Capping copy numbers at max value 8: {data.eval("cn_a > 8").sum()} entries are affected')
        data.loc[data['cn_a'] > 8, 'cn_a'] = 8
        data.loc[data['cn_b'] > 8, 'cn_b'] = 8
    return data


def split_alleles(data, total_cn=False, cn_columns=['cn_a', 'cn_b']):
    if total_cn:
        data_stacked = data.copy()
        data_stacked = data_stacked.rename(columns={'cn_a': 'cn'})
        data_stacked['allele'] = 'cn_a'
    else:
        data.columns.name = 'allele'
        data_stacked = data.set_index(['sample_id', 'chrom', 'start', 'end']).stack().to_frame('cn').reset_index().copy()
        assert len(data) == len(data_stacked)//2, (len(data), len(data_stacked), len(data_stacked)//2)
        assert data_stacked['cn'].sum() == data[cn_columns].sum().sum()
    data_stacked['cn'] = data_stacked['cn'].astype(int)
    data_stacked['id'] = data_stacked['sample_id'] + ':' + data_stacked['chrom'].astype(str) + ':' + data_stacked['allele']

    return data_stacked

def _prepare_split_inputs(name, keep_old=False, selected_ids=None):
    """Prepare data, directories, and lookup tables for splitting.

    Returns a context dict and a list of grouped chromosome dataframes.
    """
    results_events_dir = os.path.join(config['directories']['results_dir'], name, 'events')
    if not os.path.exists(results_events_dir):
        os.makedirs(results_events_dir)

    total_cn = config['params'].get('total_cn', False)

    copynumber_file = resolve_copynumber_file(return_raw=True)
    diploid_fsas = {b: get_diploid_fsa(total_copy_numbers=b) for b in [False, True]}

    log_debug(logger, f"Loading from input file: {copynumber_file}")
    if total_cn:
        data = load_raw_copy_number_data(copynumber_file, alleles=['total_cn'])
        data = data[['sample_id', 'chrom', 'start', 'end', 'total_cn']].copy()
        data['cn_a'] = data['total_cn']
        data['cn_b'] = 0
    else:
        data = load_raw_copy_number_data(copynumber_file)
        data = data[['sample_id', 'chrom', 'start', 'end', 'cn_a', 'cn_b']]
    if data['sample_id'].str.contains(':').any():
        raise ValueError('Sample IDs in input file cannot contain ":" character.')
    log_debug(logger, f'Found {data["sample_id"].nunique()} samples in input file.')
    # Resolve XY status (male samples). For haplotype-specific CN, we'll zero minor CN for chrX/chrY later.
    xy_status = get_or_infer_xy_status(data)
    xy_samples = xy_status.loc[xy_status.values].index.tolist()

    wgd_status = get_or_infer_wgd_status( data=data, total_cn=total_cn, )

    data = format_input(data, total_cn=total_cn)

    log_debug(logger, 'Merging neighbouring segments with same copy-numbers')
    data = merge_neighbours_mod(
        data, cn_columns=['cn_a', 'cn_b'], start_end_must_overlap=False)
    assert (data.reset_index().index == data.index).all(), 'index is messed up'

    log_debug(logger, 'Splitting input TSV')
    data = split_alleles(data, total_cn=total_cn, cn_columns=['cn_a', 'cn_b'])

    if selected_ids is not None:
        data = data.loc[data['id'].isin(selected_ids)]
        if len(data) == 0:
            raise ValueError('No data left after selecting IDs!')

    log_debug(logger, 'Second time merging neighbouring segments with same copy-numbers')
    data = data.sort_values(['id', 'start']).reset_index(drop=True)
    data['sample_id'] = data['sample_id'] + '_' + data['allele']
    data = merge_neighbours_mod(
        data, cn_columns=['cn'], start_end_must_overlap=False)
    data['sample_id'] = data['sample_id'].str.replace('_cn_a', '').str.replace('_cn_b', '')

    split_copynumber_file = add_filename_suffix(copynumber_file, '_split')
    log_debug(logger, f'Saving split input TSV to {split_copynumber_file}')
    data.to_csv(split_copynumber_file, sep='\t', index=False)

    # Load lookup tables from package resources
    if total_cn:
        lookup_table_single_solution = dict()
        lookup_table_multiple_solutions = dict()
    else:
        if files is None:
            raise FileNotFoundError("importlib.resources unavailable for lookup tables")
        try:
            single_content = files('spice').joinpath('objects', 'lookup_table_single_solution_full_paths.pickle').read_bytes()
            lookup_table_single_solution = pickle.loads(single_content)
            
            multiple_content = files('spice').joinpath('objects', 'lookup_table_multiple_solutions_full_paths.pickle').read_bytes()
            lookup_table_multiple_solutions = pickle.loads(multiple_content)
        except (TypeError, ImportError, AttributeError, FileNotFoundError) as exc:
            raise FileNotFoundError("Could not find lookup tables in spice/objects/") from exc

    if not keep_old:
        log_debug(logger, 'Deleting old files')
    directories = [f'{wgd}/{subdir}' for wgd in ['nowgd', 'wgd'] for subdir in ['chrom_data_full', 'chrom_data_large']]
    for directory in directories:
        directory = os.path.join(results_events_dir, directory)
        if not os.path.exists(directory):
            os.makedirs(directory)
        elif not keep_old:
            for file in os.listdir(directory):
                os.remove(os.path.join(directory, file))

    groupby = data.groupby(['sample_id', 'chrom', 'allele'])
    tasks = list(groupby)

    sv_data_file = config['input_files'].get('sv', None)
    sv_data = load_sv_data(sv_data_file)  # loads full table; filtering per-chrom happens in _process_group

    context = {
        'results_events_dir': results_events_dir,
        'total_cn': total_cn,
        'xy_samples': xy_samples,
        'wgd_status': wgd_status,
        'diploid_fsas': diploid_fsas,
        'lookup_table_single_solution': lookup_table_single_solution,
        'lookup_table_multiple_solutions': lookup_table_multiple_solutions,
        'copynumber_file': copynumber_file,
        'sv_data': sv_data,
    }

    return context, tasks


def _process_group(context, key, chrom_segments):
    """Process a single (sample, chrom, allele) group to produce ChromData artifacts."""

    def _neutral_value_key(wgd, total_cn, cur_allele, is_xy_sample, cur_chrom):
        if is_xy_sample and cur_chrom in ['chrX', 'chrY'] and cur_allele == 'cn_b':
            neutral_value = 0.0
        else:
            neutral_value = 1.0
        if total_cn and not (is_xy_sample and cur_chrom in ['chrX', 'chrY']):
            neutral_value *= 2.0
        if wgd:
            neutral_value *= 2.0
        return neutral_value

    (cur_sample, cur_chrom, cur_allele) = key
    results_events_dir = context['results_events_dir']
    total_cn = context['total_cn']
    xy_samples = context['xy_samples']
    wgd_status = context['wgd_status']
    diploid_fsas = context['diploid_fsas']
    lookup_table_single_solution = context['lookup_table_single_solution']
    lookup_table_multiple_solutions = context['lookup_table_multiple_solutions']
    copynumber_file = context['copynumber_file']
    sv_data = context['sv_data']

    cur_total_cn = total_cn and not (cur_sample in xy_samples and cur_chrom in ['chrX', 'chrY'])
    wgd = wgd_status.loc[cur_sample]
    fst = T_forced_WGD if wgd else nowgd_fst
    cur_id = f"{cur_sample}:{cur_chrom}:{cur_allele}"
    chrom_segments = chrom_segments.sort_values('start')
    chrom_string = ''.join(chrom_segments['cn'].values.astype(str))

    cur_neutral_value = _neutral_value_key(wgd, total_cn, cur_allele, cur_sample in xy_samples, cur_chrom)
    if (set(chrom_string) == {str(int(cur_neutral_value))}):
        log_debug(logger, f'{cur_id}: Copy-number profile = {chrom_string}. No events, skipping.')
        return ('none', cur_id)

    chrom_fsa = fsa_from_string(chrom_string)
    chrom_dist = int(float(fstlib.score(fst, diploid_fsas[cur_total_cn], chrom_fsa)))
    # max is required because cn_profile "0" should have dist 1 even in the case of WGD. Also all diploid profiles are filtered out
    n_events = max(1, chrom_dist - int(wgd))
    if cur_total_cn and (chrom_segments['cn'].values==0).all():
        n_events = 2

    if chrom_dist > config['params']['dist_limit']:
        log_debug(logger, f'{cur_id}: {chrom_string} - {chrom_dist}: too many events, skipping (event limit at {config["params"]["dist_limit"]})')
        return ('none', cur_id)

    # Determine if any SV breakpoints coincide with this chromosome's segment boundaries.
    # If so, skip the lookup tables — the full pipeline must run to apply SV constraints.
    chrom_id = f"{cur_sample}:{cur_chrom}"
    cur_sv_data = sv_data.query('chrom_id == @chrom_id') if sv_data is not None else None
    has_sv_overlap = False
    if cur_sv_data is not None and len(cur_sv_data) > 0:
        sv_threshold = config['params']['sv_matching_threshold']
        relevant_svs = cur_sv_data.query('svclass == "DUP" or svclass == "DEL"')
        sv_size_filter = config['params'].get('sv_size_filter', 0)
        if sv_size_filter:
            relevant_svs = relevant_svs.query('(end - start) >= @sv_size_filter')
        if len(relevant_svs) > 0:
            breakpoints = np.unique(np.concatenate([
                chrom_segments['start'].values, chrom_segments['end'].values]))
            has_sv_overlap = (
                np.any(np.abs(relevant_svs['start'].values[:, None] - breakpoints) < sv_threshold) or
                np.any(np.abs(relevant_svs['end'].values[:, None] - breakpoints) < sv_threshold)
            )
    if has_sv_overlap:
        log_debug(logger, f'{cur_id}: SV data overlaps with breakpoints, skipping lookup table')

    if (wgd, chrom_string) in lookup_table_single_solution:
        cur = create_events_df_from_single_path_solution(
            lookup_table_single_solution[(wgd, chrom_string)], cur_id, chrom_segments=chrom_segments,
            create_full=True, calc_telomere_bound=False)
        save_pickle(
            cur, os.path.join(str(results_events_dir), 'wgd' if wgd else 'nowgd', 'full_paths_single_solution', f"{cur_id}.pickle"))
        log_debug(logger, f'{cur_id}: Found in lookup table! Copy-number profile = {chrom_string}. Nr of events = {n_events}. Saved to {"wgd" if wgd else "nowgd"}/full_paths_single_solution')
        return ('single', cur_id)
    elif not has_sv_overlap and (wgd, chrom_string) in lookup_table_multiple_solutions:  # skip if SVs constrain the solution
        cur = lookup_table_multiple_solutions[(wgd, chrom_string)]
        assert getattr(cur, '__class__', None).__name__ == 'FullPaths', type(cur)
        assert cur.is_wgd == wgd, (cur.is_wgd, wgd)
        cur = cur._replace(id=cur_id, sample=cur_sample, chrom=cur_chrom, allele=cur_allele)
        save_pickle(cur,
                    os.path.join(str(results_events_dir), 'wgd' if wgd else 'nowgd', 'full_paths_multiple_solutions', f"{cur_id}.pickle"))
        log_debug(logger, f'{cur_id}: Found in lookup table! Copy-number profile = {chrom_string}. Nr of events = {n_events}. Saved to {"wgd" if wgd else "nowgd"}/full_paths_multiple_solutions')
        return ('multiple', cur_id)

    chrom_data = ChromData(id=cur_id, sample=cur_sample, chrom=cur_chrom, allele=cur_allele, 
                            cn_profile=chrom_segments['cn'].values, string=chrom_string, dist=chrom_dist,
                            n_events=n_events, has_wgd=wgd, copynumber_file=copynumber_file)

    assert config['params']['full_path_high_mem_dist_limit'] <= config['params']['full_path_dist_limit']

    if (chrom_dist - int(wgd)) <= config['params']['full_path_dist_limit']:
        log_debug(logger, f'{cur_id}: Copy-number profile = {chrom_string}. Nr of events = {n_events}. Saved to {"events/wgd" if wgd else "events/nowgd"}/chrom_data_full')
        save_pickle(chrom_data, os.path.join(str(results_events_dir), 'wgd' if wgd else 'nowgd', 'chrom_data_full', f"{cur_id}.pickle"))
        if (chrom_dist - int(wgd)) > config['params']['full_path_high_mem_dist_limit']:
            save_pickle(None, os.path.join(str(results_events_dir), 'wgd' if wgd else 'nowgd', 'chrom_data_use_high_memory', f"{cur_id}.placeholder"))
        return ('full', cur_id)
    else:
        log_debug(logger, f'{cur_id}: Copy-number profile = {chrom_string}. Nr of events = {n_events}. Saved to {"events/wgd" if wgd else "events/nowgd"}/chrom_data_large')
        save_pickle(chrom_data, os.path.join(str(results_events_dir), "wgd" if wgd else "nowgd", 'chrom_data_large', f"{cur_id}.pickle"))
        return ('large', cur_id)


def split_tsv_file(name, keep_old=False, cores=None, selected_ids=None):
    """Split input TSV into per-allele chrom data, optionally in parallel."""
    context, tasks = _prepare_split_inputs(name, keep_old=keep_old, selected_ids=selected_ids)

    number_of_ids = len(tasks)
    counts = {'full': 0, 'large': 0, 'single': 0, 'multiple': 0, 'none': 0}

    if cores is None or cores <= 1:
        # Serial processing with progress logging
        for i, (key, chrom_dat) in enumerate(tasks):
            logger.info(f'{i+1} / {number_of_ids} finished ({100*i/number_of_ids:.1f}%) - {key[0]}:{key[1]}:{key[2]}')
            kind, _ = _process_group(context, key, chrom_dat)
            counts[kind] = counts.get(kind, 0) + 1
    else:
        # Parallel processing (use threads to reduce process startup overhead; IO-bound saves benefit)
        results = Parallel(n_jobs=cores, prefer='threads')(
            delayed(_process_group)(context, key, chrom_dat) for (key, chrom_dat) in tasks
        )
        for kind, _ in results:
            counts[kind] = counts.get(kind, 0) + 1

    n_full = counts.get('full', 0)
    n_large = counts.get('large', 0)
    logger.info(f'Created a total of {n_full+n_large} chrom_data files: {n_full} in full and {n_large} in large.')
