"""Add loci one at a time, highest fitness_statistic first, and measure fit improvement.

Standalone post-hoc diagnostic over an already-fit, combined loci set -- NOT part of the main
spice loci_detection / loci_assignment / components pipelines. Run via scripts/add_loci_one_by_one.py.

Three input modes, each reading only from the conventional on-disk layout its own pipeline writes
under `<results_dir>/<name>/...` -- nothing else is required to locate the data:
  loci_inference  -- final_loci_detection.tsv  (+ loci_of_selection/data_per_length_scale)
  loci_assignment -- final_loci_assignment.tsv (+ loci_of_selection/data_per_length_scale)
  components      -- components/components_<all|filtered>.tsv, components/fits/<label>/<chrom>/
                      selection_points.pickle, and whichever data_per_length_scale cache that
                      components run used (fresh/shared/saved alike)

For loci_inference/loci_assignment, selection points are derived straight from the loci_df's own
fitness_<ls>_<dir> columns (loci.full_selection_points_from_loci_df) -- there is no separate
selection-points pickle for those. components already saves a selection_points.pickle per
chromosome alongside its fit, so that is loaded directly instead.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from spice.utils import open_pickle, CALC_NEW
from spice.logging import log_debug, get_logger
from spice.length_scales import DEFAULT_SEGMENT_SIZE_DICT
from spice.tsg_og.simulation import SelectionPoints, copy_list_of_selection_points, convolution_simulation_per_ls
from spice.tsg_og.detection import calc_mse_loss, calc_within_ci_bootstrap, _optimize_selection_points
from spice.tsg_og.loci import full_selection_points_from_loci_df
from spice.tsg_og.permutation import fitness_statistic
from spice.cli_functions import _run_batch

logger = get_logger('add_loci_one_by_one')
CHROMS = ['chr' + str(x) for x in range(1, 23)] + ['chrX', 'chrY']
MODES = ('loci_inference', 'loci_assignment', 'components')


@CALC_NEW()
def add_loci_one_by_one(
    cur_chrom,
    chrom_loci_df,
    chrom_selection_points,
    data_per_length_scale,
    N_iterations_base=3_000,
    segment_size_dict=DEFAULT_SEGMENT_SIZE_DICT,
):
    """
    Add one chromosome's loci one at a time and re-measure the fit after each addition.

    Loci are added in decreasing order of `permutation.fitness_statistic` (the same statistic the
    fitness p-value ranks on) -- the most meaningful locus first -- so the resulting curve shows how
    quickly the fit saturates as weaker loci are included.

    Parameters
    ----------
    cur_chrom : str
        Chromosome to process.
    chrom_loci_df : pd.DataFrame
        This chromosome's rows only, already row-aligned with `chrom_selection_points[0]` (same
        order) -- see the `_load_*` loaders below, which build both together per mode.
    chrom_selection_points : list
        Per-length-scale (8) lists of single-locus SelectionPoints, aligned with `chrom_loci_df`.
    data_per_length_scale : dict
        This chromosome's data per length scale.
    N_iterations_base : int
        Base optimizer iteration count for the newly-added locus; scaled by sqrt(N_loci) as more
        loci accumulate, matching the schedule used elsewhere in detection.
    segment_size_dict : dict
        Per-length-scale segment sizes.

    Returns
    -------
    list of dict
        One entry per locus count (0..N), each with 'selection_points', 'conv', 'mse_loss',
        'within_ci' for the fit using only the N most meaningful loci.
    """
    log_debug(logger, f'Adding loci one by one for {cur_chrom}')

    assert all(x['chrom'] == cur_chrom for x in data_per_length_scale.values()), \
        f'Wrong data_per_length_scale for current chrom {cur_chrom}'
    assert len(chrom_loci_df) == len(chrom_selection_points[0]), (
        f'{cur_chrom}: {len(chrom_loci_df)} loci in loci_df vs '
        f'{len(chrom_selection_points[0])} selection points')

    # Most meaningful (highest fitness_statistic) locus first
    add_order = np.argsort(-fitness_statistic(chrom_loci_df))
    raw_selection_points = [[track[i] for i in add_order] for track in chrom_selection_points]

    up_down_order = [any(raw_selection_points[j][cluster_j][0].fitness > 0 for j in range(0, 8, 2))
                      for cluster_j in range(len(raw_selection_points[0]))]

    loci_one_by_one = []
    best_selection_points = None
    empty_conv = convolution_simulation_per_ls(cur_chrom, data_per_length_scale, best_selection_points)
    empty_loss = calc_mse_loss(data_per_length_scale, empty_conv)
    best_loss = empty_loss

    for N_loci in range(len(raw_selection_points[0]) + 1):
        log_debug(logger, f'Adding locus nr {N_loci} (out of {len(raw_selection_points[0])}) for {cur_chrom}')
        if N_loci > 0:
            new_selection_points = [[x[N_loci - 1]] for x in raw_selection_points]
            if best_selection_points is None:
                cur_selection_points = copy_list_of_selection_points(new_selection_points)
            else:
                cur_selection_points = copy_list_of_selection_points(
                    [list(x) + list(y) for x, y in zip(best_selection_points, new_selection_points)])
            for i in range(8):
                cur_selection_points[i][-1] = SelectionPoints(loci=[[cur_selection_points[i][-1][0].pos, 0]])
            cur_selection_points_per_cluster = list(zip(*cur_selection_points))
            assert len(cur_selection_points_per_cluster) == N_loci, (
                f'Expected {N_loci} clusters, got {len(cur_selection_points_per_cluster)}')

            base_conv = convolution_simulation_per_ls(cur_chrom, data_per_length_scale, cur_selection_points)
            base_loss = calc_mse_loss(data_per_length_scale, base_conv)
            assert base_loss == best_loss, f'Current loss {base_loss:.4e} is different from best loss {best_loss:.4e}'

            N_iterations = int(N_iterations_base * np.sqrt(N_loci + 1))
            optimized_selection_points_per_cluster, optim_loss, loss_over_time = _optimize_selection_points(
                N_iterations,
                cur_selection_points_per_cluster,
                data_per_length_scale,
                cur_chrom,
                best_loss=base_loss,
                show_progress=False,
                N_iterations_base=0,
                segment_size_dict=segment_size_dict,
                max_pos_change=0,
                up_down_order=up_down_order[:N_loci],
                allow_pos_change=False
            )

            if optim_loss < best_loss and len(loss_over_time) > 0:
                best_selection_points = list(zip(*optimized_selection_points_per_cluster))
                best_loss = optim_loss
            else:
                logger.warning(f'Optimization did not improve the loss at added locus {N_loci} '
                                f'(optim_loss: {optim_loss:.4e}, best_loss: {best_loss:.4e}). '
                                f'Setting fitness to zero for the new locus.')
                best_selection_points = copy_list_of_selection_points(cur_selection_points)

        cur_conv = convolution_simulation_per_ls(cur_chrom, data_per_length_scale, best_selection_points)
        cur_loss = calc_mse_loss(data_per_length_scale, cur_conv)
        cur_within_ci = calc_within_ci_bootstrap(data_per_length_scale, cur_conv)

        loci_one_by_one.append({
            'selection_points': best_selection_points,
            'conv': cur_conv,
            'mse_loss': cur_loss,
            'within_ci': cur_within_ci,
        })
        log_debug(logger, f'MSE loss: {cur_loss:.4e} - within CI: {np.mean(cur_within_ci):.2f}')

    log_debug(logger, f'Finished adding loci one by one for {cur_chrom}')
    return loci_one_by_one


# --------------------------------------------------------------------------- mode-specific loaders

def _results_root(config):
    return Path(config['directories']['results_dir']) / config['name']


def _per_chrom_from_loci_df(loci_df):
    """chrom -> (chrom_loci_df, chrom_selection_points), derived from loci_df's own fitness columns."""
    all_selection_points = full_selection_points_from_loci_df(loci_df)
    chroms = sorted(set(loci_df['chrom'].unique()) & set(all_selection_points), key=CHROMS.index)
    return {
        cur_chrom: (loci_df.query('chrom == @cur_chrom').sort_values('rank_on_chrom'),
                    all_selection_points[cur_chrom])
        for cur_chrom in chroms
    }


def _load_loci_inference(config):
    root = _results_root(config)
    loci_df_path = root / 'final_loci_detection.tsv'
    if not loci_df_path.is_file():
        raise FileNotFoundError(f"{loci_df_path} not found -- run `spice loci_detection` through "
                                 "the 'combine' step first.")
    loci_df = pd.read_csv(loci_df_path, sep='\t', index_col=0)
    per_chrom = _per_chrom_from_loci_df(loci_df)
    data_dir = root / 'loci_of_selection' / 'data_per_length_scale'
    output_dir = root / 'loci_of_selection' / 'detection' / 'add_loci_one_by_one'
    return per_chrom, data_dir, output_dir


def _load_loci_assignment(config):
    root = _results_root(config)
    loci_df_path = root / 'final_loci_assignment.tsv'
    if not loci_df_path.is_file():
        raise FileNotFoundError(f'{loci_df_path} not found -- run `spice loci_assignment` first.')
    loci_df = pd.read_csv(loci_df_path, sep='\t', index_col=0)
    per_chrom = _per_chrom_from_loci_df(loci_df)
    data_dir = root / 'loci_of_selection' / 'data_per_length_scale'
    output_dir = root / 'loci_of_selection' / 'assignment' / 'add_loci_one_by_one'
    return per_chrom, data_dir, output_dir


def _component_model_data_dir(components_dir):
    """Where <components_dir>'s run read data_per_length_scale from -- fresh, shared or saved."""
    fresh = components_dir / 'component_model' / 'data_per_length_scale'
    if fresh.is_dir():
        return fresh
    from spice.cohort_model import resolve as resolve_model_path
    frozen_config = yaml.safe_load((components_dir / 'config.yaml').read_text())
    inputs = frozen_config['input_files']
    if inputs.get('cohort_model_dir'):
        return resolve_model_path(frozen_config, inputs['cohort_model_dir']) / 'report' / 'data_per_length_scale'
    if inputs.get('component_model_dir'):
        return resolve_model_path(frozen_config, inputs['component_model_dir']) / 'data_per_length_scale'
    raise FileNotFoundError(f'Cannot locate the component reference model for {components_dir}')


def _load_components(config, label='filtered'):
    if label not in ('all', 'filtered'):
        raise ValueError(f"components_label must be 'all' or 'filtered', got {label!r}")
    components_dir = _results_root(config) / 'components'
    loci_df_path = components_dir / f'components_{label}.tsv'
    if not loci_df_path.is_file():
        raise FileNotFoundError(f'{loci_df_path} not found -- run `spice components` first.')
    loci_df = pd.read_csv(loci_df_path, sep='\t')
    data_dir = _component_model_data_dir(components_dir)
    output_dir = components_dir / 'add_loci_one_by_one' / label

    per_chrom = {}
    for cur_chrom in sorted(loci_df['chrom'].unique(), key=CHROMS.index):
        chrom_loci_df = loci_df.query('chrom == @cur_chrom').reset_index(drop=True)
        chrom_selection_points = open_pickle(
            str(components_dir / 'fits' / label / cur_chrom / 'selection_points.pickle'))
        if len(chrom_loci_df) != len(chrom_selection_points[0]) or not np.allclose(
                chrom_loci_df['pos'].to_numpy(dtype=float),
                [locus[0].pos for locus in chrom_selection_points[0]]):
            raise ValueError(f'{cur_chrom}: components_{label}.tsv row order does not match '
                              f"fits/{label}/{cur_chrom}/selection_points.pickle")
        per_chrom[cur_chrom] = (chrom_loci_df, chrom_selection_points)
    return per_chrom, data_dir, output_dir


_LOADERS = {
    'loci_inference': _load_loci_inference,
    'loci_assignment': _load_loci_assignment,
}


# ------------------------------------------------------------------------------------ orchestrator

def run_add_loci_one_by_one(
    mode,
    config,
    chroms=None,
    N_iterations_base=3_000,
    segment_size_dict=DEFAULT_SEGMENT_SIZE_DICT,
    cores=1,
    overwrite=False,
    components_label='filtered',
):
    """
    Run `add_loci_one_by_one` across every chromosome found for `mode`, in parallel.

    Everything besides `mode` (and, for `components`, which label to use) is read straight off
    disk via each pipeline's own conventional output layout under
    `<config.directories.results_dir>/<config.name>/...` -- see the module docstring.

    Parameters
    ----------
    mode : str
        One of 'loci_inference', 'loci_assignment', 'components'.
    config : dict
        The loaded SPICE config (`spice.config` after `spice.load_config(...)`).
    chroms : list of str, optional
        Restrict to these chromosomes; default is every chromosome the mode's own output covers.
    N_iterations_base : int
        Base optimizer iteration count for a newly-added locus; scaled by sqrt(N_loci).
    segment_size_dict : dict
        Per-length-scale segment sizes.
    cores : int
        Parallel workers across chromosomes; chromosomes are independent, so this scales like the
        other per-unit batch steps (see `cli_functions._run_batch`).
    overwrite : bool
        Recompute a chromosome's result even if its cache file already exists.
    components_label : str
        'all' or 'filtered' -- which components table/fit to use; only relevant for mode='components'.

    Returns
    -------
    (dict, pd.DataFrame)
        chrom -> `add_loci_one_by_one`'s per-locus-count list, and a flat summary table with
        columns ['chrom', 'n_loci', 'mse_loss', 'mean_within_ci'].
    """
    if mode not in MODES:
        raise ValueError(f'mode must be one of {MODES}, got {mode!r}')

    if mode == 'components':
        per_chrom, data_dir, output_dir = _load_components(config, label=components_label)
    else:
        per_chrom, data_dir, output_dir = _LOADERS[mode](config)

    if chroms is not None:
        missing = sorted(set(chroms) - set(per_chrom))
        if missing:
            raise ValueError(f'Chromosome(s) not present in {mode} output: {missing}')
        per_chrom = {c: per_chrom[c] for c in chroms}
    chroms = list(per_chrom)

    data_dir, output_dir = str(data_dir), str(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    def _run_one_chrom(cur_chrom):
        chrom_loci_df, chrom_selection_points = per_chrom[cur_chrom]
        data_per_length_scale = open_pickle(os.path.join(data_dir, f'{cur_chrom}.pickle'))
        return add_loci_one_by_one(
            cur_chrom=cur_chrom,
            chrom_loci_df=chrom_loci_df,
            chrom_selection_points=chrom_selection_points,
            data_per_length_scale=data_per_length_scale,
            N_iterations_base=N_iterations_base,
            segment_size_dict=segment_size_dict,
            calc_new_filename=os.path.join(output_dir, f'{cur_chrom}.pickle'),
            calc_new_force_new=overwrite,
        )

    results = _run_batch(chroms, cores, 'Add loci one by one', _run_one_chrom, logger)

    results_per_chrom = {}
    summary_rows = []
    for cur_chrom, result in zip(chroms, results):
        if isinstance(result, dict) and result.get('status') == 'failed':
            logger.error(f"add_loci_one_by_one failed for {cur_chrom}: {result['error']}")
            continue
        results_per_chrom[cur_chrom] = result
        for n_loci, step in enumerate(result):
            summary_rows.append({
                'chrom': cur_chrom,
                'n_loci': n_loci,
                'mse_loss': step['mse_loss'],
                'mean_within_ci': float(np.mean(step['within_ci'])),
            })

    summary_df = pd.DataFrame(summary_rows, columns=['chrom', 'n_loci', 'mse_loss', 'mean_within_ci'])
    summary_path = os.path.join(output_dir, 'summary.tsv')
    summary_df.to_csv(summary_path, sep='\t', index=False)
    logger.info(f'Saved add-loci-one-by-one summary ({len(summary_df)} rows across '
                f'{len(results_per_chrom)}/{len(chroms)} chromosomes) to {summary_path}')

    return results_per_chrom, summary_df
