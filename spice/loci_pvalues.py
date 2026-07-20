"""Fitness p-value helpers for per-chromosome loci detection.

Wired into `spice loci_detection`: each per-chromosome run emits a raw (pre-FDR) p-value part
(`compute_chrom_parts`), and the combine step joins the parts by (chrom, rank_on_chrom) and applies
one global BH-FDR (`merge_parts`). A "track" is a loci `type` -- OG (gains) or TSG (losses). The
actual p-value machinery lives in spice.tsg_og.p_values (null resim, empirical + GPD tail); this
module is orchestration only.
"""
import os
import glob
import pandas as pd
from scipy.stats import false_discovery_control

from spice.utils import open_pickle, save_pickle
from spice.tsg_og.p_values import p_value_using_resim, get_actual_p_values_from_results
from spice.tsg_og.loci import create_loci_df

TRACKS = ('OG', 'TSG')
_TYPE_TO_UPDOWN = {'OG': 'up', 'TSG': 'down'}


def _null_for_track(cur_chrom, cur_type, data_per_length_scale, N_random, n_iterations_optim,
                    cache_dir=None, overwrite=False, mode='random', n_jobs=1):
    """resim null for one (chrom, type), cached under cache_dir/p_values/ (statistic-agnostic:
    each iteration records both added_events and fitness_stat). `mode` ('random'|'top') selects the
    resim-locus strategy and is part of the cache key so top/random nulls don't collide."""
    cache = (os.path.join(cache_dir, 'p_values',
             f'{cur_chrom}_{cur_type}_N_random_{N_random}_N_optim_{n_iterations_optim}_mode_{mode}.pickle')
             if cache_dir else None)
    if cache and os.path.exists(cache) and not overwrite:
        return open_pickle(cache)
    results = p_value_using_resim(
        cur_chrom=cur_chrom, cur_up_down=_TYPE_TO_UPDOWN[cur_type], N_test=N_random,
        data_per_length_scale=data_per_length_scale, n_iterations_optim=n_iterations_optim,
        mode=mode, n_jobs=n_jobs, skip_tqdm=True)
    if cache:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        save_pickle(results, cache)
    return results


def compute_track(loci_df, cur_chrom, cur_type, data_per_length_scale, N_random,
                  n_iterations_optim=1000, statistics=('fitness',), methods=('empirical', 'gpd'),
                  cache_dir=None, overwrite=False, mode='random', n_jobs=1):
    """Raw (pre-FDR) p for the loci of one (chrom, type). Returns a DataFrame indexed like the
    matching rows of loci_df, one column per (statistic, method): '<statistic>_<method>_raw'."""
    cur = loci_df.query('chrom == @cur_chrom and type == @cur_type')
    out = pd.DataFrame(index=cur.index)
    if len(cur) == 0:
        return out
    results = _null_for_track(cur_chrom, cur_type, data_per_length_scale, N_random,
                              n_iterations_optim, cache_dir, overwrite, mode=mode, n_jobs=n_jobs)
    for stat in statistics:
        for meth in methods:
            out[f'{stat}_{meth}_raw'] = get_actual_p_values_from_results(
                cur, results, N_random, statistic=stat, method=meth)
    return out


def compute_chrom_parts(cur_chrom, loci_results_dir, processed_events, N_random,
                        n_iterations_optim=1000, statistics=('fitness',), methods=('empirical', 'gpd'),
                        overwrite=False, mode='random', n_jobs=1):
    """Per-chromosome raw (pre-FDR) p-value part. Rebuilds this chromosome's loci_df from its
    detection cache with the same create_loci_df combine uses (so the (chrom, rank_on_chrom) keys and
    the fitness columns line up), computes raw p for both tracks, and writes
    loci_results_dir/p_values/parts/{chrom}.tsv keyed by (chrom, rank_on_chrom, type). `processed_events`
    is accepted for signature parity but not needed here (the fitness statistic reads only the
    fitness_* columns; no added_events)."""
    det = os.path.join(loci_results_dir, 'detection', cur_chrom)
    # The fitness p-value needs a COMPLETED detection run: final_selection_points.pickle and
    # final_loci_widths.pickle are written only by the terminal steps (final_limiting / final_loci_widths).
    # A partial `--loci-steps` (a single/early step) leaves them absent; fail with an actionable message
    # instead of an opaque FileNotFoundError from open_pickle below.
    missing = [f for f in ('final_selection_points.pickle', 'final_loci_widths.pickle')
               if not os.path.exists(os.path.join(det, f))]
    if missing:
        raise FileNotFoundError(
            f"Cannot compute the fitness p-value for {cur_chrom}: missing {missing} in {det}. "
            f"These are produced only by a complete detection run -- rerun loci detection with "
            f"--loci-steps default (or a '<step>+' selection reaching the final steps), or set "
            f"calculate_p_value: false in the loci config to skip the p-value.")
    sp = {cur_chrom: open_pickle(os.path.join(det, 'final_selection_points.pickle'))}
    widths = {cur_chrom: open_pickle(os.path.join(det, 'final_loci_widths.pickle'))}
    dpls = open_pickle(os.path.join(loci_results_dir, 'data_per_length_scale', f'{cur_chrom}.pickle'))

    loci_df = create_loci_df(sp, widths, nr_stds_widths=2, min_widths_is_small_kernel=True)

    raw = [compute_track(loci_df, cur_chrom, typ, dpls, N_random, n_iterations_optim,
                         statistics=statistics, methods=methods, cache_dir=loci_results_dir,
                         overwrite=overwrite, mode=mode, n_jobs=n_jobs) for typ in TRACKS]
    part = loci_df[['chrom', 'rank_on_chrom', 'type']].join(pd.concat(raw))

    out_dir = os.path.join(loci_results_dir, 'p_values', 'parts')
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f'{cur_chrom}.tsv')
    part.to_csv(out, sep='\t', index=False)
    return out


def merge_parts(loci_df, loci_results_dir, statistics=('fitness',), methods=('empirical', 'gpd')):
    """Join the per-chromosome raw-p parts onto loci_df by (chrom, rank_on_chrom). For each
    (statistic, method) it keeps the raw p-value as p_<statistic>_<method> and adds the global
    BH-FDR q-value as q_<statistic>_<method>. Drops the intermediate *_raw columns. Leaves loci_df
    untouched if no parts are present."""
    files = sorted(glob.glob(os.path.join(loci_results_dir, 'p_values', 'parts', '*.tsv')))
    if not files:
        return loci_df
    raw = pd.concat([pd.read_csv(f, sep='\t') for f in files], ignore_index=True)
    rawcols = [f'{s}_{m}_raw' for s in statistics for m in methods if f'{s}_{m}_raw' in raw.columns]
    loci_df = loci_df.merge(raw[['chrom', 'rank_on_chrom'] + rawcols],
                            on=['chrom', 'rank_on_chrom'], how='left')
    for s in statistics:
        for m in methods:
            col = f'{s}_{m}_raw'
            if col in loci_df.columns:
                ok = loci_df[col].notna()
                loci_df[f'p_{s}_{m}'] = loci_df[col]                     # raw (pre-FDR) p-value
                loci_df.loc[ok, f'q_{s}_{m}'] = false_discovery_control(  # global BH-FDR q-value
                    loci_df.loc[ok, col].to_numpy())
    return loci_df.drop(columns=rawcols)
