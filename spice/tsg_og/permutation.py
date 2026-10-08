"""Positional-permutation null for the fitness p-value.

Permute event positions in the real cohort, run the same
detection on the result, and pool the loci it finds. Null loci are then produced by the same cascade
as the observed ones. Rotate raw events within their original chromosome arms, then apply the observed
preprocessing filters in the same order as the validated production null.

See docs/PERMUTATION_NULL.MD in the pipeline repo for the derivation and the measured calibration.
"""
import numpy as np
import pandas as pd

from spice import data_loaders
from spice.length_scales import LENGTH_SCALE_NAMES
from spice.logging import get_logger, log_debug

logger = get_logger('spice.permutation')

#: pooled-null filename written by `spice permute --pool` and read by loci detection
NULL_FILENAME = 'permutation_null.tsv'
#: per-unit filename template (scatter selection uses --index, not --seed)
UNIT_TEMPLATE = 'permutation_unit_s{seed}_{chrom}.tsv'
DEFAULT_K = 16
STRATEGIES = ('zpool_chrom',)
SCORING_METHODS = ('combined_fitness',)
PERMUTE_MODES = ('rotate',)


# --------------------------------------------------------------------------------------- statistic

def fitness_per_ls(loci_df):
    """Per-locus, per-length-scale fitness, direction-matched (OG=gain / TSG=loss) and clipped at 0.

    The clip matters: the optimizer's sign clamp holds opposite-direction fitness at <= 0, so an
    unclipped mean would let the wrong direction drag the statistic down.
    """
    if not len(loci_df):
        return np.zeros((0, len(LENGTH_SCALE_NAMES)))
    gain = (loci_df['type'].to_numpy() == 'OG')
    out = np.empty((len(loci_df), len(LENGTH_SCALE_NAMES)))
    for j, ls in enumerate(LENGTH_SCALE_NAMES):
        g = loci_df[f'fitness_{ls}_gain'].to_numpy(float)
        l = loci_df[f'fitness_{ls}_loss'].to_numpy(float)
        out[:, j] = np.where(gain, g, l)
    return np.maximum(0.0, out)


def fitness_statistic(loci_df):
    """Mean same-direction fitness; retained for detection and fit summaries."""
    return fitness_per_ls(loci_df).mean(axis=1)


def scoring_per_ls(loci_df, method='combined_fitness'):
    """Same-direction positive fitness plus opposite-direction negative magnitude."""
    if method not in SCORING_METHODS:
        raise ValueError(f'Unknown p_values_method: {method}')
    same = fitness_per_ls(loci_df)
    if not loci_df['type'].isin(['OG', 'TSG']).all():
        raise ValueError('Unknown locus direction')
    gain = loci_df['type'].to_numpy() == 'OG'
    opposite = np.column_stack([np.where(gain, loci_df[f'fitness_{ls}_loss'],
                                        loci_df[f'fitness_{ls}_gain']) for ls in LENGTH_SCALE_NAMES])
    if not np.isfinite(same).all() or not np.isfinite(opposite).all():
        raise ValueError('Nonfinite fitness in combined score')
    return same + np.maximum(-opposite, 0.)


def scoring_statistic(loci_df, method='combined_fitness'):
    scoring_per_ls(loci_df, method)  # Validate direction and finite signed fitness.
    # Match the validated A+B definition, including its floating-point order.
    gain = loci_df['type'].to_numpy() == 'OG'
    opposite = np.column_stack([np.where(gain, loci_df[f'fitness_{ls}_loss'],
                                        loci_df[f'fitness_{ls}_gain']) for ls in LENGTH_SCALE_NAMES])
    return fitness_statistic(loci_df) + np.maximum(-opposite, 0.).mean(axis=1)


def validate_null_scoring(null_df, method):
    if method not in SCORING_METHODS:
        raise ValueError(f'Unknown p_values_method: {method}')
    if 'p_values_method' not in null_df:
        raise ValueError('Untagged null cannot score combined_fitness; re-pool signed null loci')
    elif not null_df.p_values_method.eq(method).all():
        raise ValueError('Null p_values_method does not match observed scoring')


def _direction(loci_df):
    return np.where(loci_df['type'].to_numpy() == 'OG', 'gain', 'loss')


# ------------------------------------------------------------------------------------- permutation

def arm_bounds():
    """(chrom -> (p_lo, p_hi, q_lo, q_hi)) from the cohort's OBSERVED tables at the `large` scale.

    Deliberately the observed tables and not the assembly extents: they define the searchable span
    that candidate-locus seeding walks, so an event permuted inside them can never land in sequence
    detection does not search.
    """
    tel = data_loaders.load_telomeres_observed()
    cen = data_loaders.load_centromeres(extended=False, observed=True)
    out = {}
    for c in tel.index:
        out[c] = (float(tel.loc[c, ('large', 'chrom_start')]),
                  float(cen.loc[c, ('large', 'centro_start')]),
                  float(cen.loc[c, ('large', 'centro_end')]),
                  float(tel.loc[c, ('large', 'chrom_end')]))
    return out


def _rotation_offset(starts, ends, lo, hi, rng):
    """Uniformly choose an integer circular cut outside every event's interior.

    A cut at an event boundary is valid; a cut through an event would require splitting
    its interval. Gaps are half-open ranges of legal integer cuts, so touching events
    leave a single legal cut between them and nested events do not add extra cuts.
    """
    lo, hi = int(np.ceil(lo)), int(np.floor(hi))
    if hi <= lo:
        return 0
    cursor = lo
    gaps = []
    for start, end in sorted(zip(starts, ends)):
        gap_end = min(int(np.floor(start)) + 1, hi)
        if cursor < gap_end:
            gaps.append((cursor, gap_end))
        cursor = max(cursor, int(np.ceil(end)))
    if cursor < hi:
        gaps.append((cursor, hi))
    total = sum(right - left for left, right in gaps)
    if not total:
        return 0
    draw = int(rng.integers(total))
    for left, right in gaps:
        if draw < right - left:
            return (lo - (left + draw)) % (hi - lo)
        draw -= right - left


def prepare_permutation_events(raw_events, seed, mode, loci_params):
    """Rotate raw events before applying the observed preprocessing filters."""
    from spice.main_loci_functions import process_final_events_for_loci_routines
    options = {key: loci_params.get(key, True) for key in
               ['remove_plateaus', 'remove_chrY', 'drop_duplicates', 'use_observed_centromeres']}
    permuted, moved, fixed = permute_events(raw_events, seed=seed, mode=mode)
    return process_final_events_for_loci_routines(final_events_df=permuted, **options), moved, fixed


def validate_null_mode(frame, mode):
    """Reject incompatible null provenance; untagged historical rotate units remain readable."""
    if mode not in PERMUTE_MODES:
        raise ValueError(f'Unknown permutation mode: {mode}')
    if 'permutation_mode' not in frame:
        return
    modes = set(frame.permutation_mode.dropna())
    if frame.permutation_mode.isna().any() or (len(frame) and modes != {mode}):
        raise ValueError(f'Permutation null mode mismatch: expected {mode}, got {modes}')


def validate_permutation_strategy(mode, strategy):
    if mode not in PERMUTE_MODES:
        raise ValueError(f'Unknown permutation mode: {mode}')
    if strategy not in STRATEGIES:
        raise ValueError(f'strategy must be one of {STRATEGIES}, got {strategy!r}')


def permute_events(events_df, seed, mode='rotate', bounds=None):
    """Permute internal events; return (df, n_moved, n_fixed).

    Only rows with pos == "internal" move: `detection.get_cur_widths` filters on exactly that, so
    they are the only events detection consumes, and every other row passes through untouched so the
    frame stays a valid final_events table. Positions stay inside the arm the event already occupies
    -- one circular offset per (sample, chrom, arm) under `mode='rotate'`, preserving
    circular spacing and overlaps. Cuts are sampled uniformly from integer positions
    outside event interiors, so no event is split at the arm boundary.

    Preserved: per-sample event burden, every width, chromosome and arm membership, non-internal
    positions. Destroyed: the cross-sample alignment of events at the same locus, which is precisely
    what recurrence detection keys on.

    Internal events outside these bounds remain fixed. Dense groups can have few legal
    cuts; a group containing an arm-spanning event can only retain its original position.
    Counts report actual moved and fixed internal rows, including sampled identity moves.
    """
    if mode not in PERMUTE_MODES:
        raise ValueError(f'Unknown permutation mode: {mode}')
    if bounds is None:
        bounds = arm_bounds()
    rng = np.random.default_rng(seed)
    ev = events_df.copy()
    internal = ev['pos'].eq('internal').to_numpy()
    start = ev['start'].to_numpy(float).copy()
    width = ev['width'].to_numpy(float)
    end = ev['end'].to_numpy(float)
    chrom = ev['chrom'].to_numpy()

    arm = np.full(len(ev), '', dtype=object)
    lo = np.full(len(ev), np.nan)
    hi = np.full(len(ev), np.nan)
    for c, (p_lo, p_hi, q_lo, q_hi) in bounds.items():
        m = internal & (chrom == c)
        p = m & (end <= p_hi) & (start >= p_lo)
        q = m & (start >= q_lo) & (end <= q_hi)
        arm[p], lo[p], hi[p] = 'p', p_lo, p_hi
        arm[q], lo[q], hi[q] = 'q', q_lo, q_hi

    ok = internal & (arm != '')
    groups = {}
    samples = ev['sample'].to_numpy()
    for i in np.flatnonzero(ok):
        groups.setdefault((samples[i], chrom[i], arm[i]), []).append(i)
    for indices in groups.values():
        lower, upper = lo[indices[0]], hi[indices[0]]
        delta = _rotation_offset(start[indices], end[indices], lower, upper, rng)
        start[indices] = lower + np.mod(start[indices] - lower + delta, upper - lower)

    moved = internal & (np.rint(start) != events_df['start'].to_numpy())
    ev['start'] = np.rint(start).astype(np.int64)
    ev['end'] = np.rint(start + width).astype(np.int64)
    return ev, int(moved.sum()), int(internal.sum() - moved.sum())


# ------------------------------------------------------------------------------------- null tables

def assign_arm(chrom, pos, bounds=None):
    """'p' or 'q' per locus, split at the observed centromere start (`large` scale)."""
    if bounds is None:
        bounds = arm_bounds()
    mid = np.array([bounds[c][1] if c in bounds else np.inf for c in chrom], float)
    return np.where(np.asarray(pos, float) < mid, 'p', 'q')


def null_from_loci(loci_frames, method='combined_fitness'):
    """Pool per-permutation loci tables into the null: one row per null locus.

    Keeps chrom, direction, ARM, pos, the aggregate statistic and the four per-scale values. `arm`
    records the original chromosome arm for auditing the rotate null. Calibration uses
    chromosome/direction strata. Signed fitness is retained for scoring verification.
    """
    from spice.cohort_model import table_model_id, MODEL_COLUMN
    parts = []
    model_ids = set()
    modes = set()
    untagged = False
    bounds = arm_bounds()
    for df in loci_frames:
        if not len(df):
            continue
        if 'permutation_mode' in df:
            if df.permutation_mode.isna().any():
                raise ValueError('Missing permutation mode in tagged null unit')
            modes.update(df.permutation_mode.unique())
        else:
            untagged = True
        model_ids.add(table_model_id(df))
        validate_null_mode(df, 'rotate')
        per_ls = scoring_per_ls(df, method)
        parts.append(pd.DataFrame({
            'chrom': df['chrom'].to_numpy(), 'direction': _direction(df),
            'arm': assign_arm(df['chrom'].to_numpy(), df['pos'].to_numpy(), bounds),
            'pos': df['pos'].to_numpy(float),
            'stat': scoring_statistic(df, method),
            'p_values_method': method,
            **{f'fitness_{ls}_{dr}': df[f'fitness_{ls}_{dr}'].to_numpy(float)
               for ls in LENGTH_SCALE_NAMES for dr in ('gain', 'loss')},
            **{f'stat_{ls}': per_ls[:, j] for j, ls in enumerate(LENGTH_SCALE_NAMES)}}))
    if not parts:
        raise ValueError('no null loci: every permutation produced an empty loci table')
    if len(modes) > 1 or (modes and untagged):
        raise ValueError('Cannot pool mixed or partly untagged permutation modes')
    if len(model_ids) != 1:
        raise ValueError('Cannot pool mixed shared/legacy cohort models')
    result = pd.concat(parts, ignore_index=True)
    model_id = next(iter(model_ids))
    if model_id is not None:
        result[MODEL_COLUMN] = model_id
    if modes:
        result['permutation_mode'] = next(iter(modes))
    return result


def _empirical_p(obs, ref):
    """Upper-tail empirical p with the +1 correction, which is what floors it at 1/(len(ref)+1).

    Counts null draws >= each observation. searchsorted needs an ASCENDING array, so the count is
    taken as len(ref) - insertion_point rather than by searching a negated (descending) copy.
    """
    ref = np.sort(np.asarray(ref, float))
    n_ge = len(ref) - np.searchsorted(ref, np.asarray(obs, float), side='left')
    return (n_ge + 1) / (len(ref) + 1)


def permutation_p(loci_df, null_df, strategy='zpool_chrom', column='stat', method='combined_fitness'):
    """Upper-tail empirical A+B p-value with null-only chromosome/direction moments.

    Standardize each stratum using its null mean and sample standard deviation,
    pool all standardized null draws, and count ties inclusively with add-one
    correction. Missing observed strata score p=1; zero variance uses scale 1.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f'strategy must be one of {STRATEGIES}, got {strategy!r}')
    from spice.cohort_model import check_scoring_models
    if len(loci_df):
        check_scoring_models(loci_df, null_df)
    validate_null_scoring(null_df, method)
    validate_null_mode(null_df, 'rotate')
    if not len(loci_df):
        return np.zeros(0)
    obs_stat = (scoring_statistic(loci_df, method) if column == 'stat'
                else scoring_per_ls(loci_df, method)[:, LENGTH_SCALE_NAMES.index(column.split('_', 1)[1])])
    direction = _direction(loci_df)
    chrom = loci_df['chrom'].to_numpy()

    obs_keys = list(zip(chrom, direction))
    null_keys = list(zip(null_df['chrom'], null_df['direction']))

    zo = np.full(len(loci_df), np.nan)
    zn = []
    by_key = {}
    for k, v in zip(null_keys, null_df[column].to_numpy(float)):
        by_key.setdefault(k, []).append(v)
    for k, vals in by_key.items():
        v = np.asarray(vals, float)
        mu, sd = float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0
        zn.append((v - mu) / (sd or 1.0))
    ref = np.sort(np.concatenate(zn)) if zn else np.zeros(1)

    for i, k in enumerate(obs_keys):
        v = np.asarray(by_key.get(k, []), float)
        if not len(v):
            continue                      # no null in this stratum -> left at p = 1 below
        mu = float(v.mean())
        sd = float(v.std(ddof=1)) if len(v) > 1 else 0.0
        zo[i] = (obs_stat[i] - mu) / (sd or 1.0)
    missing = np.isnan(zo)
    if missing.any():
        logger.warning(f'{int(missing.sum())} loci have no matching null stratum; '
                       'scored as non-significant')
        zo[missing] = -np.inf
    return _empirical_p(zo, ref)
