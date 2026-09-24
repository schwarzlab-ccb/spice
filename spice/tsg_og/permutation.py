"""Positional-permutation null for the fitness p-value.

Permute event positions in the real cohort, run the same
detection on the result, and pool the loci it finds. Null loci are then produced by the same cascade
as the observed ones. Legacy modes filter after randomization; chromosome modes
select the observed preprocessed event set first, then randomize its positions.

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
STRATEGIES = ('zpool', 'zpool_chrom', 'pooled', 'perchrom')
CHROMOSOME_MODES = ('chromosome_hybrid', 'chromosome_exclusion')
PERMUTE_MODES = ('rotate', 'uniform') + CHROMOSOME_MODES
#: If either arm has fewer draws than this, both arms use their chromosome/direction
#: stratum. This avoids estimating calibration moments from a sparse or absent arm.
MIN_STRATUM_DRAWS = 20


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
    """The tested statistic: mean over the four same-direction length scales."""
    return fitness_per_ls(loci_df).mean(axis=1)


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


def _hybrid_start_ranges(width, bounds):
    """Disjoint inclusive integer start ranges; gap boundaries themselves are allowed.

    Short events fit wholly in either usable arm. Events longer than the shorter
    arm may also bridge the gap, with both endpoints in usable arms. A missing
    arm never enables bridging. Widths stay in physical bp, including any gap.
    """
    if not np.isfinite([width, *bounds]).all() or width <= 0 or width != int(width):
        raise ValueError('Hybrid placement requires finite bounds and positive integer event widths')
    p_lo, p_hi, q_lo, q_hi = bounds
    arms = [(lo, hi) for lo, hi in [(p_lo, p_hi), (q_lo, q_hi)] if hi > lo]
    if len(arms) == 2 and p_hi > q_lo:
        raise ValueError('Hybrid chromosome arms must not overlap')
    ranges = [(int(np.ceil(lo)), int(np.floor(hi-width))) for lo, hi in arms]
    if len(arms) == 2 and width > min(p_hi-p_lo, q_hi-q_lo):
        # Start in p, end in q: the intersection of their endpoint constraints.
        ranges.append((int(np.ceil(max(p_lo, q_lo-width))),
                       int(np.floor(min(p_hi, q_hi-width)))))
    merged = []
    for lo, hi in sorted((lo, hi) for lo, hi in ranges if lo <= hi):
        if merged and lo <= merged[-1][1]+1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return merged


def _exclusion_start_ranges(width, bounds):
    """Legal integer starts after excluding either endpoint inside the centromere.

    For gap (Cstart, Cend), forbidden starts are its interior and the shifted
    interior (Cstart-width, Cend-width). Boundary contact and spanning are legal
    at every width. Sample only where the complete event fits in the observed
    chromosome; a missing arm restricts placement to the remaining usable arm.
    """
    if not np.isfinite([width, *bounds]).all() or width <= 0 or width != int(width):
        raise ValueError('Exclusion placement requires finite bounds and positive integer event widths')
    p_lo, p_hi, q_lo, q_hi = bounds
    arms = [(lo, hi) for lo, hi in [(p_lo, p_hi), (q_lo, q_hi)] if hi > lo]
    if len(arms) == 2 and p_hi > q_lo:
        raise ValueError('Exclusion chromosome arms must not overlap')
    if not arms:
        return []
    lo, hi = int(np.ceil(arms[0][0])), int(np.floor(arms[-1][1]-width))
    ranges = [(lo, hi)] if lo <= hi else []
    if len(arms) == 2:
        for left, right in [(p_hi, q_lo), (p_hi-width, q_lo-width)]:
            # Convert open physical intervals into inclusive integer exclusions.
            first, last = int(np.floor(left))+1, int(np.ceil(right))-1
            if first > last:
                continue
            remaining = []
            for lo, hi in ranges:
                if last < lo or first > hi:
                    remaining.append((lo, hi))
                else:
                    if lo < first:
                        remaining.append((lo, first-1))
                    if last < hi:
                        remaining.append((last+1, hi))
            ranges = remaining
    return ranges


def _draw_start(ranges, rng):
    """Uniform over legal integer coordinates, not uniform over arms or ranges."""
    total = sum(hi-lo+1 for lo, hi in ranges)
    if not total:
        return None
    draw = int(rng.integers(total))
    for lo, hi in ranges:
        count = hi-lo+1
        if draw < count:
            return lo+draw
        draw -= count


def _permute_chromosome(events_df, seed, bounds, mode):
    start_ranges = (_hybrid_start_ranges if mode == 'chromosome_hybrid'
                    else _exclusion_start_ranges)
    ev = events_df.copy()
    internal = ev['pos'].eq('internal').to_numpy()
    rng = np.random.default_rng(seed)
    starts = ev['start'].to_numpy(copy=True)
    ends = ev['end'].to_numpy(copy=True)
    for i in np.flatnonzero(internal):
        row = ev.iloc[i]
        if row.chrom not in bounds:
            raise ValueError(f'Missing {mode} arm bounds for {row.chrom}')
        # Imported event tables can carry a model width different from their
        # coordinate span. Preserve both: geometry controls placement, while
        # the original width continues to determine SPICE's scale/kernel inputs.
        span = row.end-row.start
        start = _draw_start(start_ranges(span, bounds[row.chrom]), rng)
        # No valid placement: retain the event and include it in the fixed count.
        if start is not None:
            starts[i], ends[i] = start, start+span
    ev['start'], ev['end'] = starts, ends
    moved = int((internal & (starts != events_df['start'].to_numpy())).sum())
    return ev, moved, int(internal.sum())-moved


def prepare_permutation_events(raw_events, seed, mode, loci_params):
    """Return the exact event frame to pass directly to null detection/combination.

    Each chromosome null conditions on the observed preprocessed event set. Filtering
    randomized coordinates again would selectively remove the new bridge events,
    alter sample burdens and potentially drop whole IDs. Legacy order is unchanged.
    """
    from spice.main_loci_functions import process_final_events_for_loci_routines
    options = {key: loci_params.get(key, True) for key in
               ['remove_plateaus', 'remove_chrY', 'drop_duplicates', 'use_observed_centromeres']}
    if mode in CHROMOSOME_MODES:
        prepared = process_final_events_for_loci_routines(final_events_df=raw_events, **options)
        return permute_events(prepared, seed=seed, mode=mode)
    permuted, moved, fixed = permute_events(raw_events, seed=seed, mode=mode)
    return process_final_events_for_loci_routines(final_events_df=permuted, **options), moved, fixed


def validate_null_mode(frame, mode):
    """Chromosome tables must carry their mode through scatter, pooling and scoring."""
    if 'permutation_mode' not in frame:
        if mode in CHROMOSOME_MODES:
            raise ValueError(f'{mode} null requires matching provenance; generate a fresh null')
        return
    modes = set(frame.permutation_mode.dropna())
    if frame.permutation_mode.isna().any() or (len(frame) and modes != {mode}):
        raise ValueError(f'Permutation null mode mismatch: expected {mode}, got {modes}')


def validate_permutation_strategy(mode, strategy):
    if mode not in PERMUTE_MODES:
        raise ValueError(f'Unknown permutation mode: {mode}')
    if mode in CHROMOSOME_MODES and strategy not in ('zpool_chrom', 'perchrom'):
        raise ValueError(f'{mode} requires chromosome calibration: zpool_chrom or perchrom')


def permute_events(events_df, seed, mode='rotate', bounds=None):
    """Permute internal events; return (df, n_moved, n_fixed).

    `chromosome_hybrid` places events independently across the chromosome using
    _hybrid_start_ranges with the coordinate span (end-start). It preserves both
    that span and the stored model width, plus sample/direction/chromosome
    identities, but not spacing or overlaps. Use prepare_permutation_events for
    raw inputs so the observed event set is selected before this randomization.

    `chromosome_exclusion` uses the same independent sampling but excludes only
    starts placing either endpoint inside the centromere. Any event can span the
    gap, irrespective of the shorter arm's length; physical boundaries may touch.

    The legacy rotate/uniform modes operate within each original arm:

    Only rows with pos == "internal" move: `detection.get_cur_widths` filters on exactly that, so
    they are the only events detection consumes, and every other row passes through untouched so the
    frame stays a valid final_events table. Positions stay inside the arm the event already occupies
    -- one circular offset per (sample, chrom, arm) under `mode='rotate'`, preserving
    circular spacing and overlaps. Cuts are sampled uniformly from integer positions
    outside event interiors, so no event is split at the arm boundary. `mode='uniform'`
    places each event independently.

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
    if mode in CHROMOSOME_MODES:
        return _permute_chromosome(events_df, seed, bounds, mode)
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
    span = hi - lo
    room = np.maximum(span - width, 0.0)
    if mode == 'uniform':
        start[ok] = lo[ok] + rng.uniform(0, 1, int(ok.sum())) * room[ok]
    else:
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


def null_from_loci(loci_frames):
    """Pool per-permutation loci tables into the null: one row per null locus.

    Keeps chrom, direction, ARM, pos, the aggregate statistic and the four per-scale values. `arm`
    is the stratum the default scoring uses, and it belongs here rather than being recomputed later
    because it is defined by the cohort's own observed centromere table -- the same one the
    permutation rotated within. `pos` is kept so the arm call can be audited or redone.
    """
    parts = []
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
        per_ls = fitness_per_ls(df)
        parts.append(pd.DataFrame({
            'chrom': df['chrom'].to_numpy(), 'direction': _direction(df),
            'arm': assign_arm(df['chrom'].to_numpy(), df['pos'].to_numpy(), bounds),
            'pos': df['pos'].to_numpy(float),
            'stat': per_ls.mean(axis=1),
            **{f'stat_{ls}': per_ls[:, j] for j, ls in enumerate(LENGTH_SCALE_NAMES)}}))
    if not parts:
        raise ValueError('no null loci: every permutation produced an empty loci table')
    if len(modes) > 1 or (modes and untagged):
        raise ValueError('Cannot pool mixed or partly untagged permutation modes')
    result = pd.concat(parts, ignore_index=True)
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


def _z(values, mu, sd):
    return (np.asarray(values, float) - mu) / (sd if sd else 1.0)


def _strata(chrom, direction, arm, null_df, level):
    """Stratum key per locus, falling back from arm to chromosome where the arm is too thin."""
    if level == 'chrom':
        return list(zip(chrom, direction))
    counts = null_df.groupby(['chrom', 'direction', 'arm']).size()
    # Collapse BOTH arms to keep a disjoint partition of the reference: merely renaming
    # the thin arm's key leaves its sample size unchanged. Missing arms count as zero.
    fallback = {(c, d) for c, d in zip(null_df['chrom'], null_df['direction'])
                if any(counts.get((c, d, a), 0) < MIN_STRATUM_DRAWS for a in ('p', 'q'))}
    return [(c, d) if (c, d) in fallback else (c, d, a)
            for c, d, a in zip(chrom, direction, arm)]


def permutation_p(loci_df, null_df, strategy='zpool', column='stat'):
    """Empirical p of each observed locus against the pooled permutation null.

    `zpool` standardizes within (chromosome, direction, arm), then pools the standardized
    null draws. If either arm has fewer than MIN_STRATUM_DRAWS draws (including zero),
    BOTH arms use the chromosome/direction stratum instead. Each null draw enters the
    pooled reference exactly once. A chromosome/direction with no null draws scores p=1.

    The observed locus is included in its stratum's mean and sample standard deviation
    when standardizing that observation; the pooled null uses null-only moments.

    `zpool_chrom` always uses chromosome/direction strata and null-only moments.
    `pooled` compares raw fitness to a direction-matched genome-wide reference.
    `perchrom` compares raw fitness within chromosome/direction; its smaller reference
    gives a higher minimum attainable p-value.

    mu/sd never see any observed locus other than the one being scored, so no other locus's signal
    leaks into its reference.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f'strategy must be one of {STRATEGIES}, got {strategy!r}')
    if 'permutation_mode' in null_df:
        for mode in CHROMOSOME_MODES:
            if (null_df.permutation_mode == mode).any():
                validate_null_mode(null_df, mode)
                validate_permutation_strategy(mode, strategy)
    if not len(loci_df):
        return np.zeros(0)
    obs_stat = (fitness_statistic(loci_df) if column == 'stat'
                else fitness_per_ls(loci_df)[:, LENGTH_SCALE_NAMES.index(column.split('_', 1)[1])])
    direction = _direction(loci_df)
    chrom = loci_df['chrom'].to_numpy()

    if strategy == 'pooled':
        p = np.ones(len(loci_df))
        for dr in ('gain', 'loss'):
            m = direction == dr
            if m.any():
                p[m] = _empirical_p(obs_stat[m], null_df.loc[null_df.direction == dr, column])
        return p
    if strategy == 'perchrom':
        p = np.ones(len(loci_df))
        for (c, dr), g in null_df.groupby(['chrom', 'direction']):
            m = (chrom == c) & (direction == dr)
            if m.any():
                p[m] = _empirical_p(obs_stat[m], g[column])
        return p

    # ---- zpool / zpool_chrom ----
    level = 'chrom' if strategy == 'zpool_chrom' else 'arm'
    add_one_in = (strategy == 'zpool')
    if level == 'arm' and 'arm' not in null_df.columns:
        raise ValueError("the null has no 'arm' column -- it predates arm stratification; re-pool "
                         "it with `spice permute --pool`, or score with strategy='zpool_chrom'")
    arm = (assign_arm(chrom, loci_df['pos'].to_numpy()) if level == 'arm'
           else np.array([''] * len(loci_df)))
    obs_keys = _strata(chrom, direction, arm, null_df, level)
    null_keys = _strata(null_df['chrom'].to_numpy(), null_df['direction'].to_numpy(),
                        null_df['arm'].to_numpy() if level == 'arm' else
                        np.array([''] * len(null_df)), null_df, level)

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
        if add_one_in:
            v = np.append(v, obs_stat[i])
        mu = float(v.mean())
        sd = float(v.std(ddof=1)) if len(v) > 1 else 0.0
        zo[i] = (obs_stat[i] - mu) / (sd or 1.0)
    missing = np.isnan(zo)
    if missing.any():
        logger.warning(f'{int(missing.sum())} loci have no matching null stratum; '
                       'scored as non-significant')
        zo[missing] = -np.inf
    return _empirical_p(zo, ref)
