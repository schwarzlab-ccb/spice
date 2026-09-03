"""Positional-permutation null for the fitness p-value.

Replaces the resimulation null. The resim null drew each null locus from a single optimizer pass,
while an observed locus is the survivor of the full multi-stage detection cascade, and the tested
statistic is the MEAN OVER FOUR same-direction length scales with a fixed four-slot denominator. A
null whose loci populate fewer scales than the observed ones is therefore beaten by construction --
measured on a driver-free cohort, the resim null's loci carried 2.00 non-zero scales against the
observed 2.72, and no setting of the old knobs fixed it (one was anti-conservative, the other
conservative, and which was which flipped with cohort composition).

This null closes that gap by construction: permute event POSITIONS in the real cohort, run the SAME
detection on the result, and pool the loci it finds. Null loci are then produced by the same cascade
as the observed ones, including every event-preprocessing and filtering step, so the asymmetry
cannot arise. It also leaves a driver-free cohort free to VALIDATE the null, which scoring against
that cohort's own loci could not (that is circular).

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
#: per-unit filename written by `spice permute --seed S --chrom C`
UNIT_TEMPLATE = 'permutation_unit_s{seed}_{chrom}.tsv'
DEFAULT_K = 16
STRATEGIES = ('zpool', 'pooled', 'perchrom')


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


def permute_events(events_df, seed, mode='rotate', bounds=None):
    """Permute internal-event POSITIONS within each chromosome arm. Returns (df, n_moved, n_fixed).

    Only rows with pos == "internal" move: `detection.get_cur_widths` filters on exactly that, so
    they are the only events detection consumes, and every other row passes through untouched so the
    frame stays a valid final_events table. Positions stay inside the arm the event already occupies
    -- one offset per (sample, chrom, arm) under `mode='rotate'`, so the arm is rigidly shifted and
    relative spacing survives; `mode='uniform'` places each event independently.

    Preserved: per-sample event burden, every width, chromosome and arm membership, non-internal
    positions. Destroyed: the cross-sample alignment of events at the same locus, which is precisely
    what recurrence detection keys on.

    An event straddling the centromere belongs to neither arm and is LEFT IN PLACE (~4-5% of
    internal events), so a little real recurrence survives: the null is mildly conservative there.
    """
    if mode not in ('rotate', 'uniform'):
        raise ValueError(f"mode must be 'rotate' or 'uniform', got {mode!r}")
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
    span = hi - lo
    room = np.maximum(span - width, 1.0)          # keeps the event inside its arm
    if mode == 'uniform':
        start[ok] = lo[ok] + rng.uniform(0, 1, int(ok.sum())) * room[ok]
    else:
        key = pd.Series(list(zip(ev['sample'].to_numpy()[ok], chrom[ok], arm[ok])))
        codes, uniq = pd.factorize(key)
        delta = rng.uniform(0, 1, len(uniq))[codes] * span[ok]
        start[ok] = lo[ok] + np.mod(start[ok] - lo[ok] + delta, room[ok])

    ev['start'] = np.rint(start).astype(np.int64)
    ev['end'] = np.rint(start + width).astype(np.int64)
    return ev, int(ok.sum()), int(internal.sum() - ok.sum())


# ------------------------------------------------------------------------------------- null tables

def null_from_loci(loci_frames):
    """Pool per-permutation loci tables into the null: one row per null locus.

    Keeps chrom + direction + the aggregate statistic + the four per-scale values, which is
    everything the scoring needs; the loci's coordinates are irrelevant once pooled.
    """
    parts = []
    for df in loci_frames:
        if not len(df):
            continue
        per_ls = fitness_per_ls(df)
        parts.append(pd.DataFrame({
            'chrom': df['chrom'].to_numpy(), 'direction': _direction(df),
            'stat': per_ls.mean(axis=1),
            **{f'stat_{ls}': per_ls[:, j] for j, ls in enumerate(LENGTH_SCALE_NAMES)}}))
    if not parts:
        raise ValueError('no null loci: every permutation produced an empty loci table')
    return pd.concat(parts, ignore_index=True)


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


def permutation_p(loci_df, null_df, strategy='zpool', column='stat'):
    """Empirical p of each observed locus against the pooled permutation null.

    `zpool` (the default) standardizes within each (chrom, direction) stratum using the NULL's own
    mean and sd, then pools the standardized values. That keeps the reference set large while
    restoring the per-chromosome context plain pooling discards -- a permutation yields only ~6-8
    loci per chromosome, so a per-stratum empirical p (`perchrom`) floors at ~1/100 and BH can then
    never reach significance. `pooled` scores the raw statistic against a direction-matched
    genome-wide reference: perfectly precise in practice but it costs recall, because a quiet
    chromosome's loci are compared against a reference dominated by busy ones.

    mu/sd always come from the null, never from the observed loci, so no observed signal leaks into
    its own reference.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f'strategy must be one of {STRATEGIES}, got {strategy!r}')
    if not len(loci_df):
        return np.zeros(0)
    obs_stat = (fitness_statistic(loci_df) if column == 'stat'
                else fitness_per_ls(loci_df)[:, LENGTH_SCALE_NAMES.index(column.split('_', 1)[1])])
    direction = _direction(loci_df)
    chrom = loci_df['chrom'].to_numpy()
    p = np.ones(len(loci_df))

    if strategy == 'pooled':
        for dr in ('gain', 'loss'):
            m = direction == dr
            if m.any():
                p[m] = _empirical_p(obs_stat[m], null_df.loc[null_df.direction == dr, column])
    elif strategy == 'perchrom':
        for (c, dr), g in null_df.groupby(['chrom', 'direction']):
            m = (chrom == c) & (direction == dr)
            if m.any():
                p[m] = _empirical_p(obs_stat[m], g[column])
    else:                                                                   # zpool
        zo = np.full(len(loci_df), np.nan)
        zn = []
        for (c, dr), g in null_df.groupby(['chrom', 'direction']):
            vals = g[column].to_numpy(float)
            # ddof=1: the null draws are a SAMPLE of the stratum's null distribution, not the
            # population. Stated explicitly because numpy defaults to ddof=0 and pandas to
            # ddof=1, and the two disagree by ~3e-3 in the resulting p at ~100 draws/stratum.
            mu, sd = float(vals.mean()), float(vals.std(ddof=1))
            zn.append(_z(vals, mu, sd))
            m = (chrom == c) & (direction == dr)
            if m.any():
                zo[m] = _z(obs_stat[m], mu, sd)
        ref = np.concatenate(zn) if zn else np.zeros(1)
        missing = np.isnan(zo)
        if missing.any():
            # An observed (chrom, direction) with no null loci at all: cannot be standardized, so
            # it is scored at the bottom of the reference (p = 1) rather than silently as extreme.
            logger.warning(f'{int(missing.sum())} loci have no matching null stratum '
                           f'(chrom x direction); scored as non-significant')
            zo[missing] = -np.inf
        p = _empirical_p(zo, ref)
    return p
