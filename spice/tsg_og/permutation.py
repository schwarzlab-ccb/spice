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
STRATEGIES = ('zpool', 'zpool_chrom', 'pooled', 'perchrom')
#: a (chrom, direction, arm) stratum with fewer null draws than this falls back to its chromosome:
#: mu/sd from a handful of draws is noise, and the acrocentric p arms (chr13/14/15/21/22) plus a few
#: gene-poor short arms genuinely hold almost no loci. Measured at K=16: 4-5 of ~80 arm strata fall
#: below it, holding ~1% of null loci (smallest 6-7 draws, median stratum 54-87).
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
    bounds = arm_bounds()
    for df in loci_frames:
        if not len(df):
            continue
        per_ls = fitness_per_ls(df)
        parts.append(pd.DataFrame({
            'chrom': df['chrom'].to_numpy(), 'direction': _direction(df),
            'arm': assign_arm(df['chrom'].to_numpy(), df['pos'].to_numpy(), bounds),
            'pos': df['pos'].to_numpy(float),
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


def _strata(chrom, direction, arm, null_df, level):
    """Stratum key per locus, falling back from arm to chromosome where the arm is too thin."""
    if level == 'chrom':
        return list(zip(chrom, direction))
    counts = null_df.groupby(['chrom', 'direction', 'arm']).size()
    thin = {k for k, v in counts.items() if v < MIN_STRATUM_DRAWS}
    return [(c, d) if (c, d, a) in thin else (c, d, a)
            for c, d, a in zip(chrom, direction, arm)]


def permutation_p(loci_df, null_df, strategy='zpool', column='stat'):
    """Empirical p of each observed locus against the pooled permutation null.

    `zpool` (the default) standardizes within each stratum using that stratum's null draws, then
    pools the standardized values. Pooling is what keeps the reference set large: a permutation
    yields only ~6-8 loci per chromosome, so a per-stratum empirical p (`perchrom`) floors near
    1/100 and BH can then never reach significance. Two details make the default the calibrated
    choice, and both were measured rather than assumed:

    * **The stratum is the ARM**, not the chromosome. `permute_events` rotates within the arm, so the
      arm is the null's actual exchangeability unit and chromosome strata pool two arms the
      permutation never mixed. Measured on a driver-free cohort this improves KS D 0.075 -> 0.062
      AND finds more true drivers on the selection cohort (84 vs 81) at higher precision (86.6% vs
      82.7%) -- a strict improvement. Arms holding fewer than MIN_STRATUM_DRAWS null loci fall back
      to their chromosome.
    * **The observed locus is included in its own stratum's mu/sd** (`add_one_in`). This matches the
      +1 already in the empirical p: a value must be part of the calibration it is judged against,
      or it is not exchangeable with the null. Without it mu/sd come from the null alone and an
      observed locus can sit 7 sd outside its stratum and beat the entire pooled reference -- which
      is exactly what put 2 loci at the p-floor on a driver-free cohort where 0.06 were expected.
      With it, that cohort yields ZERO rejections. It costs ~7 true positives (77 vs 84), i.e. it
      buys a balanced null with a little power.

    `zpool_chrom` is the previous behaviour (chromosome strata, mu/sd from the null alone), kept so
    earlier runs can be reproduced. `pooled` scores the raw statistic against a direction-matched
    genome-wide reference: more precise, roughly half the recall, because a quiet chromosome's loci
    are judged against a reference dominated by busy ones. `perchrom` is a diagnostic only.

    mu/sd never see any observed locus other than the one being scored, so no other locus's signal
    leaks into its reference.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f'strategy must be one of {STRATEGIES}, got {strategy!r}')
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
