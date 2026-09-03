"""Tests for the positional-permutation null (spice.tsg_og.permutation)."""

import numpy as np
import pandas as pd
import pytest

from spice.length_scales import LENGTH_SCALE_NAMES
from spice.tsg_og.permutation import (
    STRATEGIES, fitness_per_ls, fitness_statistic, null_from_loci, permutation_p, permute_events)


def _loci(n=6, seed=0):
    rng = np.random.default_rng(seed)
    # works for odd n too: a bare ['OG','TSG'] * (n//2) is empty at n=1
    d = {'chrom': ['chr1'] * (n // 2) + ['chr2'] * (n - n // 2),
         'type': [('OG', 'TSG')[i % 2] for i in range(n)]}
    for ls in LENGTH_SCALE_NAMES:
        d[f'fitness_{ls}_gain'] = rng.uniform(-0.5, 2, n)
        d[f'fitness_{ls}_loss'] = rng.uniform(-0.5, 2, n)
    return pd.DataFrame(d)


def _events(n=400, seed=1):
    rng = np.random.default_rng(seed)
    width = rng.integers(10_000, 2_000_000, n)
    start = rng.integers(3_000_000, 100_000_000, n)
    return pd.DataFrame({
        'sample': [f'S{i % 20}' for i in range(n)], 'chrom': 'chr1',
        'start': start, 'end': start + width, 'width': width,
        'type': rng.choice(['gain', 'loss'], n),
        'pos': rng.choice(['internal', 'telomere_bound'], n, p=[0.8, 0.2])})


BOUNDS = {'chr1': (1e4, 120e6, 148e6, 249e6)}


class TestStatistic:
    def test_direction_matched_and_clipped(self):
        loci = _loci()
        per_ls = fitness_per_ls(loci)
        assert per_ls.shape == (len(loci), 4)
        assert (per_ls >= 0).all(), 'negative fitness must be clipped: the sign clamp holds the ' \
                                    'opposite direction <= 0 and it must not drag the mean down'
        # an OG locus reads the gain columns, a TSG the loss columns
        og = loci.index[loci.type == 'OG'][0]
        assert per_ls[og, 0] == max(0.0, loci.loc[og, 'fitness_small_gain'])
        tsg = loci.index[loci.type == 'TSG'][0]
        assert per_ls[tsg, 0] == max(0.0, loci.loc[tsg, 'fitness_small_loss'])

    def test_statistic_is_the_mean_over_four_slots(self):
        loci = _loci()
        assert np.allclose(fitness_statistic(loci), fitness_per_ls(loci).mean(axis=1))

    def test_empty_input(self):
        assert fitness_per_ls(_loci().iloc[:0]).shape == (0, 4)


class TestPermuteEvents:
    def test_preserves_burden_widths_and_non_internal_rows(self):
        ev = _events()
        out, moved, fixed = permute_events(ev, seed=7, bounds=BOUNDS)
        assert len(out) == len(ev)
        assert (out['width'].to_numpy() == ev['width'].to_numpy()).all()
        assert out['sample'].value_counts().equals(ev['sample'].value_counts())
        # non-internal rows are untouched -- they never reach detection anyway
        keep = ~ev['pos'].eq('internal').to_numpy()
        assert (out.loc[keep, 'start'].to_numpy() == ev.loc[keep, 'start'].to_numpy()).all()
        assert moved > 0 and moved + fixed == int(ev['pos'].eq('internal').sum())

    def test_internal_events_actually_move_and_stay_in_bounds(self):
        ev = _events()
        out, _, _ = permute_events(ev, seed=7, bounds=BOUNDS)
        internal = ev['pos'].eq('internal').to_numpy()
        assert not np.array_equal(out.loc[internal, 'start'], ev.loc[internal, 'start'])
        lo, p_hi, q_lo, hi = BOUNDS['chr1']
        s, e = out.loc[internal, 'start'].to_numpy(), out.loc[internal, 'end'].to_numpy()
        assert (s >= lo).all() and (e <= hi).all()
        # nothing is rotated across the centromere into unsearchable sequence
        assert not (((s < q_lo) & (e > p_hi)).any())

    def test_deterministic_and_seed_dependent(self):
        ev = _events()
        a, _, _ = permute_events(ev, seed=3, bounds=BOUNDS)
        b, _, _ = permute_events(ev, seed=3, bounds=BOUNDS)
        c, _, _ = permute_events(ev, seed=4, bounds=BOUNDS)
        assert a['start'].equals(b['start'])
        assert not a['start'].equals(c['start'])

    def test_rejects_unknown_mode(self):
        with pytest.raises(ValueError, match='mode'):
            permute_events(_events(), seed=1, mode='nope', bounds=BOUNDS)


class TestPValue:
    def test_null_from_loci_shape(self):
        null = null_from_loci([_loci(6, 0), _loci(6, 1)])
        assert len(null) == 12
        assert {'chrom', 'direction', 'stat'} <= set(null.columns)
        assert all(f'stat_{ls}' in null.columns for ls in LENGTH_SCALE_NAMES)

    def test_null_from_loci_rejects_all_empty(self):
        with pytest.raises(ValueError, match='no null loci'):
            null_from_loci([_loci().iloc[:0]])

    @pytest.mark.parametrize('strategy', STRATEGIES)
    def test_p_in_unit_interval_and_floored(self, strategy):
        null = null_from_loci([_loci(20, s) for s in range(5)])
        obs = _loci(6, 99)
        p = permutation_p(obs, null, strategy)
        assert len(p) == len(obs)
        assert ((p > 0) & (p <= 1)).all()
        assert p.min() >= 1 / (len(null) + 1) - 1e-12, 'the +1 correction sets the floor'

    def test_a_locus_above_every_null_draw_hits_the_floor(self):
        null = null_from_loci([_loci(30, 0)])
        obs = _loci(1, 0).assign(**{f'fitness_{ls}_gain': 1e6 for ls in LENGTH_SCALE_NAMES})
        obs['type'] = 'OG'
        p = permutation_p(obs, null, 'pooled')
        assert p[0] == pytest.approx(1 / (len(null[null.direction == 'gain']) + 1))

    def test_rejects_unknown_strategy(self):
        with pytest.raises(ValueError, match='strategy'):
            permutation_p(_loci(), null_from_loci([_loci()]), 'nope')
