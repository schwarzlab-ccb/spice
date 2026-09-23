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
         'type': [('OG', 'TSG')[i % 2] for i in range(n)],
         # `pos` is required now: the default stratification is per chromosome ARM, so both the
         # null and the observed loci need a coordinate to be assigned one. Spread across both
         # arms of each chromosome so the strata are populated.
         'pos': [(20e6 if i % 2 else 200e6) for i in range(n)]}
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
        assert {'chrom', 'direction', 'arm', 'pos', 'stat'} <= set(null.columns)
        assert all(f'stat_{ls}' in null.columns for ls in LENGTH_SCALE_NAMES)
        assert set(null['arm']) <= {'p', 'q'}

    def test_zpool_needs_an_arm_column(self):
        """A null pooled before arm stratification must fail loudly, not silently mis-stratify."""
        null = null_from_loci([_loci(20, 0)]).drop(columns=['arm'])
        with pytest.raises(ValueError, match='arm'):
            permutation_p(_loci(6, 1), null, 'zpool')

    def test_thin_arm_strata_fall_back_to_the_chromosome(self):
        """An arm with < MIN_STRATUM_DRAWS null loci is scored against its whole chromosome."""
        from spice.tsg_og.permutation import MIN_STRATUM_DRAWS, _strata
        null = null_from_loci([_loci(60, s) for s in range(6)])
        counts = null.groupby(['chrom', 'direction', 'arm']).size()
        thin = [k for k, v in counts.items() if v < MIN_STRATUM_DRAWS]
        keys = _strata(null['chrom'].to_numpy(), null['direction'].to_numpy(),
                       null['arm'].to_numpy(), null, 'arm')
        for c, d, a in thin:                      # a thin arm collapses to a 2-tuple
            assert (c, d) in keys

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
        obs = _loci(2, 0).assign(**{f'fitness_{ls}_gain': 1e6 for ls in LENGTH_SCALE_NAMES})
        obs['type'] = 'OG'
        p = permutation_p(obs, null, 'pooled')
        assert p[0] == pytest.approx(1 / (len(null[null.direction == 'gain']) + 1))

    def test_rejects_unknown_strategy(self):
        with pytest.raises(ValueError, match='strategy'):
            permutation_p(_loci(), null_from_loci([_loci()]), 'nope')


@pytest.mark.parametrize('n_p', [0, 5, 19, 20])
def test_arm_fallback_uses_a_disjoint_whole_chromosome(n_p):
    from spice.tsg_og.permutation import _strata
    null = pd.DataFrame({'chrom': ['chr1'] * (n_p + 100),
                         'direction': ['gain'] * (n_p + 100),
                         'arm': ['p'] * n_p + ['q'] * 100})
    keys = _strata(null.chrom, null.direction, null.arm, null, 'arm')
    if n_p < 20:
        assert keys == [('chr1', 'gain')] * len(null)
        assert _strata(['chr1'], ['gain'], ['p'], null, 'arm') == [('chr1', 'gain')]
    else:
        assert keys.count(('chr1', 'gain', 'p')) == 20
        assert keys.count(('chr1', 'gain', 'q')) == 100


@pytest.mark.parametrize('n_p', [0, 5])
def test_sparse_arm_p_value_matches_whole_chromosome_calibration(n_p, monkeypatch):
    from spice.tsg_og import permutation
    monkeypatch.setattr(permutation, 'arm_bounds', lambda: BOUNDS)
    values = np.r_[np.arange(1, n_p + 1), np.linspace(10, 20, 100)]
    null = pd.DataFrame({'chrom': 'chr1', 'direction': 'gain',
                         'arm': ['p'] * n_p + ['q'] * 100, 'stat': values})
    obs = _loci(1).assign(chrom='chr1', type='OG', pos=20e6)
    for ls in LENGTH_SCALE_NAMES:
        obs[f'fitness_{ls}_gain'] = 12.0
    augmented = np.r_[values, 12.0]
    z_obs = (12 - augmented.mean()) / augmented.std(ddof=1)
    z_null = (values - values.mean()) / values.std(ddof=1)
    expected = (np.count_nonzero(z_null >= z_obs) + 1) / (len(values) + 1)
    assert permutation_p(obs, null)[0] == pytest.approx(expected)
    assert expected < 1



def test_rotation_preserves_circular_spacing_and_overlaps_with_unequal_widths():
    ev = pd.DataFrame({'sample': ['S'] * 4, 'chrom': 'chr1', 'pos': 'internal',
                       'start': [100, 100, 300, 650], 'end': [200, 400, 550, 750]})
    ev['width'] = ev.end - ev.start
    bounds = {'chr1': (0, 1000, 1100, 2000)}

    def overlaps(frame):
        starts, ends = frame.start.to_numpy(), frame.end.to_numpy()
        return np.maximum(0, np.minimum(ends[:, None], ends) - np.maximum(starts[:, None], starts))

    observed_moves = set()
    for seed in range(32):
        out, moved, fixed = permute_events(ev, seed, bounds=bounds)
        shifts = (out.start.to_numpy() - ev.start.to_numpy()) % 1000
        assert (shifts == shifts[0]).all()
        assert out.start.iloc[0] == out.start.iloc[1]
        np.testing.assert_array_equal(overlaps(out), overlaps(ev))
        np.testing.assert_array_equal(out.end - out.start, ev.width)
        assert (out.start >= 0).all() and (out.end <= 1000).all()
        assert moved + fixed == len(ev)
        observed_moves.add(int(shifts[0]))
    assert len(observed_moves) > 1


@pytest.mark.parametrize('mode', ['rotate', 'uniform'])
def test_arm_spanning_event_has_no_room_to_move(mode):
    ev = pd.DataFrame({'sample': ['S'], 'chrom': 'chr1', 'pos': 'internal',
                       'start': [0], 'end': [1000], 'width': [1000]})
    for seed in range(10):
        out, moved, fixed = permute_events(ev, seed, mode=mode,
                                          bounds={'chr1': (0, 1000, 1100, 2000)})
        pd.testing.assert_frame_equal(out, ev)
        assert (moved, fixed) == (0, 1)


def test_adjacent_events_allow_a_cut_at_the_shared_boundary():
    ev = pd.DataFrame({'sample': ['S', 'S'], 'chrom': 'chr1', 'pos': 'internal',
                       'start': [0, 400], 'end': [400, 1000], 'width': [400, 600]})
    outcomes = set()
    for seed in range(16):
        out, _, _ = permute_events(ev, seed, bounds={'chr1': (0, 1000, 1100, 2000)})
        assert (out.end <= 1000).all()
        outcomes.add(tuple(out.start))
    assert outcomes == {(0, 400), (600, 0)}



def test_rotation_samples_each_legal_integer_cut_once():
    from spice.tsg_og.permutation import _rotation_offset
    starts, ends = [1, 1, 4, 7], [3, 5, 6, 9]
    legal = [cut for cut in range(10)
             if not any(start < cut < end for start, end in zip(starts, ends))]

    class FixedDraw:
        def __init__(self, draw):
            self.draw = draw

        def integers(self, high):
            assert high == len(legal)
            return self.draw

    offsets = [_rotation_offset(starts, ends, 0, 10, FixedDraw(draw))
               for draw in range(len(legal))]
    assert sorted(offsets) == sorted((-cut) % 10 for cut in legal)


class TestChromosomeHybrid:
    @pytest.mark.parametrize('bounds', [(0, 4, 6, 15), (4, 0, 6, 15), (0, 4, 15, 15), (0, 4, 4, 15)])
    def test_all_legal_integer_starts_sampled_exactly_once(self, bounds):
        from spice.tsg_og.permutation import _hybrid_start_ranges, _draw_start
        p_lo, p_hi, q_lo, q_hi = bounds
        arms = [(lo, hi) for lo, hi in [(p_lo,p_hi),(q_lo,q_hi)] if hi > lo]
        for width in range(1, 18):
            allow_bridge = len(arms)==2 and width>min(hi-lo for lo,hi in arms)
            legal = [start for start in range(16) if
                     any(lo <= start and start+width <= hi for lo,hi in arms) or
                     (allow_bridge and p_lo <= start <= p_hi and q_lo <= start+width <= q_hi)]
            ranges = _hybrid_start_ranges(width, bounds)
            class FixedDraw:
                def __init__(self, draw): self.draw=draw
                def integers(self, high):
                    assert high==len(legal)
                    return self.draw
            if legal:
                assert [_draw_start(ranges,FixedDraw(i)) for i in range(len(legal))]==legal
            else:
                assert _draw_start(ranges,FixedDraw(0)) is None

    def test_short_events_move_between_arms_and_long_events_bridge(self):
        ev=pd.DataFrame({'sample':['S']*400,'chrom':'chr1','pos':'internal',
                         'start':60,'width':[20]*200+[60]*200,'type':['gain','loss']*200})
        ev['end']=ev.start+ev.width
        out,moved,fixed=permute_events(ev,7,mode='chromosome_hybrid',bounds={'chr1':(0,40,50,140)})
        short,long=out.iloc[:200],out.iloc[200:]
        assert (short.end<=40).any() and (short.start>=50).any()
        assert ((short.end<=40)|(short.start>=50)).all()
        assert ((long.start<=40)&(long.end>=50)).any()
        assert (long.start>=50).any()
        assert ((out.start<=40)|(out.start>=50)).all()
        assert ((out.end<=40)|(out.end>=50)).all()
        assert (out.start>=0).all() and (out.end<=140).all()
        np.testing.assert_array_equal(out.end-out.start,ev.width)
        pd.testing.assert_frame_equal(out.drop(columns=['start','end']),ev.drop(columns=['start','end']))
        assert moved+fixed==len(ev)
        repeated=permute_events(ev,7,mode='chromosome_hybrid',bounds={'chr1':(0,40,50,140)})[0]
        pd.testing.assert_frame_equal(out,repeated)
        other=permute_events(ev,8,mode='chromosome_hybrid',bounds={'chr1':(0,40,50,140)})[0]
        assert not out.start.equals(other.start)

    def test_single_arm_noninternal_and_unplaceable_events(self):
        ev=pd.DataFrame({'sample':['S']*3,'chrom':'chr1','pos':['internal','internal','whole_arm'],
                         'start':[60,0,0],'width':[20,160,40],'end':[80,160,40]})
        out,moved,fixed=permute_events(ev,7,mode='chromosome_hybrid',bounds={'chr1':(20,0,50,140)})
        assert 50<=out.start.iloc[0] and out.end.iloc[0]<=140
        pd.testing.assert_frame_equal(out.iloc[1:],ev.iloc[1:])
        assert moved+fixed==2

    def test_invalid_geometry_and_width_rejected(self):
        from spice.tsg_og.permutation import _hybrid_start_ranges
        for width,bounds in [(0,(0,40,50,140)),(1.5,(0,40,50,140)),(10,(0,60,50,140)),(10,(0,np.nan,50,140))]:
            with pytest.raises(ValueError): _hybrid_start_ranges(width,bounds)

    def test_null_provenance_and_chromosome_calibration(self):
        from spice.tsg_og.permutation import validate_null_mode, validate_permutation_strategy
        hybrid=_loci().assign(permutation_mode='chromosome_hybrid')
        null=null_from_loci([hybrid])
        validate_null_mode(null,'chromosome_hybrid')
        validate_permutation_strategy('chromosome_hybrid','zpool_chrom')
        validate_permutation_strategy('chromosome_hybrid','perchrom')
        with pytest.raises(ValueError): validate_permutation_strategy('chromosome_hybrid','zpool')
        with pytest.raises(ValueError): permutation_p(_loci(),null,'zpool')
        assert np.isfinite(permutation_p(_loci(),null,'zpool_chrom')).all()
        with pytest.raises(ValueError): validate_null_mode(null,'rotate')
        with pytest.raises(ValueError): validate_null_mode(null.drop(columns='permutation_mode'),'chromosome_hybrid')
        with pytest.raises(ValueError): null_from_loci([hybrid,_loci()])
        with pytest.raises(ValueError): null_from_loci([hybrid,hybrid.assign(permutation_mode='rotate')])
