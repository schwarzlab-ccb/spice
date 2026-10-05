"""Paired opposite-direction fitness must use the same statistic in observed and null."""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import false_discovery_control

from spice.tsg_og.permutation import (fitness_statistic, scoring_statistic,
    scoring_per_ls, null_from_loci, permutation_p)
from spice.tsg_og.loci import assign_p_values
from spice.production_compatibility import validate_config


def loci(a, b, direction='OG'):
    d = pd.DataFrame(dict(chrom='chr1', type=direction, pos=np.arange(len(a))*1e6+20e6))
    for scale in ['small','mid1','mid2','large']:
        d[f'fitness_{scale}_gain'] = a if direction == 'OG' else -np.asarray(b)
        d[f'fitness_{scale}_loss'] = -np.asarray(b) if direction == 'OG' else a
    return d


@pytest.mark.parametrize('direction', ['OG', 'TSG'])
def test_equal_weight_opposite_magnitude_and_zero_same_direction(direction):
    d = loci([0., 1., 2.], [3., .5, 0.], direction)
    np.testing.assert_array_equal(scoring_statistic(d, 'combined_fitness'), [3., 1.5, 2.])
    np.testing.assert_array_equal(fitness_statistic(d), [0., 1., 2.])
    np.testing.assert_array_equal(scoring_per_ls(d, 'combined_fitness'), np.array([[3.,1.5,2.]]*4).T)
    assert scoring_statistic(d.iloc[:0], 'combined_fitness').shape == (0,)


def test_pooled_null_and_bh_use_paired_combined_values():
    raw = loci([0., 1., 2., 3.], [4., 0., 0., 0.])
    observed = loci([0., 2.], [3.5, 0.])
    null = null_from_loci([raw], method='combined_fitness')
    np.testing.assert_array_equal(null.stat, [4., 1., 2., 3.])
    assert null.p_values_method.eq('combined_fitness').all()
    assert 'fitness_small_loss' in null
    scored = assign_p_values(observed, null, strategy='zpool_chrom', method='combined_fitness')
    expected = np.array([2/5, 4/5])  # inclusive tail, +1, same-stratum monotone transform
    np.testing.assert_allclose(scored.p_value_raw, expected)
    np.testing.assert_allclose(scored.p_value, false_discovery_control(expected))
    assert scored.p_values_method.eq('combined_fitness').all()


def test_scoring_identity_rejects_mixed_or_legacy_null_for_combined_method():
    d = loci([0., 1., 2.], [.1, .2, .3])
    legacy = null_from_loci([d]).drop(columns='p_values_method')
    with pytest.raises(ValueError, match='Untagged'):
        permutation_p(d, legacy, 'zpool_chrom', method='combined_fitness')
    combined = null_from_loci([d], method='combined_fitness')
    with pytest.raises(ValueError, match='does not match'):
        permutation_p(d, combined, 'zpool_chrom', method='mean_fitness')
    combined.loc[0,'p_values_method'] = 'mean_fitness'
    with pytest.raises(ValueError, match='does not match'):
        permutation_p(d, combined, 'zpool_chrom', method='combined_fitness')
    validate_config(dict(p_values_method='combined_fitness', p_values_strategy='zpool_chrom'))


def test_combined_nonfinite_opposite_fitness_cannot_enter_the_null():
    d = loci([0., 1.], [1., np.nan])
    with pytest.raises(ValueError, match='Nonfinite'):
        null_from_loci([d], method='combined_fitness')
