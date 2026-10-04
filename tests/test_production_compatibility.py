"""The reduced production branch must reject artifacts from excluded modes."""
import json

import pandas as pd
import pytest

from spice.production_compatibility import validate_cache, validate_config, validate_data, validate_table


def test_selected_production_and_legacy_defaults_are_accepted():
    validate_config({})
    validate_config(dict(p_values_method='mean_fitness', detection_scale_mode='joint',
                         p_values_strategy='zpool_chrom', p_values_permute_mode='rotate'))
    validate_data({'track': dict(cur_widths=[1], event_geometry=None)})
    validate_table(pd.DataFrame({'stat': [1]}))
    validate_table(pd.DataFrame({'stat': [1], 'detection_scale_mode': ['joint']}))


@pytest.mark.parametrize('params', [dict(p_values_method='any_scale'),
    dict(detection_scale_mode='independent'), dict(p_values_permute_mode='chromosome_preserve_bridges')])
def test_experimental_config_is_rejected(params):
    with pytest.raises(ValueError):
        validate_config(params)


def test_old_joint_cache_accepted_independent_marker_rejected(tmp_path):
    validate_cache(tmp_path, 'chr1')
    folder = tmp_path / 'detection/chr1'
    folder.mkdir(parents=True)
    marker = folder / 'scale_mode.json'
    marker.write_text(json.dumps({'detection_scale_mode': 'joint'}))
    validate_cache(tmp_path, 'chr1')
    marker.write_text(json.dumps({'detection_scale_mode': 'independent'}))
    with pytest.raises(ValueError, match='detection_scale_mode'):
        validate_cache(tmp_path, 'chr1')
    marker.unlink()
    (folder / 'final_locus_scales.pickle').touch()
    with pytest.raises(ValueError, match='independent-scale'):
        validate_cache(tmp_path, 'chr1')


@pytest.mark.parametrize('data', [{'event_geometry': {}}, {'fit_active': False}])
def test_experimental_background_rejected(data):
    with pytest.raises(ValueError, match='incompatible'):
        validate_data({'track': data})


@pytest.mark.parametrize('metadata', [{'detection_scale_mode': ['independent']},
                                     {'detection_scale_mode': [None]}, {'length_scale': ['small']}])
def test_scoring_rejects_experimental_observed_or_null_tables(metadata):
    from spice.tsg_og.loci import assign_p_values
    good = pd.DataFrame({'stat': [1]})
    bad = good.assign(**metadata)
    for observed, null in [(bad, good), (good, bad)]:
        with pytest.raises(ValueError, match='detection_scale_mode|provenance'):
            assign_p_values(observed, null, strategy='zpool_chrom')
