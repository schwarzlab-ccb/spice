"""Component models are prepared after clustering and independent of every seed fit."""
from copy import deepcopy
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spice import component_model as cm
from spice.components import run_components
from spice.random_state import get_seed, np_rng, set_seed
from test_components import model, config_fixture


@pytest.fixture
def fresh_config(tmp_path, model, monkeypatch):
    cfg = config_fixture(tmp_path, model)
    cfg['input_files'].pop('component_model_dir')
    cfg['components'].pop('reference_seed')
    cfg['components'].update(model_seed=123, N_bootstrap=7, N_kernel=11)
    events = pd.DataFrame(dict(chrom=['chr21'], start=[20], end=[30]))
    p = tmp_path/'events.tsv'; events.to_csv(p, sep='\t', index=False)
    cfg['input_files']['final_events'] = str(p)
    from spice import data_loaders, loci_preprocessing
    monkeypatch.setattr(data_loaders, 'load_final_events', lambda: events.copy())
    monkeypatch.setattr(loci_preprocessing, 'process_final_events_for_loci_routines', lambda data, **kw: data)
    calls = []
    def prepare(events_in, chrom, root, n_bootstrap, n_kernel):
        pd.testing.assert_frame_equal(events, events_in)
        calls.append((chrom, get_seed(), n_bootstrap, n_kernel))
        data = deepcopy(model)
        for track in data.values(): track['kernel'] = track['kernel'] * np_rng().uniform(.5, 1.)
        path = Path(root)/'data_per_length_scale'/f'{chrom}.pickle'; path.parent.mkdir(parents=True)
        path.write_bytes(pickle.dumps(data))
        return data
    monkeypatch.setattr(cm, 'prepare_data', prepare)
    return cfg, calls


def test_fresh_models_ignore_seed_model_ids_and_preserve_groups(fresh_config, model, tmp_path):
    cfg, calls = fresh_config
    cohort = cm.identity(cm.cohort_spec(cfg))
    for seed, path in cfg['input_files']['component_loci'].items():
        p = cm.resolve(cfg, path); f = pd.read_csv(p, sep='\t', index_col=0)
        f['cohort_model_id'] = f'different-model-{seed}'
        f['cohort_id'] = cohort
        f.to_csv(p, sep='\t')
    set_seed(77)
    new = run_components(cfg, 9, chrom='chr21')
    assert calls == [('chr21', 123, 7, 11)]
    audit = json.loads((new/'audit.json').read_text())
    assert audit['status'] == 'complete' and audit['reference_seed'] is None
    assert audit['model_source'] == 'events' and audit['component_model_seed'] == 123
    assert audit['cohort_validation'] == 'verified'
    saved = deepcopy(cfg); saved['name'] = 'saved'; saved['input_files']['component_model_dir'] = 'model'
    old = run_components(saved, 9, chrom='chr21')
    for kind in ('all', 'filtered'):
        a = pd.read_csv(new/f'components_{kind}.tsv', sep='\t')
        b = pd.read_csv(old/f'components_{kind}.tsv', sep='\t')
        pd.testing.assert_frame_equal(a[['component_id', 'pos', 'start', 'end', 'n_peaks', 'stouffer_q_score']],
                                      b[['component_id', 'pos', 'start', 'end', 'n_peaks', 'stouffer_q_score']])
    assert (new/'component_members.tsv').read_bytes() == (old/'component_members.tsv').read_bytes()


def test_model_only_depends_on_events_and_own_seed(fresh_config, tmp_path):
    cfg, calls = fresh_config
    set_seed(7)
    before = np_rng().get_state()
    a = cm.prepare_component_model(cfg, ['chr21'], tmp_path/'first')
    assert get_seed() == 7
    np.testing.assert_equal(np_rng().get_state(), before)
    other = deepcopy(cfg); other['params']['seed'] = 989; other['components']['n_seeds'] = 50
    other['input_files']['component_loci'] = {}  # Model builder never reads them.
    set_seed(989)
    b = cm.prepare_component_model(other, ['chr21'], tmp_path/'second')
    assert a == b and calls == [('chr21', 123, 7, 11)]*2
    with pytest.raises(ValueError, match='fresh'): cm.prepare_component_model(cfg, ['chr21'], tmp_path/'first')


def test_cohort_checks_do_not_compare_model_seeds(fresh_config):
    cfg, _ = fresh_config
    frames = {i: pd.DataFrame({'cohort_model_id':[str(i)]}) for i in range(3)}
    assert cm.validate_seed_cohorts(frames, cfg) == 'legacy_untagged'
    for frame in frames.values(): cm.tag_cohort(cfg, frame)
    assert cm.validate_seed_cohorts(frames, cfg) == 'verified'
    frames[1]['cohort_id'] = 'other'
    with pytest.raises(ValueError, match='do not match'): cm.validate_seed_cohorts(frames, cfg)
    frames[1].drop(columns='cohort_id', inplace=True)
    with pytest.raises(ValueError, match='mix'): cm.validate_seed_cohorts(frames, cfg)


@pytest.mark.parametrize('key,value', [('model_seed', -1), ('model_seed', 2**32), ('N_bootstrap', 0), ('N_kernel', True)])
def test_invalid_model_settings(fresh_config, tmp_path, key, value):
    cfg, _ = fresh_config; cfg['components'][key] = value
    with pytest.raises(ValueError): cm.prepare_component_model(cfg, ['chr21'], tmp_path/'bad')


def test_chromosome_scope_rejected_before_model_build(fresh_config):
    cfg, calls = fresh_config
    with pytest.raises(ValueError, match='absent from processed cohort events'): run_components(cfg, 9, chrom='chr22')
    assert not calls


def test_empty_requested_chromosome_still_has_model_and_ci(fresh_config):
    cfg, calls = fresh_config
    for path in cfg['input_files']['component_loci'].values():
        p = cm.resolve(cfg, path)
        frame = pd.read_csv(p, sep='\t', index_col=0).iloc[:0]
        frame.to_csv(p, sep='\t')
    output = run_components(cfg, 9, chrom='chr21')
    audit = json.loads((output/'audit.json').read_text())
    assert calls == [('chr21', 123, 7, 11)]
    assert audit['status'] == 'complete'
    for label in ['all', 'filtered']:
        assert pd.read_csv(output/f'components_{label}.tsv', sep='\t').empty
        assert audit['fits'][label]['chr21']['components'] == 0
        assert np.isfinite(audit['fits'][label]['chr21']['within_ci'])
        assert (output/'fits'/label/'chr21/fitted_signal.npz').is_file()
