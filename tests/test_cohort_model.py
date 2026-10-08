"""Shared preprocessing is independent of detection seeds and cannot mix caches/nulls."""
from copy import deepcopy
import json
from pathlib import Path
import pickle
import random

import numpy as np
import pandas as pd
import pytest

import spice
from spice import cohort_model as cm
from spice import random_state as rng
from test_components import model, frame, config_fixture


@pytest.fixture
def bundle(tmp_path, model, monkeypatch):
    cfg = config_fixture(tmp_path, model)
    cfg['input_files'].pop('component_model_dir')
    events = pd.DataFrame(dict(chrom=['chr21']*2, start=[20, 25], end=[30, 35]))
    event_path = tmp_path/'events.tsv'; events.to_csv(event_path, sep='\t', index=False)
    cfg['input_files']['final_events'] = str(event_path)
    cfg['cohort_model'].update(seed=123, N_bootstrap_report=5)
    cfg['loci_detection'].update(N_bootstrap=2, N_kernel=4)
    from spice import data_loaders, loci_preprocessing
    from spice.tsg_og import signal_bootstrap
    monkeypatch.setattr(data_loaders, 'load_final_events', lambda: events.copy())
    monkeypatch.setattr(loci_preprocessing, 'process_final_events_for_loci_routines', lambda data, **kw: data)
    def prepare(events, chrom, root, n_bootstrap, n_kernel, overwrite=False):
        data = deepcopy(model)
        for i, value in enumerate(data.values()):
            value['length_scale_i'] = i
            value['kernel'] = value['kernel'] * rng.np_rng().uniform(.5, 1)
        p = Path(root)/'data_per_length_scale'/f'{chrom}.pickle'; p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(pickle.dumps(data))
        p = Path(root)/'signal_bootstrap'/f'{chrom}_N_{n_bootstrap}.pickle'; p.parent.mkdir(exist_ok=True)
        p.write_bytes(pickle.dumps([np.zeros((n_bootstrap, len(v['signals']))) for v in data.values()]))
        return data
    monkeypatch.setattr(cm, 'prepare_data', prepare)
    def bootstrap(**kw):
        p = Path(kw['calc_new_filename'])
        n = kw['N_bootstrap']
        p.write_bytes(pickle.dumps([rng.np_rng().uniform(1, 10, (n, len(v['signals']))) for v in model.values()]))
    monkeypatch.setattr(signal_bootstrap, 'bootstrap_sampling_of_signal', bootstrap)
    output = cm.build_model(cfg, ['chr21'])
    cfg['input_files']['cohort_model_dir'] = str(output)
    return cfg, events, output


def test_model_seed_restores_every_rng_even_on_exception():
    rng.set_seed(17)
    a = rng.np_rng().get_state(); b = rng.py_rng().getstate()
    c = np.random.get_state(); d = random.getstate()
    with pytest.raises(RuntimeError), cm.model_random_seed(123):
        assert rng.get_seed() == 123
        rng.np_rng().normal(); rng.py_rng().random(); np.random.normal(); random.random()
        raise RuntimeError()
    assert rng.get_seed() == 17
    assert rng.py_rng().getstate() == b and random.getstate() == d
    np.testing.assert_equal(rng.np_rng().get_state(), a)
    np.testing.assert_equal(np.random.get_state(), c)


def test_build_seed_independent_separate_report_bounds_no_overwrite(bundle):
    cfg, events, output = bundle
    manifest = json.loads((output/'manifest.json').read_text())
    cfg2 = deepcopy(cfg); cfg2['name'] = 'second'; cfg2['params']['seed'] = 876
    rng.set_seed(876)
    other = cm.build_model(cfg2, ['chr21'])
    assert manifest == json.loads((other/'manifest.json').read_text())
    a = pickle.loads((output/'data_per_length_scale/chr21.pickle').read_bytes())
    b = pickle.loads((output/'report/data_per_length_scale/chr21.pickle').read_bytes())
    for k in a:
        for field in a[k]:
            if field != 'signal_bounds':
                np.testing.assert_equal(a[k][field], b[k][field])
        assert not np.array_equal(a[k]['signal_bounds'], b[k]['signal_bounds'])
    with pytest.raises(ValueError, match='already exists'): cm.build_model(cfg, ['chr21'])


def test_detection_seed_copies_and_permutation_isolation(bundle, tmp_path):
    cfg, events, output = bundle
    shared_bytes = (output/'data_per_length_scale/chr21.pickle').read_bytes()
    for seed in [0, 9]:
        rng.set_seed(seed)
        dest = tmp_path/f'seed{seed}'
        cm.detection_data(cfg, events, 'chr21', dest, 2, 4)
        assert (dest/'data_per_length_scale/chr21.pickle').read_bytes() == shared_bytes
        assert not (dest/'data_per_length_scale/chr21.pickle').is_symlink()
        cm.detection_data(cfg, events, 'chr21', dest, 2, 4)  # validated reuse
        assert rng.get_seed() == seed
    permuted = events.assign(start=[18, 26])
    with pytest.raises(ValueError, match='Observed events'): cm.detection_data(cfg, permuted, 'chr21', tmp_path/'bad', 2, 4)
    cm.detection_data(cfg, permuted, 'chr21', tmp_path/'perm', 2, 4, permutation=True)
    marker = json.loads((tmp_path/'perm/cohort_model/chr21.json').read_text())
    assert marker['role'] == 'permutation' and marker['events_sha256'] == cm.events_digest(permuted, 'chr21')
    with pytest.raises(ValueError, match='identity changed'):
        cm.detection_data(cfg, events, 'chr21', tmp_path/'perm', 2, 4)
    assert (output/'data_per_length_scale/chr21.pickle').read_bytes() == shared_bytes


@pytest.mark.parametrize('damage', ['config', 'input', 'source_file', 'local_file', 'old_cache', 'overwrite', 'counts', 'disabled'])
def test_mismatches_fail_before_reusing_fit(bundle, tmp_path, damage):
    cfg, events, output = bundle
    dest = tmp_path/'run'
    cm.detection_data(cfg, events, 'chr21', dest, 2, 4)
    kw = {}
    if damage == 'config': cfg['cohort_model']['seed'] += 1
    elif damage == 'input': Path(cfg['input_files']['final_events']).write_text('different events')
    elif damage == 'source_file':
        (output/'data_per_length_scale/chr21.pickle').write_bytes(b'changed')
        dest = tmp_path/'fresh'
    elif damage == 'local_file': (dest/'data_per_length_scale/chr21.pickle').write_bytes(b'changed')
    elif damage == 'old_cache': (dest/'cohort_model/chr21.json').unlink()
    elif damage == 'overwrite': kw['overwrite'] = True
    elif damage == 'counts': cfg['loci_detection']['N_kernel'] = 999
    elif damage == 'disabled': cfg['input_files'].pop('cohort_model_dir')
    with pytest.raises(ValueError): cm.detection_data(cfg, events, 'chr21', dest, 2, 4, **kw)


def test_combine_tag_and_null_guard(bundle, tmp_path):
    cfg, events, _ = bundle
    dest = tmp_path/'run'; cm.detection_data(cfg, events, 'chr21', dest, 2, 4)
    table = frame([22e6]); cm.tag_loci(cfg, dest, ['chr21'], table, events)
    assert cm.table_model_id(table) == cm.read_model(cfg)[1]['model_id']
    with pytest.raises(ValueError, match='different cohort models'):
        cm.check_scoring_models(table, frame([22e6]))
    from spice.tsg_og.permutation import null_from_loci
    pooled = null_from_loci([table.assign(permutation_mode='rotate')]*2)
    cm.check_scoring_models(table, pooled)
    with pytest.raises(ValueError, match='mixed shared/legacy'):
        null_from_loci([table, frame([22e6])])
    with pytest.raises(ValueError, match='mismatching'):
        cm.tag_loci(cfg, dest, ['chr21'], table, events.assign(start=0))


def test_components_use_shared_report_model_without_reference_seed(bundle):
    cfg, _, output = bundle
    from spice.components import run_components
    cfg['components']['reference_seed'] = 9876  # Irrelevant with a shared model.
    mid = cm.read_model(cfg)[1]['model_id']
    with pytest.raises(ValueError, match='seed tables'): run_components(cfg, 9)
    for path in cfg['input_files']['component_loci'].values():
        p = cm.resolve(cfg, path)
        f = pd.read_csv(p, sep='\t', index_col=0); f[cm.MODEL_COLUMN] = mid; f.to_csv(p, sep='\t')
    result = run_components(cfg, 9)
    audit = json.loads((result/'audit.json').read_text())
    assert audit['status'] == 'complete' and audit['reference_seed'] is None
    assert audit['cohort_model_id'] == mid
    assert str(output/'report/data_per_length_scale/chr21.pickle') in audit['input_sha256']
