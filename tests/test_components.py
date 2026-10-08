"""Native components preserve production grouping, selection and fitting semantics."""
from copy import deepcopy
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm
import yaml

import spice
from spice.components import (FITNESS, TRACKS, group_components, fit_components,
                              run_components, stouffer_q)
from spice.length_scales import DEFAULT_SEGMENT_SIZE_DICT as SEG

SPANS = dict(small=1., mid1=2., mid2=4., large=8.)


def frame(positions, fits=None, q=.01):
    pos = np.asarray(positions, dtype=float)
    result = pd.DataFrame(dict(chrom='chr21', type='OG', pos=pos, start=pos-100,
                               end=pos+100, p_value=q/2, q_value=q,
                               p_values_method='combined_fitness'))
    fit = np.tile([1, -.5, 0, 0, 0, 0, 0, 0], (len(pos), 1)) if fits is None else np.asarray(fits)
    for i, col in enumerate(FITNESS):
        result[col] = fit[:, i]
    return result


def group(frames, **kwargs):
    return group_components(frames, SPANS, min_support=kwargs.pop('min_support', len(frames)), **kwargs)


def test_selection_changes_no_membership_and_counts_one_per_seed():
    frames = {s: frame([22e6+s*10000, 30e6+s*10000]) for s in range(3)}
    frames[0] = pd.concat([frames[0], frame([22e6+2000])], ignore_index=True)
    all_, selected, members, _, audit = group(frames)
    assert all_.n_seeds.tolist() == [3, 3, 1]
    assert selected.n_peaks.tolist() == [3, 3]
    assert not members.duplicated(['component_id', 'seed']).any()
    assert len(members) == 7
    assert all_.combined_p.isna().all() and all_.q_value.isna().all()
    assert selected.stouffer_q_score.iloc[0] == pytest.approx(norm.sf(norm.isf(.01)*np.sqrt(3)))
    changed = {s: f.assign(q_value=.99, p_value=.98) for s, f in frames.items()}
    other, chosen, nodes, _, _ = group(changed, min_support=2)
    pd.testing.assert_frame_equal(members[['seed', 'peak_row', 'component_id']], nodes[['seed', 'peak_row', 'component_id']])
    assert chosen.empty
    assert audit['p_values_used_for_membership'] is False
    assert group(frames, score_threshold=1e-30)[1].empty


@pytest.mark.parametrize('i,span', list(enumerate([1, 2, 4, 8])))
def test_scale_spans_are_inclusive_without_centroid_radius(i, span):
    fit = np.zeros((1, 8)); fit[0, i*2] = 1
    frames = {s: frame([20e6 + v*span*1e6], fit) for s, v in enumerate([0, .9, 1])}
    assert group(frames)[0].n_seeds.tolist() == [3]
    frames[2].loc[0, ['pos', 'start', 'end']] += 1
    assert group(frames)[0].n_seeds.max() < 3


def test_zero_fitness_is_discarded_and_tight_member_limits_span():
    frames = {0: frame([20e6, 40e6], [[1, 0, 0, 0, 0, 0, 0, 0], [0]*8]),
              1: frame([23e6], [[0, 0, 0, 0, 0, 0, 1, 0]])}
    all_, selected, _, discarded, _ = group(frames)
    assert all_.n_seeds.tolist() == [1, 1]
    assert selected.empty
    assert discarded.discard_reason.tolist() == ['no_positive_directional_fitness']


@pytest.mark.parametrize('damage', ['duplicate', 'score', 'method', 'direction', 'fitness'])
def test_invalid_seed_tables_rejected(damage):
    frames = {s: frame([22e6, 30e6]) for s in range(2)}
    if damage == 'duplicate': frames[0].index = [0, 0]
    if damage == 'score': frames[0].loc[0, 'q_value'] = np.nan
    if damage == 'method': frames[0]['p_values_method'] = 'mean_fitness'
    if damage == 'direction': frames[0].loc[0, 'type'] = 'other'
    if damage == 'fitness': frames[0].loc[0, FITNESS[0]] = np.inf
    with pytest.raises(ValueError): group(frames)


@pytest.fixture
def model():
    from spice.tsg_og.detection import CHROM_LENS
    data = {}
    for i, (s, d) in enumerate(TRACKS):
        grid = np.arange(int(CHROM_LENS.loc['chr21']) // SEG[s])*SEG[s]
        signal = 10 + (8 if d == 'gain' else -3)*np.exp(-((grid-22e6)/1e6)**2)
        data[(s, d)] = dict(chrom='chr21', signals=signal, length_scale=s, type=d,
            length_scale_i=i, cur_widths=np.array([2e6]), loci_width=max(4, int(2e6/SEG[s])),
            kernel=np.ones(int(np.ceil(4e6/SEG[s]))), non_centromere_index=np.arange(len(signal)),
            cur_loss_norm=10., height_multiplier=np.ones(len(signal)), centromere_values={},
            signal_bounds=(signal-.1, signal+.1), signal_upsampling=SEG[s]/SEG['small'])
    return data


def test_native_fit_reproducible_fixed_geometry_and_nonincreasing_loss(model):
    frames = {s: frame([22e6+s*1000]) for s in range(2)}
    all_, _, members, _, _ = group(frames)
    first, points, _, audit = fit_components(all_, members, frames, 'chr21', model, 30, 9)
    second, _, _, _ = fit_components(all_, members, frames, 'chr21', model, 30, 9)
    pd.testing.assert_frame_equal(first, second)
    np.testing.assert_array_equal(first[['pos', 'start', 'end']], all_[['pos', 'start', 'end']])
    assert audit['final_loss'] <= audit['initial_loss']
    assert (first[FITNESS[::2]] >= 0).all().all()
    assert (first[FITNESS[1::2]] <= 0).all().all()
    for tr in points:
        assert tr[0][0].pos == first.pos.iloc[0]


def test_all_eight_tracks_can_move_including_initial_zeros(model, monkeypatch):
    from spice.tsg_og import detection
    frames = {s: frame([22e6]) for s in range(2)}
    all_, _, members, _, _ = group(frames)
    actual = detection._optimize_selection_points
    calls = []
    def checked(*args, **kwargs):
        calls.append(kwargs)
        assert kwargs['allowed_fitness_change'].shape == (8, 1)
        assert kwargs['allowed_fitness_change'].all()
        assert kwargs['allow_pos_change'] is False
        return actual(*args, **kwargs)
    monkeypatch.setattr(detection, '_optimize_selection_points', checked)
    fit_components(all_, members, frames, 'chr21', model, 2, 9)
    assert len(calls) == 1


def config_fixture(tmp_path, model):
    cfg = deepcopy(spice.default_config)
    cfg.update(name='test_components', directories=dict(base_dir=str(tmp_path), results_dir=str(tmp_path/'results'), log_dir=str(tmp_path/'logs')))
    cfg['params']['seed'] = 9
    cfg['input_files'] = {k: str(Path(__file__).parent/'objects'/f'{k}.tsv')
                          for k in ['centromeres_observed', 'telomeres_observed']}
    cfg['input_files']['component_loci'] = {}
    for seed in range(3):
        path = tmp_path/f'seed_{seed}.tsv'
        table = frame([22e6+1000*seed, 30e6+1000*seed])
        table.loc[1, 'q_value'] = .99
        table.to_csv(path, sep='\t')
        cfg['input_files']['component_loci'][seed] = path.name
    folder = tmp_path/'model/data_per_length_scale'; folder.mkdir(parents=True)
    with (folder/'chr21.pickle').open('wb') as handle: pickle.dump(model, handle)
    cfg['input_files']['component_model_dir'] = 'model'
    cfg['components'].update(n_seeds=3, reference_seed=2, refit_iterations=2)
    cfg['components']['selection'].update(min_support=3)
    return cfg


def test_yaml_cli_end_to_end_and_refuse_overwrite(tmp_path, model):
    cfg = config_fixture(tmp_path, model)
    path = tmp_path/'components.yaml'; path.write_text(yaml.safe_dump(cfg))
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).parents[1]), SPICE_CONFIG=str(path),
               OPENBLAS_NUM_THREADS='1', MPLBACKEND='Agg')
    command = [sys.executable, '-m', 'spice.cli', 'components', '--config', str(path), '--seed', '17']
    result = subprocess.run(command, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    out = tmp_path/'results/test_components/components'
    all_ = pd.read_csv(out/'components_all.tsv', sep='\t')
    selected = pd.read_csv(out/'components_filtered.tsv', sep='\t')
    assert len(all_) == 2 and len(selected) == 1
    assert all_.n_peaks.tolist() == [3, 3]
    assert set(FITNESS) <= set(all_.columns) & set(selected.columns)
    audit = json.loads((out/'audit.json').read_text())
    assert audit['status'] == 'complete'
    assert audit['fits']['all']['chr21']['components'] == 2
    assert audit['fits']['filtered']['chr21']['components'] == 1
    assert audit['fits']['filtered']['chr21']['seed'] == 17
    before = (out/'components_all.tsv').read_bytes()
    result = subprocess.run(command, env=env, capture_output=True, text=True)
    assert result.returncode != 0 and 'already exists' in result.stderr
    assert (out/'components_all.tsv').read_bytes() == before


@pytest.mark.parametrize('key,value', [('n_seeds', 4), ('refit_iterations', 0),
    ('selection', dict(method='unknown', min_support=3, threshold=.05)),
    ('selection', dict(method='stouffer_q', min_support=4, threshold=.05))])
def test_bad_configuration_creates_no_output(tmp_path, model, key, value):
    cfg = config_fixture(tmp_path, model)
    cfg['components'][key] = value
    with pytest.raises(ValueError): run_components(cfg, 9)
    assert not (tmp_path/'results').exists()


@pytest.mark.parametrize('zero_fitness', [False, True])
def test_empty_selected_or_all_sets_still_export_headers_and_baseline_fit(tmp_path, model, zero_fitness):
    cfg = config_fixture(tmp_path, model)
    for name in cfg['input_files']['component_loci'].values():
        path = tmp_path/name
        table = pd.read_csv(path, sep='\t', index_col=0)
        table['q_value'] = 1.
        if zero_fitness: table[FITNESS] = 0.
        table.to_csv(path, sep='\t')
    output = run_components(cfg, 9)
    assert pd.read_csv(output/'components_filtered.tsv', sep='\t').empty
    all_ = pd.read_csv(output/'components_all.tsv', sep='\t')
    assert len(all_) == (0 if zero_fitness else 2)
    audit = json.loads((output/'audit.json').read_text())
    assert audit['status'] == 'complete'
    assert audit['fits']['filtered']['chr21']['components'] == 0
