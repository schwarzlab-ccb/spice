"""Permutation CLI orchestration, with expensive fitting replaced by fixed loci."""
import argparse
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

import spice
from spice import cli, data_loaders, main_loci_functions, random_state
from spice.length_scales import LENGTH_SCALE_NAMES
from spice.tsg_og import permutation


def locus_frame(fitness=1.0):
    row = {'chrom': 'chr1', 'type': 'OG', 'pos': 20e6}
    for ls in LENGTH_SCALE_NAMES:
        row[f'fitness_{ls}_gain'] = fitness
        row[f'fitness_{ls}_loss'] = 0.0
    return pd.DataFrame([row])


@pytest.fixture
def permutation_run(tmp_path, monkeypatch):
    cfg = deepcopy(spice.config)
    cfg['name'] = 'permutation_test'
    cfg['directories'].update(base_dir=str(tmp_path), results_dir=str(tmp_path / 'results'),
                              log_dir=str(tmp_path / 'logs'))
    cfg['params']['seed'] = 73
    cfg['loci_detection']['p_values_K'] = 1
    monkeypatch.setattr(spice, 'config', cfg)
    monkeypatch.setattr(spice, 'load_config', lambda _: cfg)
    events = pd.DataFrame({'chrom': ['chr1'], 'sample': ['S']})
    monkeypatch.setattr(data_loaders, 'load_final_events', lambda: events)
    monkeypatch.setattr(main_loci_functions, 'process_final_events_for_loci_routines',
                        lambda **kwargs: events)
    monkeypatch.setattr(main_loci_functions, 'run_loci_detection_per_chrom', lambda **kwargs: None)
    monkeypatch.setattr(main_loci_functions, 'combine_loci', lambda **kwargs: (locus_frame(), {}, {}, locus_frame()))
    args = argparse.Namespace(config_path='test.yaml', debug=False, log='terminal',
                              permutations=None, mode=None, loci_steps=None, pool=False,
                              chrom=None, index=None, seed=None, overwrite=False, cores=1)
    root = tmp_path / 'results' / cfg['name'] / 'loci_of_selection'
    previous_seed = random_state.get_seed()
    random_state.set_seed(42)
    yield args, cfg, root, events
    random_state.set_seed(previous_seed)


@pytest.mark.parametrize('mode', ['inline', 'scatter', 'pool'])
@pytest.mark.parametrize('seed', [None, 7, 99])
def test_permute_applies_base_seed_in_every_mode(permutation_run, monkeypatch, mode, seed):
    args, cfg, root, events = permutation_run
    args.seed = seed
    expected_base = cfg['params']['seed'] if seed is None else seed
    random_state.set_seed(expected_base)
    expected = random_state.derive_seed('permutation', 1)
    random_state.set_seed(42)
    seen = []

    def permute(frame, seed, mode):
        seen.append((random_state.get_seed(), seed))
        return frame, 1, 0

    monkeypatch.setattr(permutation, 'permute_events', permute)
    if mode == 'scatter':
        args.index, args.chrom = 1, 'chr1'
    elif mode == 'pool':
        args.pool = True
        (root / 'permutations' / 's1').mkdir(parents=True)
    cli.main_permute(args)
    assert seen == [(expected_base, expected)]


@pytest.mark.parametrize('overwrite', [False, True])
def test_pool_overwrite_recombines_existing_unit_tables(permutation_run, monkeypatch, overwrite):
    args, cfg, root, events = permutation_run
    unit = root / 'permutations' / 's1'
    unit.mkdir(parents=True)
    locus_frame(1).to_csv(unit / 'unit_loci.tsv', sep='\t', index=False)
    calls = []

    def combine(**kwargs):
        calls.append(kwargs['loci_results_dir'])
        return locus_frame(99), {}, {}, locus_frame(99)

    monkeypatch.setattr(main_loci_functions, 'combine_loci', combine)
    monkeypatch.setattr(permutation, 'permute_events', lambda frame, **kw: (frame, 1, 0))
    args.pool, args.overwrite = True, overwrite
    cli.main_permute(args)
    assert len(calls) == int(overwrite)
    null = pd.read_csv(root / permutation.NULL_FILENAME, sep='\t')
    assert null['stat'].tolist() == [99 if overwrite else 1]


def test_scatter_invalidates_tables_before_refitting(permutation_run, monkeypatch):
    args, cfg, root, events = permutation_run
    unit = root / 'permutations' / 's1'
    unit.mkdir(parents=True)
    locus_frame(1).to_csv(unit / 'unit_loci.tsv', sep='\t', index=False)
    (root / permutation.NULL_FILENAME).write_text('old null')

    def detect(**kwargs):
        assert not (unit / 'unit_loci.tsv').exists()
        assert not (root / permutation.NULL_FILENAME).exists()

    monkeypatch.setattr(main_loci_functions, 'run_loci_detection_per_chrom', detect)
    monkeypatch.setattr(permutation, 'permute_events', lambda frame, **kw: (frame, 1, 0))
    monkeypatch.setattr(main_loci_functions, 'combine_loci', lambda **kw: (locus_frame(99), {}, {}, locus_frame(99)))
    args.index, args.chrom, args.overwrite = 1, 'chr1', True
    cli.main_permute(args)
    args.index, args.chrom, args.pool, args.overwrite = None, None, True, False
    cli.main_permute(args)
    null = pd.read_csv(root / permutation.NULL_FILENAME, sep='\t')
    assert null['stat'].tolist() == [99]


@pytest.mark.parametrize('configured', ['fast', 'full'])
@pytest.mark.parametrize('overwrite', [False, True])
def test_combine_builds_null_with_complete_detection_steps(permutation_run, monkeypatch,
                                                         configured, overwrite):
    args, cfg, root, events = permutation_run
    args.loci_steps = ['combine']
    args.overwrite = overwrite
    cfg['loci_detection']['loci_steps'] = configured
    if overwrite:
        root.mkdir(parents=True)
        (root / permutation.NULL_FILENAME).write_text('old null')
    seen = []
    monkeypatch.setattr(main_loci_functions, 'run_loci_detection_per_chrom',
                        lambda **kw: seen.append(kw['which']))
    monkeypatch.setattr(permutation, 'permute_events', lambda frame, **kw: (frame, 1, 0))
    cli.main_loci_detection(args)
    assert seen == [configured]
    assert (root.parent / 'final_loci_detection.tsv').exists()
    assert pd.read_csv(root / permutation.NULL_FILENAME, sep='\t')['stat'].tolist() == [1]


def test_resumed_detection_uses_configured_cascade_for_new_null():
    assert cli._permutation_detection_steps('final_filter_loci+', 'full') == 'full'
    complete = ['detection', 'flipping', 'final_filter_loci', 'final_loci_widths', 'combine']
    assert cli._permutation_detection_steps(complete, 'fast') == complete[:-1]
    with pytest.raises(ValueError, match='complete detection cascade'):
        cli._permutation_detection_steps('combine', 'combine')
