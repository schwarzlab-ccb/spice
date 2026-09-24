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
    # CI output is unrelated to orchestration and requires real fitted caches.
    from spice.tsg_og import detection
    monkeypatch.setattr(detection, 'calc_genome_wide_within_ci',
                        lambda *args, **kwargs: (0.5, pd.DataFrame({'within_ci': [0.5]}, index=['chr1'])))
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


@pytest.mark.parametrize('mode',['rotate','uniform','chromosome_hybrid'])
def test_preprocessing_order_preserves_hybrid_bridges(monkeypatch,mode):
    raw=pd.DataFrame({'chrom':['chr1'],'sample':['S'],'pos':['internal'],
                      'start':[60],'end':[120],'width':[60]})
    calls=[]
    def preprocess(final_events_df,**kwargs):
        calls.append('preprocess')
        # Model the location-dependent filter that would discard a new bridge.
        return final_events_df[final_events_df.start>=50].copy()
    def permute(frame,seed,mode):
        calls.append('permute')
        return frame.assign(start=10,end=70),1,0
    monkeypatch.setattr(main_loci_functions,'process_final_events_for_loci_routines',preprocess)
    monkeypatch.setattr(permutation,'permute_events',permute)
    result,_,_=permutation.prepare_permutation_events(raw,seed=1,mode=mode,loci_params={})
    if mode=='chromosome_hybrid':
        assert calls==['preprocess','permute']
        assert len(result)==1 and result.start.iloc[0]==10
    else:
        assert calls==['permute','preprocess']
        assert result.empty


@pytest.mark.parametrize('execution',['inline','scatter','pool'])
def test_hybrid_event_frame_reaches_detection_or_combine(permutation_run,monkeypatch,execution):
    args,cfg,root,events=permutation_run
    cfg['loci_detection'].update(p_values_permute_mode='chromosome_hybrid',p_values_strategy='zpool_chrom')
    processed=events.assign(start=10,end=70,width=60,pos='internal')
    calls=[]
    def prepare(frame,**kwargs):
        assert kwargs['mode']=='chromosome_hybrid'
        return processed,1,0
    def detect(**kwargs):
        assert kwargs['final_events_df'] is processed
        calls.append('detect')
    def combine(**kwargs):
        assert kwargs['processed_events'] is processed
        calls.append('combine')
        return locus_frame(),{},{},locus_frame()
    monkeypatch.setattr(permutation,'prepare_permutation_events',prepare)
    monkeypatch.setattr(main_loci_functions,'run_loci_detection_per_chrom',detect)
    monkeypatch.setattr(main_loci_functions,'combine_loci',combine)
    if execution=='scatter':args.index,args.chrom=1,'chr1'
    if execution=='pool':
        args.pool=True
        cli._check_permutation_chrom_mode(str(root/'permutations/s1'),'chr1','chromosome_hybrid',write=True)
    cli.main_permute(args)
    assert calls=={'inline':['detect','combine'],'scatter':['detect'],'pool':['combine']}[execution]
    if execution!='scatter':
        null=pd.read_csv(root/permutation.NULL_FILENAME,sep='\t')
        assert set(null.permutation_mode)=={'chromosome_hybrid'}


def test_hybrid_rejects_old_scatter_cache_and_pool(permutation_run):
    args,cfg,root,events=permutation_run
    unit=root/'permutations/s1';(unit/'detection/chr1').mkdir(parents=True)
    with pytest.raises(ValueError,match='fresh output'):
        cli._check_permutation_chrom_mode(str(unit),'chr1','chromosome_hybrid',write=True)
    cfg['loci_detection'].update(p_values_permute_mode='chromosome_hybrid',p_values_strategy='zpool_chrom')
    locus_frame().to_csv(unit/'unit_loci.tsv',sep='\t',index=False)
    args.pool=True
    with pytest.raises(ValueError,match='fresh null'): cli.main_permute(args)


def test_cached_hybrid_null_cannot_be_read_as_legacy(permutation_run):
    args,cfg,root,events=permutation_run
    root.mkdir(parents=True)
    pd.DataFrame({'stat':[1],'permutation_mode':['chromosome_hybrid']}).to_csv(root/permutation.NULL_FILENAME,sep='\t',index=False)
    with pytest.raises(ValueError,match='mode mismatch'):cli._load_permutation_null_or_none(str(root))


def test_mode_mismatch_does_not_invalidate_existing_tables(permutation_run,monkeypatch):
    args,cfg,root,events=permutation_run
    unit=root/'permutations/s1'
    cli._check_permutation_chrom_mode(str(unit),'chr1','rotate',write=True)
    (unit/'unit_loci.tsv').write_text('original unit')
    (root/permutation.NULL_FILENAME).write_text('original null')
    cfg['loci_detection'].update(p_values_permute_mode='chromosome_hybrid',p_values_strategy='zpool_chrom')
    monkeypatch.setattr(permutation,'prepare_permutation_events',lambda *a,**kw:(events,1,0))
    args.index,args.chrom=1,'chr1'
    with pytest.raises(ValueError,match='mode mismatch'):cli.main_permute(args)
    assert (unit/'unit_loci.tsv').read_text()=='original unit'
    assert (root/permutation.NULL_FILENAME).read_text()=='original null'


@pytest.mark.parametrize('execution', ['pool', 'pool_overwrite', 'inline'])
@pytest.mark.parametrize('marker_mode', [None, 'rotate'])
def test_hybrid_rejects_cached_chromosome_absent_from_processed_events(
        permutation_run, monkeypatch, execution, marker_mode):
    """chr2 is absent from events but combine_loci would still load its cache."""
    from unittest.mock import Mock
    args, cfg, root, events = permutation_run
    assert set(events.chrom) == {'chr1'}
    cfg['loci_detection'].update(p_values_permute_mode='chromosome_hybrid', p_values_strategy='zpool_chrom')
    unit = root/'permutations/s1'
    cli._check_permutation_chrom_mode(str(unit), 'chr1', 'chromosome_hybrid', write=True)
    if marker_mode:
        cli._check_permutation_chrom_mode(str(unit), 'chr2', marker_mode, write=True)
    (unit/'data_per_length_scale').mkdir()
    (unit/'data_per_length_scale/chr2.pickle').write_bytes(b'legacy cache')
    (root/permutation.NULL_FILENAME).write_text('original pooled null')
    if execution != 'pool':
        (unit/'unit_loci.tsv').write_text('original unit')
    monkeypatch.setattr(permutation, 'prepare_permutation_events', lambda *a, **kw: (events, 1, 0))
    combine = Mock()
    detect = Mock()
    monkeypatch.setattr(main_loci_functions, 'combine_loci', combine)
    monkeypatch.setattr(main_loci_functions, 'run_loci_detection_per_chrom', detect)
    args.pool = execution.startswith('pool')
    args.overwrite = execution == 'pool_overwrite'
    with pytest.raises(ValueError, match='chr2.*fresh output'):
        cli.main_permute(args)
    combine.assert_not_called()
    detect.assert_not_called()
    assert (root/permutation.NULL_FILENAME).read_text() == 'original pooled null'
    if execution != 'pool':
        assert (unit/'unit_loci.tsv').read_text() == 'original unit'
    else:
        assert not (unit/'unit_loci.tsv').exists()


def test_hybrid_pool_accepts_marked_eventless_cache_and_ignores_unconsumed_chromosomes(
        permutation_run, monkeypatch):
    from unittest.mock import Mock
    args, cfg, root, events = permutation_run
    cfg['loci_detection'].update(p_values_permute_mode='chromosome_hybrid', p_values_strategy='zpool_chrom')
    unit = root/'permutations/s1'
    cli._check_permutation_chrom_mode(str(unit), 'chr2', 'chromosome_hybrid', write=True)
    (unit/'data_per_length_scale').mkdir()
    for chrom in ['chr2', 'chrY', 'unknown']:
        (unit/f'data_per_length_scale/{chrom}.pickle').write_bytes(b'fixture cache')
    assert main_loci_functions.cached_loci_chromosomes(str(unit)) == ['chr2']
    # No chr1 cache: combination would not consume it, so no marker is required.
    monkeypatch.setattr(permutation, 'prepare_permutation_events', lambda *a, **kw: (events, 1, 0))
    combine = Mock(return_value=(locus_frame(), {}, {}, locus_frame()))
    monkeypatch.setattr(main_loci_functions, 'combine_loci', combine)
    args.pool = True
    cli.main_permute(args)
    combine.assert_called_once()
    table = pd.read_csv(root/permutation.NULL_FILENAME, sep='\t')
    assert set(table.permutation_mode) == {'chromosome_hybrid'}
