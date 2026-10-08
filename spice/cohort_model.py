"""Prepare one seed-independent cohort model and consume it without changing its caches.

CLI workers are separate processes: the temporary base-seed context must not be
used concurrently with other work in the same process.
"""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import shutil
import tempfile

import numpy as np
import pandas as pd

PREPROCESSING = ('remove_plateaus', 'remove_chrY', 'drop_duplicates', 'use_observed_centromeres')
SOURCES = ('cohort_model.py', 'loci_preprocessing.py', 'length_scales.py', 'segmentation.py',
           'data_loaders.py', 'random_state.py', 'tsg_og/detection.py', 'tsg_og/simulation.py',
           'tsg_og/signal_bootstrap.py', 'tsg_og/plateaus.py')
MODEL_COLUMN = 'cohort_model_id'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def resolve(config, path):
    if not isinstance(path, str) or not path:
        raise ValueError('Shared cohort model requires nonempty input paths')
    p = Path(path).expanduser()
    return (p if p.is_absolute() else Path(config['directories']['base_dir'])/p).resolve()


def policy(config):
    """Identity excludes detection/refit RNG, name and output paths."""
    settings = deepcopy(config['cohort_model'])
    for k in ('seed', 'N_bootstrap_report'):
        v = settings[k]
        if isinstance(v, bool) or not isinstance(v, int) or v < (0 if k == 'seed' else 1):
            raise ValueError(f'cohort_model.{k} must be a valid nonnegative seed or positive count')
    if settings['seed'] >= 2**32:
        raise ValueError('cohort_model.seed must be smaller than 2**32')
    params = config['loci_detection']
    for k in ('N_bootstrap', 'N_kernel'):
        if isinstance(params[k], bool) or not isinstance(params[k], int) or params[k] < 1:
            raise ValueError(f'loci_detection.{k} must be a positive integer')
    inputs = config['input_files']
    hashes = {k: digest(resolve(config, inputs[k])) for k in
              ('final_events', 'centromeres_observed', 'telomeres_observed')}
    hashes['plateaus'] = digest(resolve(config, inputs['plateaus'])) if inputs.get('plateaus') else None
    # Include packaged geometry/plateau tables too, including assembly-specific fallbacks.
    objects = Path(__file__).parent/'objects'
    assets = {str(p.relative_to(objects)): digest(p) for p in sorted(objects.rglob('*.tsv'))}
    return dict(schema=1, assembly=config['params']['assembly'], settings=settings,
                preprocessing={k: params.get(k, True) for k in PREPROCESSING},
                N_bootstrap=params['N_bootstrap'], N_kernel=params['N_kernel'], inputs=hashes,
                source_sha256={p: digest(Path(__file__).parent/p) for p in SOURCES},
                asset_sha256=assets)


def events_digest(events, chrom):
    # Order is intentional: resampling draws row indices. Ignore only the pandas index.
    frame = events.loc[events.chrom.eq(chrom)].reset_index(drop=True)
    return hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest()


@contextmanager
def model_random_seed(seed):
    """Use an independent model RNG, restoring all caller streams even on errors."""
    from spice import random_state as rng
    rng.get_seed()
    saved = (rng._base_seed, dict(vars(rng._local)), os.environ.get(rng.SEED_ENV_VAR),
             np.random.get_state(), random.getstate())
    try:
        rng.set_seed(seed)
        yield
    finally:
        rng._base_seed, local, env, np_state, py_state = saved
        vars(rng._local).clear()
        vars(rng._local).update(local)
        if env is None:
            os.environ.pop(rng.SEED_ENV_VAR, None)
        else:
            os.environ[rng.SEED_ENV_VAR] = env
        np.random.set_state(np_state)
        random.setstate(py_state)


def prepare_data(events, chrom, root, n_bootstrap, n_kernel, overwrite=False):
    """The historical preprocessing calls, kept identical for the unshared path."""
    from spice.tsg_og.signal_bootstrap import bootstrap_sampling_of_signal
    from spice.tsg_og.detection import collect_data_per_length_scale
    bootstrap_sampling_of_signal(cur_chrom=chrom, final_events_df=events, N_bootstrap=n_bootstrap,
        calc_new_force_new=overwrite,
        calc_new_filename=str(Path(root)/'signal_bootstrap'/f'{chrom}_N_{n_bootstrap}.pickle'))
    return collect_data_per_length_scale(events, chrom, N_bootstrap=n_bootstrap, N_kernel=n_kernel,
        loci_results_dir=str(root), calc_new_force_new=overwrite,
        calc_new_filename=str(Path(root)/'data_per_length_scale'/f'{chrom}.pickle'))


def build_model(config, chroms=None):
    """Build into a new directory, publishing only after every chromosome succeeds."""
    from spice.data_loaders import load_final_events
    from spice.loci_preprocessing import process_final_events_for_loci_routines
    from spice.tsg_og.signal_bootstrap import bootstrap_sampling_of_signal, get_signal_bootstrap_bounds
    from spice.production_compatibility import validate_config, validate_data
    validate_config(config['loci_detection'])
    if not config.get('name') or config['name'] in ('.', '..') or Path(config['name']).name != config['name']:
        raise ValueError('Config name must be a nonempty directory name')
    output = Path(config['directories']['results_dir'])/config['name']/'cohort_model'
    if output.exists():
        raise ValueError(f'Cohort model already exists; use a new name/results directory: {output}')
    spec = policy(config)
    events = process_final_events_for_loci_routines(load_final_events(), **spec['preprocessing'])
    chroms = sorted(set(events.chrom)) if chroms is None else list(chroms)
    if not chroms or len(set(chroms)) != len(chroms) or not set(chroms).issubset(set(events.chrom)):
        raise ValueError('Requested model chromosomes must be distinct and present in processed events')
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.cohort_model-', dir=output.parent))
    manifest = dict(status='complete', policy=spec, model_id=identity(spec), chromosomes={}, files={})
    try:
        with model_random_seed(spec['settings']['seed']):
            for chrom in chroms:
                data = prepare_data(events, chrom, stage, spec['N_bootstrap'], spec['N_kernel'])
                validate_data(data)
                n_report = spec['settings']['N_bootstrap_report']
                bootstrap_sampling_of_signal(cur_chrom=chrom, final_events_df=events, N_bootstrap=n_report,
                    calc_new_filename=str(stage/'signal_bootstrap'/f'{chrom}_N_{n_report}.pickle'))
                bounds = get_signal_bootstrap_bounds(chrom, str(stage), N_bootstrap=n_report)
                report = deepcopy(data)
                for value in report.values():
                    value['signal_bounds'] = bounds[value['length_scale_i']]
                target = stage/'report'/'data_per_length_scale'/f'{chrom}.pickle'
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open('wb') as f:
                    pickle.dump(report, f)
                manifest['chromosomes'][chrom] = events_digest(events, chrom)
        for p in sorted(stage.rglob('*.pickle')):
            manifest['files'][str(p.relative_to(stage))] = digest(p)
        if policy(config) != spec:
            raise ValueError('Cohort inputs or model sources changed during preparation')
        import yaml
        (stage/'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
        (stage/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        stage.rename(output)
    except BaseException:
        shutil.rmtree(stage)
        raise
    return output


def read_model(config):
    root = resolve(config, config['input_files']['cohort_model_dir'])
    manifest = json.loads((root/'manifest.json').read_text())
    spec = policy(config)
    if (manifest.get('status') != 'complete' or manifest.get('policy') != spec
            or manifest.get('model_id') != identity(spec)):
        raise ValueError('Shared cohort model does not match inputs, model settings or source; rebuild it')
    return root, manifest


def checked_file(root, manifest, relative):
    path = root/relative
    if relative not in manifest['files'] or not path.is_file() or digest(path) != manifest['files'][relative]:
        raise ValueError(f'Missing or changed shared cohort model file: {path}')
    return path


def detection_data(config, events, chrom, root, n_bootstrap, n_kernel, overwrite=False, permutation=False):
    """Stage private observed copies, or generate a permutation's own fixed-seed model."""
    root = Path(root)
    marker = root/'cohort_model'/f'{chrom}.json'
    shared = config.get('input_files', {}).get('cohort_model_dir')
    if not shared:
        if marker.exists():
            raise ValueError('This detection cache uses a shared model; restore its configuration')
        return prepare_data(events, chrom, root, n_bootstrap, n_kernel, overwrite)
    source, manifest = read_model(config)
    spec = manifest['policy']
    if (n_bootstrap, n_kernel) != (spec['N_bootstrap'], spec['N_kernel']):
        raise ValueError('Detection bootstrap/kernel counts must match the shared cohort model')
    if overwrite:
        raise ValueError('Shared preprocessing is immutable; build a new model instead')
    event_hash = events_digest(events, chrom)
    if chrom not in manifest['chromosomes']:
        raise ValueError(f'Chromosome absent from shared cohort model: {chrom}')
    if not permutation and event_hash != manifest['chromosomes'][chrom]:
        raise ValueError(f'Observed events do not match shared cohort model: {chrom}')
    expected = dict(model_id=manifest['model_id'], events_sha256=event_hash,
                    role='permutation' if permutation else 'observed')
    relative = [f'data_per_length_scale/{chrom}.pickle', f'signal_bootstrap/{chrom}_N_{n_bootstrap}.pickle']
    if marker.exists():
        saved = json.loads(marker.read_text())
        if any(saved.get(k) != v for k, v in expected.items()):
            raise ValueError('Detection model identity changed; use a fresh output directory')
        for rel in relative:
            checked_file(root, saved, rel)
    else:
        if any((root/rel).exists() for rel in relative) or (root/'detection'/chrom).exists():
            raise ValueError('Unidentified detection caches; use a fresh output directory')
        if permutation:
            with model_random_seed(spec['settings']['seed']):
                prepare_data(events, chrom, root, n_bootstrap, n_kernel)
        else:
            for rel in relative:
                src = checked_file(source, manifest, rel)
                dst = root/rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dst)  # No writable links into the shared artifact.
        expected['files'] = {rel: digest(root/rel) for rel in relative}
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps(expected, indent=2)+'\n')
    with (root/relative[0]).open('rb') as f:
        return pickle.load(f)


def table_model_id(frame):
    if MODEL_COLUMN not in frame:
        return None
    ids = frame[MODEL_COLUMN].drop_duplicates()
    if len(ids) != 1 or ids.isna().any() or not isinstance(ids.iloc[0], str):
        raise ValueError('Missing or mixed cohort model identities in table')
    return ids.iloc[0]


def check_scoring_models(observed, null):
    if len(observed) and table_model_id(observed) != table_model_id(null):
        raise ValueError('Observed loci and permutation null use different cohort models; regenerate the null')


def tag_loci(config, root, chroms, frame, events):
    """Validate saved model identity even on combine-only invocations."""
    markers = [Path(root)/'cohort_model'/f'{c}.json' for c in chroms]
    shared = config.get('input_files', {}).get('cohort_model_dir')
    if not shared:
        if any(p.exists() for p in markers):
            raise ValueError('Shared model caches require their original configuration')
        return
    _, manifest = read_model(config)
    roles = set()
    for chrom, p in zip(chroms, markers):
        if not p.is_file():
            raise ValueError(f'Missing shared model identity for {chrom}')
        m = json.loads(p.read_text())
        if m['model_id'] != manifest['model_id'] or events is None or m['events_sha256'] != events_digest(events, chrom):
            raise ValueError('Cannot combine mismatching cohort model caches/events')
        roles.add(m['role'])
        for rel in m['files']:
            checked_file(Path(root), m, rel)
    if len(roles) != 1:
        raise ValueError('Cannot combine observed and permutation caches')
    frame[MODEL_COLUMN] = manifest['model_id']
