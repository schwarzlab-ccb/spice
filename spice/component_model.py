"""Independent event-derived models for component refitting after seed clustering."""
import json
from pathlib import Path
import shutil
import tempfile

from spice.cohort_model import (PREPROCESSING, SOURCES, digest, identity, resolve,
                                model_random_seed, prepare_data, events_digest)

COHORT_COLUMN = 'cohort_id'


def cohort_spec(config):
    inputs = config['input_files']
    files = {k: digest(resolve(config, inputs[k])) for k in
             ('final_events', 'centromeres_observed', 'telomeres_observed')}
    files['plateaus'] = digest(resolve(config, inputs['plateaus'])) if inputs.get('plateaus') else None
    return dict(assembly=config['params']['assembly'], inputs=files,
                preprocessing={k: config['loci_detection'].get(k, True) for k in PREPROCESSING})


def tag_cohort(config, frame):
    """Record cohort identity without coupling detection seeds or their models."""
    required = ('final_events', 'centromeres_observed', 'telomeres_observed')
    if all(config.get('input_files', {}).get(k) for k in required):
        frame[COHORT_COLUMN] = identity(cohort_spec(config))


def validate_seed_cohorts(frames, config):
    """Check modern table provenance; report historical inputs as unverified."""
    nonempty = [f for f in frames.values() if len(f)]
    tagged = [COHORT_COLUMN in f for f in nonempty]
    if not any(tagged):
        return 'legacy_untagged'  # Original frozen configs/provenance must establish cohort identity.
    if not all(tagged):
        raise ValueError('Cannot mix tagged and untagged seed cohort provenance')
    expected = identity(cohort_spec(config))
    if any(not f[COHORT_COLUMN].eq(expected).all() for f in nonempty):
        raise ValueError('Seed tables do not match component cohort events/bounds/preprocessing')
    return 'verified'


def prepare_component_model(config, chromosomes, output):
    """Build fresh kernels/corrections/bounds from events, without consulting seed fits."""
    from spice.data_loaders import load_final_events
    from spice.loci_preprocessing import process_final_events_for_loci_routines
    from spice.production_compatibility import validate_data
    from spice.logging import get_logger
    settings = config['components']
    counts = {k: settings[k] for k in ('model_seed', 'N_bootstrap', 'N_kernel')}
    for key, value in counts.items():
        if (isinstance(value, bool) or not isinstance(value, int)
                or value < (0 if key == 'model_seed' else 1)):
            raise ValueError(f'components.{key} must be a valid seed or positive count')
    if counts['model_seed'] >= 2**32:
        raise ValueError('components.model_seed must be smaller than 2**32')
    spec = cohort_spec(config)
    events = process_final_events_for_loci_routines(load_final_events(), **spec['preprocessing'])
    if not set(chromosomes).issubset(set(events.chrom)):
        raise ValueError('Component chromosomes are absent from processed cohort events')
    output = Path(output)
    if output.exists():
        raise ValueError('Use a fresh component model output directory')
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.component_model-', dir=output.parent))
    sources = {p: digest(Path(__file__).parent/p) for p in (*SOURCES, 'component_model.py')}
    policy = dict(cohort=spec, settings=counts, source_sha256=sources)
    manifest = dict(status='complete', source='events', model_id=identity(policy),
                    cohort_id=identity(spec), policy=policy, chromosomes={}, files={})
    try:
        with model_random_seed(counts['model_seed']):
            for chrom in chromosomes:
                get_logger('components').info(f'Preparing independent component model for {chrom}')
                data = prepare_data(events, chrom, stage, counts['N_bootstrap'], counts['N_kernel'])
                validate_data(data)
                manifest['chromosomes'][chrom] = events_digest(events, chrom)
        if cohort_spec(config) != spec or any(digest(Path(__file__).parent/p) != sha for p, sha in sources.items()):
            raise ValueError('Component model inputs or source changed during preparation')
        manifest['files'] = {str(p.relative_to(stage)): digest(p) for p in sorted(stage.rglob('*.pickle'))}
        (stage/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        stage.rename(output)
    except BaseException:
        shutil.rmtree(stage)
        raise
    return manifest
