"""Detection-mode provenance and active-track masks (no data-loader imports)."""
import json
from pathlib import Path
import numpy as np

SCALES = ('small', 'mid1', 'mid2', 'large')
MODES = ('joint', 'independent')


def validate_mode(mode):
    if mode not in MODES:
        raise ValueError(f'detection_scale_mode must be one of {MODES}, got {mode!r}')
    return mode


def active_tracks(data):
    return np.array([i for i, value in enumerate(data.values())
                     if value.get('fit_active', True)], dtype=int)


def activate_scale(data, scale):
    """Shallow copies: shared cached observations are never mutated."""
    if scale not in SCALES:
        raise ValueError(f'Unknown length scale: {scale}')
    return {key: dict(value, fit_active=key[0] == scale) for key, value in data.items()}


def scale_seed_key(chrom, data):
    tracks = active_tracks(data)
    return chrom if len(tracks) == 8 else f'{chrom}:{SCALES[tracks[0] // 2]}'


def check_cache_mode(root, chrom, mode, write=False):
    """An unmarked existing fit is legacy joint; never silently relabel it."""
    validate_mode(mode)
    folder = Path(root) / 'detection' / chrom
    marker = folder / 'scale_mode.json'
    if marker.exists():
        existing = json.loads(marker.read_text())['detection_scale_mode']
    elif ((folder.exists() and any(folder.rglob('*.pickle')))
          or (Path(root) / 'data_per_length_scale' / f'{chrom}.pickle').exists()
          or any((Path(root) / 'signal_bootstrap').glob(f'{chrom}_N_*.pickle'))):
        existing = 'joint'
    else:
        existing = None
    if existing is not None and existing != mode:
        raise ValueError(f'{chrom}: detection_scale_mode {existing!r} cache cannot be used '
                         f'with {mode!r}; choose a fresh output directory')
    if write:
        folder.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({'detection_scale_mode': mode}) + '\n')


def table_mode(frame):
    if 'detection_scale_mode' not in frame:
        if 'length_scale' in frame and len(frame):
            raise ValueError('Length-scale loci need detection_scale_mode provenance')
        return 'joint'
    if not len(frame):
        return None
    values = frame['detection_scale_mode'].unique()
    if len(values) != 1 or values[0] not in MODES:
        raise ValueError('Mixed or missing detection_scale_mode provenance')
    mode = values[0]
    if mode == 'independent':
        if 'length_scale' not in frame or not frame.length_scale.isin(SCALES).all():
            raise ValueError('Independent loci need a valid length_scale on every row')
    return mode


def validate_table_mode(frame, mode):
    validate_mode(mode)
    actual = table_mode(frame)
    if actual is not None and actual != mode:
        raise ValueError(f'detection_scale_mode {actual!r} table cannot be used with {mode!r}; '
                         'generate a matching permutation null')
