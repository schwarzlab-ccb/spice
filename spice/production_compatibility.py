"""Reject experimental settings and artifacts when using the main-based fixes branch.

Adapted from cache/table checks in 80cbcd5, d4dbbf4 and 1585164; intentionally
contains no independent-scale fitting or bridge-preserving geometry algorithms.
"""
import json
from pathlib import Path


def validate_config(params):
    for key, supported in [('detection_scale_mode', 'joint'),
                           ('p_values_method', 'mean_fitness')]:
        if params.get(key, supported) != supported:
            raise ValueError(f'{key} requires {supported} on fixes; experimental features '
                             'remain on anyscale-pvalue')
    from spice.tsg_og.permutation import validate_permutation_strategy
    validate_permutation_strategy(params.get('p_values_permute_mode', 'rotate'),
                                  params.get('p_values_strategy', 'zpool'))


def validate_cache(root, chrom):
    folder = Path(root) / 'detection' / chrom
    marker = folder / 'scale_mode.json'
    if marker.exists() and json.loads(marker.read_text()).get('detection_scale_mode') != 'joint':
        raise ValueError(f'{chrom}: incompatible detection_scale_mode cache; use a fresh output directory')
    if (folder / 'final_locus_scales.pickle').exists():
        raise ValueError(f'{chrom}: independent-scale cache; use a fresh output directory')


def validate_data(data):
    for value in data.values():
        if value.get('event_geometry') is not None:
            raise ValueError('Cached centromere geometry is incompatible; use a fresh output directory')
        if not value.get('fit_active', True):
            raise ValueError('Independent-scale model is incompatible; use a fresh output directory')


def validate_table(frame):
    if 'detection_scale_mode' in frame:
        if not frame.detection_scale_mode.eq('joint').all():
            raise ValueError('Incompatible detection_scale_mode table; use joint observed/null loci')
    elif 'length_scale' in frame and len(frame):
        raise ValueError('Length-scale loci require provenance; use joint observed/null loci')
