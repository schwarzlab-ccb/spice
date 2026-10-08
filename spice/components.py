"""Group scored seed loci and jointly fit component sets without pipeline imports.

Member q-values select components descriptively; they are not component p/FDR.
Positions and member-interval envelopes stay fixed during fitness fitting.
"""
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
import yaml

from spice._component_clustering import SCALES, cluster_peaks, validate_spans
from spice.production_compatibility import validate_config, validate_data, validate_table

TRACKS = tuple((s, d) for s in SCALES for d in ('gain', 'loss'))
FITNESS = [f'fitness_{s}_{d}' for s, d in TRACKS]


def stouffer_q(values):
    """Equal-weight combination of represented member q-values, without extra BH."""
    q = np.asarray(values, dtype=float)
    if q.ndim != 1 or not len(q) or not np.isfinite(q).all() or ((q < 0) | (q > 1)).any():
        raise ValueError('Expected finite member q-values in [0, 1]')
    return float(norm.sf(norm.isf(np.clip(q, 1e-15, 1-1e-15)).sum() / np.sqrt(len(q))))


def group_components(frames, spans_mb, scale_fitness_fraction=.25, min_support=10,
                     score_threshold=.05):
    """Return all groups, selected groups, members, discarded loci and grouping audit."""
    if (isinstance(min_support, bool) or not isinstance(min_support, int)
            or not 2 <= min_support <= len(frames)):
        raise ValueError('min_support must be an integer between 2 and the number of seeds')
    if not np.isfinite(score_threshold) or not 0 < score_threshold <= 1:
        raise ValueError('score_threshold must be in (0, 1]')
    if any(isinstance(s, bool) or not isinstance(s, int) or s < 0 for s in frames):
        raise ValueError('Seed IDs must be nonnegative integers')
    for seed, frame in frames.items():
        validate_table(frame)
        required = ['chrom', 'type', 'pos', 'start', 'end', 'p_value', 'q_value'] + FITNESS
        if not set(required) <= set(frame.columns):
            raise ValueError(f'Seed {seed}: missing locus columns: {sorted(set(required)-set(frame.columns))}')
        if (not np.isfinite(frame[['p_value', 'q_value']]).all().all()
                or not frame[['p_value', 'q_value']].apply(lambda x: x.between(0, 1).all()).all()):
            raise ValueError(f'Seed {seed}: invalid p/q values')
        if 'p_values_method' not in frame or not frame.p_values_method.eq('combined_fitness').all():
            raise ValueError(f'Seed {seed}: expected combined_fitness scoring provenance')
    spans = {s: v * 1e6 for s, v in spans_mb.items()}
    validate_spans(spans, scale_fitness_fraction)
    # Group all raw candidates, including singletons. Support filters selection only.
    tables, audit = cluster_peaks(frames, max(spans.values()), min_support=1,
                                 max_spans_bp=spans, scale_fitness_fraction=scale_fitness_fraction)
    nodes = tables['graph_nodes']
    nodes['p_value'] = [frames[r.seed].loc[r.peak_row, 'p_value'] for r in nodes.itertuples()]
    members = nodes[nodes.retained].copy()
    groups = tables['graph_components'].copy()
    score = members.groupby('component_id').q_value.apply(stouffer_q)
    groups['stouffer_q_score'] = groups.component_id.map(score).astype(float)
    groups['selected'] = (groups.n_seeds >= min_support) & (groups.stouffer_q_score < score_threshold)
    groups['pos'] = groups.mean_pos_bp
    groups['width'] = groups.end - groups.start
    groups['combined_p'] = np.nan
    groups['q_value'] = np.nan
    audit.update(min_selection_support=min_support, score_threshold=score_threshold,
                 selection='descriptive Stouffer member q; not component p/FDR',
                 width_interpretation='member positional-interval envelope; not signal extent')
    return groups, groups[groups.selected].copy(), members, tables['discarded_peaks'], audit


def initial_fitness(components, members, frames):
    """Mean signed member fitness, projected onto each component's gain/loss direction."""
    values = []
    for row in components.itertuples():
        group = members[members.component_id == row.component_id]
        if len(group) != row.n_seeds or group.seed.duplicated().any():
            raise ValueError('Expected one member per supporting seed')
        values.append(np.mean([frames[m.seed].loc[m.peak_row, FITNESS].to_numpy(float)
                               for m in group.itertuples()], axis=0))
    initial = np.asarray(values).reshape((-1, 8))
    signs = np.where(components.type.to_numpy()[:, None] == 'OG', [1, -1]*4, [-1, 1]*4)
    return signs * np.maximum(signs * initial, 0)


def fit_components(components, members, frames, chrom, data, iterations=100000, seed=9):
    """Fit one chromosome/set, resetting its RNG as in the production pipeline."""
    from spice.random_state import set_seed
    from spice.tsg_og.simulation import SelectionPoints, convolution_simulation_per_ls
    from spice.tsg_og.detection import _optimize_selection_points, calc_mse_loss, calc_within_ci_bootstrap
    from spice.tsg_og.permutation import fitness_statistic, scoring_statistic

    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
        raise ValueError('refit_iterations must be a positive integer')
    validate_data(data)
    if set(data) != set(TRACKS):
        raise ValueError('Reference model must contain all eight signal tracks')
    data = {key: data[key] for key in TRACKS}
    for key, track in data.items():
        signal = np.asarray(track['signals'])
        bounds = np.asarray(track['signal_bounds'])
        indices = track['non_centromere_index']
        if (track['chrom'] != chrom or (track['length_scale'], track['type']) != key
                or signal.ndim != 1 or bounds.shape != (2, len(signal))
                or not np.isfinite(signal).all() or not np.isfinite(bounds).all()
                or (bounds[0] > bounds[1]).any() or not signal[indices].size
                or not np.isfinite(track['cur_loss_norm']) or track['cur_loss_norm'] <= 0):
            raise ValueError('Invalid component signal model or CI bounds')
    selected = components[components.chrom == chrom].sort_values(['mean_pos_bp', 'component_id']).copy()
    selected = selected.reset_index(drop=True)
    initial = initial_fitness(selected, members, frames)
    if not np.isfinite(initial).all():
        raise ValueError('Nonfinite initial component fitness')
    set_seed(seed)
    points = [[SelectionPoints(loci=[(pos, initial[j, i])])
               for j, pos in enumerate(selected.pos)] for i in range(8)]
    conv = convolution_simulation_per_ls(chrom, data, points if len(selected) else None)
    initial_loss = loss = float(calc_mse_loss(data, conv))
    if not np.isfinite(loss):
        raise ValueError('Nonfinite initial component loss')
    if len(selected):
        fitted, reported_loss, _ = _optimize_selection_points(
            iterations, list(zip(*points)), data, chrom, best_loss=loss,
            max_pos_change=0, allow_pos_change=False, up_down_order=(selected.type == 'OG').tolist(),
            allowed_fitness_change=np.ones((8, len(selected)), dtype=bool),
            max_fitness=max(1000., float(initial.max())*1.1), max_deviation=1e-5, N_iterations_base=0)
        proposal = [list(track) for track in zip(*fitted)]
        proposal_conv = convolution_simulation_per_ls(chrom, data, proposal)
        proposal_loss = float(calc_mse_loss(data, proposal_conv))
        np.testing.assert_allclose(reported_loss, proposal_loss, rtol=1e-10, atol=1e-10)
        if np.isfinite(proposal_loss) and proposal_loss < loss:
            points, loss, conv = proposal, proposal_loss, proposal_conv
    fitness = np.array([[track[j][0].fitness for track in points] for j in range(len(selected))]).reshape((-1, 8))
    positions = np.array([[track[j][0].pos for track in points] for j in range(len(selected))]).reshape((-1, 8))
    signs = np.where(selected.type.to_numpy()[:, None] == 'OG', [1, -1]*4, [-1, 1]*4)
    if (not np.array_equal(positions, selected.pos.to_numpy()[:, None] * np.ones((1, 8)))
            or not np.isfinite(fitness).all() or (fitness * signs < 0).any()):
        raise ValueError('Component fitness, fixed position or direction constraint violated')
    for i, col in enumerate(FITNESS):
        selected['initial_' + col] = initial[:, i]
        selected[col] = fitness[:, i]
    selected['mean_fitness'] = fitness_statistic(selected)
    selected['combined_fitness'] = scoring_statistic(selected)
    ci = calc_within_ci_bootstrap(data, conv, exclude_zero_signal=False)
    audit = dict(chrom=chrom, components=len(selected), seed=seed, iterations=iterations,
                 initial_loss=initial_loss, final_loss=loss,
                 within_ci=float(np.mean(ci)),
                 within_ci_tracks=dict(zip([f'{s}_{d}' for s, d in TRACKS], map(float, ci))))
    return selected, points, conv, audit


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def run_components(config, seed):
    """Consume YAML-configured scored tables and saved reference models; write native outputs."""
    validate_config(config['loci_detection'])
    settings = config['components']
    if (not config.get('name') or config['name'] in ('.', '..')
            or Path(config['name']).name != config['name']):
        raise ValueError('Config name must be a nonempty directory name')
    base = Path(config['directories']['base_dir'])
    def resolve(value):
        if not isinstance(value, str) or not value:
            raise ValueError('Component input paths must be nonempty strings')
        path = Path(value).expanduser()
        return (path if path.is_absolute() else base / path).resolve()
    paths = config.get('input_files', {}).get('component_loci')
    if not isinstance(paths, dict) or not paths:
        raise ValueError('input_files.component_loci must map integer seed IDs to unfiltered TSV paths')
    n_seeds = settings['n_seeds']
    if isinstance(n_seeds, bool) or not isinstance(n_seeds, int) or n_seeds < 2 or len(paths) != n_seeds:
        raise ValueError('components.n_seeds must be >=2 and match the number of seed tables')
    selection = settings['selection']
    if selection['method'] != 'stouffer_q':
        raise ValueError('components.selection.method must be stouffer_q')
    paths = {s: resolve(p) for s, p in paths.items()}
    if len(set(paths.values())) != len(paths):
        raise ValueError('Each seed must supply a distinct locus table')
    frames = {s: pd.read_csv(p, sep='\t', index_col=0, float_precision='round_trip') for s, p in paths.items()}
    groups, filtered, members, discarded, grouping = group_components(
        frames, settings['max_member_spans_mb'], settings['scale_fitness_fraction'],
        selection['min_support'], selection['threshold'])
    shared = config['input_files'].get('cohort_model_dir')
    if not shared and settings['reference_seed'] not in frames:
        raise ValueError('reference_seed must be present in component_loci')
    iterations = settings['refit_iterations']
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
        raise ValueError('refit_iterations must be a positive integer')
    model_identity = None
    if shared:
        if config['input_files'].get('component_model_dir'):
            raise ValueError('Choose cohort_model_dir or component_model_dir, not both')
        from spice.cohort_model import read_model, checked_file, table_model_id
        model_root, manifest = read_model(config)
        model_identity = manifest['model_id']
        if any(table_model_id(frame) != model_identity for frame in frames.values() if len(frame)):
            raise ValueError('Component seed tables must come from the supplied shared cohort model')
        model = model_root / 'report'
    else:
        model = resolve(config['input_files'].get('component_model_dir'))
    chroms = sorted({c for frame in frames.values() for c in frame.chrom})
    if not chroms:
        raise ValueError('No chromosomes in seed tables; cannot infer reference model scope')
    model_paths = {c: model / 'data_per_length_scale' / f'{c}.pickle' for c in chroms}
    if shared:
        for chrom in chroms:
            checked_file(model_root, manifest, f'report/data_per_length_scale/{chrom}.pickle')
    for path in model_paths.values():
        if not path.is_file():
            raise ValueError(f'Missing component reference model: {path}')
    output = Path(config['directories']['results_dir']) / config['name'] / 'components'
    if output.exists():
        raise ValueError(f'Use a new name/results directory; component output already exists: {output}')
    hashes = {str(p): digest(p) for p in list(paths.values()) + list(model_paths.values())}
    for key in ('centromeres_observed', 'telomeres_observed'):
        if config['input_files'].get(key):
            path = resolve(config['input_files'][key])
            hashes[str(path)] = digest(path)
    output.mkdir(parents=True)
    (output / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    members.to_csv(output / 'component_members.tsv', sep='\t', index=False)
    discarded.to_csv(output / 'discarded_loci.tsv', sep='\t', index=False)
    audit = dict(status='running', input_sha256=hashes, reference_seed=None if shared else settings['reference_seed'],
                 cohort_model_id=model_identity,
                 grouping=grouping, fits={},
                 source_sha256={name: digest(Path(__file__).parent/name) for name in
                     ['components.py', '_component_clustering.py', 'tsg_og/detection.py', 'tsg_og/simulation.py']})
    (output / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    try:
        results = {'all': [], 'filtered': []}
        for chrom, path in model_paths.items():
            with path.open('rb') as handle:
                data = pickle.load(handle)
            for label, components in [('all', groups), ('filtered', filtered)]:
                fitted, points, conv, fit_audit = fit_components(
                    components, members, frames, chrom, data, iterations, seed)
                results[label].append(fitted)
                folder = output / 'fits' / label / chrom
                folder.mkdir(parents=True)
                with (folder / 'selection_points.pickle').open('wb') as handle:
                    pickle.dump(points, handle)
                np.savez_compressed(folder / 'fitted_signal.npz',
                                    **{f'{s}_{d}': v for (s, d), v in zip(TRACKS, conv)})
                audit['fits'].setdefault(label, {})[chrom] = fit_audit
        for label, tables in results.items():
            pd.concat(tables, ignore_index=True).to_csv(output / f'components_{label}.tsv', sep='\t', index=False)
        if any(digest(p) != value for p, value in hashes.items()):
            raise ValueError('Component inputs changed during execution')
        audit['status'] = 'complete'
    except Exception as error:
        audit.update(status='failed', error=str(error))
        raise
    finally:
        (output / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    return output
