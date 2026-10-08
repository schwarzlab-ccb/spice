"""Deterministic spatial clustering with at most one peak from each detection seed.

Sweep the one-dimensional nearest-peak boundaries to propose representative sets.
No p/q values enter proposal, ranking, or tie breaking. Clusters are extracted
by support, then squared distance to their final centroid; this is a deterministic
greedy packing, not a global optimum over all possible component partitions.
"""
from math import fsum

import numpy as np
import pandas as pd



COMPONENT_COLUMNS = [
    'component_id', 'chrom', 'type', 'n_peaks', 'n_seeds', 'supporting_seeds', 'missing_seeds', 'min_peaks_per_seed',
    'max_peaks_per_seed', 'multiple_peaks_per_seed', 'exceeds_distance_cutoff',
    'status', 'mean_pos_bp', 'start', 'end', 'extent_bp', 'center_span_bp',
    'seed_mean_variance_bp2', 'variance_of_mean_bp2', 'sem_bp']
SEED_COLUMNS = ['component_id', 'seed', 'n_peaks', 'mean_pos_bp', 'start', 'end']


SCALES = ('small', 'mid1', 'mid2', 'large')
DEFAULT_MAX_SPANS_MB = (1., 2., 4., 8.)
DEFAULT_SCALE_FITNESS_FRACTION = .25


def validate_spans(max_spans_bp, fraction):
    if (set(max_spans_bp) != set(SCALES) or
            not np.isfinite(list(max_spans_bp.values())).all() or
            min(max_spans_bp.values()) <= 0 or
            any(max_spans_bp[a] > max_spans_bp[b] for a, b in zip(SCALES, SCALES[1:]))):
        raise ValueError('Require positive, finite, nondecreasing spans for small, mid1, mid2, large')
    if not np.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError('Scale fitness fraction must be between 0 and 1')


def directional_fitness(frame):
    """Validate all joint tracks and return nonnegative same-direction fitness."""
    if not frame.type.isin(['OG', 'TSG']).all():
        raise ValueError('Unknown locus direction')
    gains = frame[[f'fitness_{scale}_gain' for scale in SCALES]].to_numpy(dtype=float)
    losses = frame[[f'fitness_{scale}_loss' for scale in SCALES]].to_numpy(dtype=float)
    if not np.isfinite(gains).all() or not np.isfinite(losses).all():
        raise ValueError('Nonfinite joint fitness')
    return np.maximum(np.where((frame.type == 'OG').to_numpy()[:, None], gains, losses), 0)


def assign_spans(frame, max_spans_bp, fraction=DEFAULT_SCALE_FITNESS_FRACTION):
    """Shortest scale with positive directional fitness >= fraction of its maximum.

    Use gain fitness for OG and loss fitness for TSG, ignoring negative fitness.
    Zero fraction means any positive support; one means the strongest scale
    (ties resolve to the shortest). No p/q values enter this assignment.
    """
    validate_spans(max_spans_bp, fraction)
    directed = directional_fitness(frame)
    maximum = directed.max(axis=1)
    if (maximum <= 0).any():
        raise ValueError('Joint locus has no positive fitness in its declared direction')
    qualifying = (directed > 0) & (directed >= fraction * maximum[:, None])
    scales = np.array(SCALES)[qualifying.argmax(axis=1)]
    spans = np.array([max_spans_bp[scale] for scale in scales], dtype=float)
    return scales, spans


def describe_component(group, cid, parent, seeds, distance_bp, min_support):
    grouped = group.groupby('seed', sort=True)
    counts, means = grouped.size(), grouped.pos.mean()
    assert len(counts) >= min_support and set(counts.index) <= set(seeds)
    span = float(group.pos.max() - group.pos.min())
    assert span <= distance_bp
    variance = float(means.var(ddof=1))
    component = {'component_id': cid, 'parent_component_id': parent,
        'chrom': group.iloc[0].chrom, 'type': group.iloc[0]['type'],
        'n_peaks': len(group), 'n_seeds': len(counts),
            'supporting_seeds': ','.join(map(str, counts.index)),
            'missing_seeds': ','.join(map(str, sorted(set(seeds) - set(counts.index)))),
        'min_peaks_per_seed': int(counts.min()), 'max_peaks_per_seed': int(counts.max()),
        'multiple_peaks_per_seed': bool(counts.max() > 1), 'exceeds_distance_cutoff': False,
        'status': 'multiple_per_seed' if counts.max() > 1 else 'unambiguous',
        'mean_pos_bp': float(means.mean()), 'start': float(group.start.min()),
        'end': float(group.end.max()), 'extent_bp': float(group.end.max() - group.start.min()),
        'center_span_bp': span, 'seed_mean_variance_bp2': variance,
        'variance_of_mean_bp2': variance / len(counts), 'sem_bp': float(np.sqrt(variance / len(counts)))}
    per_seed = [{'component_id': cid, 'seed': seed, 'n_peaks': len(part),
                 'mean_pos_bp': float(part.pos.mean()), 'start': float(part.start.min()),
                 'end': float(part.end.max())} for seed, part in grouped]
    return component, per_seed


def _seed_arrays(nodes, available):
    """One canonical row at each identical seed/position; ties use geometry then row ID."""
    result = []
    for _, group in nodes.loc[available].groupby('seed', sort=True):
        group = group.drop_duplicates('pos')
        result.append((group.index.to_numpy(), group.pos.to_numpy(dtype=float)))
    return result


def _nearest(arrays, centers, radius):
    """Return nearest available node per seed; equal distances favor lower coordinates."""
    centers = np.asarray(centers)
    result = np.full((len(centers), len(arrays)), -1, dtype=int)
    for col, (ids, positions) in enumerate(arrays):
        right = np.searchsorted(positions, centers)
        left = np.maximum(right - 1, 0)
        right = np.minimum(right, len(positions) - 1)
        pick = np.where(abs(positions[left] - centers) <= abs(positions[right] - centers), left, right)
        inside = abs(positions[pick] - centers) <= np.nextafter(radius, np.inf)
        result[inside, col] = ids[pick[inside]]
    return result


def _proposals(nodes, available, span, radius, support):
    arrays = _seed_arrays(nodes, available)
    if len(arrays) < support:
        return []
    # Assignments change only at same-seed Voronoi midpoints and radius endpoints.
    boundaries = []
    for _, positions in arrays:
        boundaries.extend([positions - radius, positions + radius,
                           positions[:-1] + np.diff(positions) / 2])
    boundaries = np.unique(np.concatenate(boundaries))
    probes = np.unique(np.concatenate([boundaries, boundaries[:-1] + np.diff(boundaries) / 2]))
    assignments = _nearest(arrays, probes, radius)
    assignments = np.unique(assignments[(assignments >= 0).sum(axis=1) >= support], axis=0)
    positions = nodes.pos.to_numpy()
    seen, proposals = set(), []
    for assignment in assignments:
        ordered = sorted(assignment[assignment >= 0], key=lambda i: (positions[i], i))
        # Consider spatially contiguous subsets too: a farthest seed may violate
        # the span bound even when all nearest representatives fit the radius.
        for first in range(len(ordered) - support + 1):
            for stop in range(first + support, len(ordered) + 1):
                ids = tuple(sorted(ordered[first:stop]))
                if ids in seen:
                    continue
                seen.add(ids)
                x = positions[list(ids)]
                center = fsum(sorted(x)) / len(x)
                width = float(x.max() - x.min())
                if width > np.nextafter(span, np.inf) or max(abs(x - center)) > np.nextafter(radius, np.inf):
                    continue
                final = set(_nearest(arrays, [center], radius)[0])
                if not set(ids) <= final:
                    continue  # Recentered representatives must still be the closest for their seeds.
                sse = fsum(sorted((x - center) ** 2))
                geometry = tuple(sorted(nodes.loc[list(ids), ['pos', 'start', 'end']].itertuples(index=False, name=None)))
                proposals.append(((-len(ids), sse, width, center, geometry), ids))
    return proposals


def cluster_peaks(frames, distance_bp, min_support=None, centroid_radius_bp=None,
                  max_spans_bp=None, scale_fitness_fraction=DEFAULT_SCALE_FITNESS_FRACTION):
    """Cluster same-chromosome/direction peaks without p/q-dependent membership.

    distance_bp bounds total center span. The centroid radius defaults to half
    that span. Representatives are closest among peaks still available when a
    cluster is extracted; rejected alternatives can form later supported clusters.
    With max_spans_bp, each joint locus receives a scale-dependent limit and a
    component obeys its tightest member limit. These are total spans only; the
    search radius at each proposal tier equals its span and adds no centroid cutoff.
    Loci with no positive directional fitness have no defined scale. Keep them
    in the discarded-node audit, without changing any original p/q values.
    Legacy graph_nodes/graph_components filenames are retained for score/plot
    compatibility, but no graph, support pruning, or chain splitting is used.
    """
    seeds = sorted(frames)
    support = len(seeds) if min_support is None else min_support
    if isinstance(support, bool) or not isinstance(support, (int, np.integer)) or not 1 <= support <= len(seeds):
        raise ValueError('Minimum centroid support must be an integer between 1 and the number of study seeds')
    support = int(support)
    adaptive = max_spans_bp is not None
    if adaptive:
        validate_spans(max_spans_bp, scale_fitness_fraction)
        if centroid_radius_bp is not None:
            raise ValueError('Fixed per-scale spans do not accept an additional centroid radius')
        distance_bp = max(max_spans_bp.values())
        radius = distance_bp
    else:
        radius = distance_bp / 2 if centroid_radius_bp is None else centroid_radius_bp
    if len(seeds) < 2 or not np.isfinite([distance_bp, radius]).all() or min(distance_bp, radius) < 0:
        raise ValueError('Need at least two seeds and finite nonnegative span/radius limits')
    chunks = []
    for seed in seeds:
        frame = frames[seed]
        if not frame.index.is_unique:
            raise ValueError(f'Seed {seed}: duplicate peak row IDs')
        if (not frame['type'].isin(['OG', 'TSG']).all() or frame.chrom.isna().any()
                or not np.isfinite(frame[['pos', 'start', 'end']]).all().all()
                or (frame.start > frame.pos).any() or (frame.pos > frame.end).any()):
            raise ValueError(f'Seed {seed}: invalid peak coordinates')
        part = frame[['chrom', 'type', 'pos', 'start', 'end', 'q_value']].copy()
        if adaptive:
            eligible = directional_fitness(frame).max(axis=1) > 0
            part['clustering_scale'] = ''
            part['max_member_span_bp'] = np.nan
            scales, spans = assign_spans(frame.loc[eligible], max_spans_bp, scale_fitness_fraction)
            part.loc[eligible, 'clustering_scale'] = scales
            part.loc[eligible, 'max_member_span_bp'] = spans
        part['seed'], part['peak_row'] = seed, frame.index
        part['_row_key'] = frame.index.map(repr)
        part['_width'] = part.end - part.start
        chunks.append(part)
    nodes = pd.concat(chunks, ignore_index=True).sort_values(
        ['chrom', 'type', 'pos', '_width', 'start', 'end', 'seed', '_row_key']).reset_index(drop=True)
    nodes = nodes.drop(columns=['_row_key', '_width'])
    nodes.insert(0, 'node_id', np.arange(len(nodes)))
    nodes['retained'] = False
    nodes['component_id'] = ''
    nodes['centroid_distance_bp'] = np.nan
    nodes['discard_reason'] = 'insufficient_support_or_geometry'
    if adaptive:
        nodes.loc[nodes.max_member_span_bp.isna(), 'discard_reason'] = 'no_positive_directional_fitness'
    components, seed_rows, choices = [], [], []
    extraction = 0
    span_limits = nodes.max_member_span_bp.to_numpy() if adaptive else None
    for _, group in nodes.groupby(['chrom', 'type'], sort=True):
        if adaptive:
            group = group[group.max_member_span_bp.notna()]
        available = group.index.to_numpy()
        while len(available) >= support:
            if adaptive:
                proposals = []
                # Propose each span tier separately. A tight focal locus between
                # broad candidates must neither join nor block their component.
                for limit in sorted(set(span_limits[available])):
                    eligible = available[span_limits[available] >= limit]
                    proposals.extend((priority, ids, limit) for priority, ids in
                                     _proposals(nodes, eligible, limit, limit, support))
            else:
                proposals = _proposals(nodes, available, distance_bp, radius, support)
            if not proposals:
                break
            choice = min(proposals, key=lambda entry: entry[0])
            _, chosen = choice[:2]
            proposal_span = float(choice[2]) if adaptive else distance_bp
            extraction += 1
            cid = f'cluster_{extraction:04d}'
            component_span = float(span_limits[list(chosen)].min()) if adaptive else distance_bp
            component, per_seed = describe_component(nodes.loc[list(chosen)], cid, '', seeds, component_span, support)
            if adaptive:
                component['max_member_span_bp'] = component_span
                component['proposal_span_bp'] = proposal_span
                component['member_clustering_scales'] = ','.join(sorted(
                    set(nodes.loc[list(chosen), 'clustering_scale'])))
            center = fsum(sorted(nodes.loc[list(chosen), 'pos'])) / len(chosen)
            component['mean_pos_bp'] = center
            component['max_centroid_distance_bp'] = max(abs(nodes.loc[list(chosen), 'pos'] - center))
            component['extraction_order'] = extraction
            components.append(component)
            seed_rows.extend(per_seed)
            for row in nodes.loc[available].itertuples():
                distance = abs(row.pos - center)
                eligible = not adaptive or row.max_member_span_bp >= proposal_span
                search_radius = proposal_span if adaptive else radius
                if eligible and distance <= np.nextafter(search_radius, np.inf):
                    choices.append(dict(component_id=cid, node_id=row.node_id, seed=row.seed,
                                        distance_bp=distance, selected=row.node_id in chosen))
            nodes.loc[list(chosen), 'retained'] = True
            nodes.loc[list(chosen), 'component_id'] = cid
            nodes.loc[list(chosen), 'centroid_distance_bp'] = abs(nodes.loc[list(chosen), 'pos'] - center)
            nodes.loc[list(chosen), 'discard_reason'] = ''
            available = available[~np.isin(available, chosen)]
    columns = COMPONENT_COLUMNS + ['parent_component_id', 'max_centroid_distance_bp', 'extraction_order']
    if adaptive:
        columns += ['max_member_span_bp', 'member_clustering_scales', 'proposal_span_bp']
    components = pd.DataFrame(components, columns=columns)
    tables = dict(graph_nodes=nodes, graph_components=components, merged_peaks=components.copy(),
                  flagged_components=components.iloc[:0].copy(),
                  component_seed_summary=pd.DataFrame(seed_rows, columns=SEED_COLUMNS),
                  cluster_choices=pd.DataFrame(choices, columns=['component_id', 'node_id', 'seed', 'distance_bp', 'selected']),
                  discarded_peaks=nodes[~nodes.retained].copy())
    audit = dict(algorithm='nearest-per-seed constrained centroid clustering v1', seeds=seeds,
                 min_support=support, distance_bp=distance_bp, centroid_radius_bp=radius,
                 input_peaks=len(nodes), retained_peaks=int(nodes.retained.sum()), components=len(components),
                 representative='nearest available peak to final centroid; ties: lower position, narrower interval, start/end, stable row ID',
                 objective='greedy: maximum represented seeds, minimum squared distances, minimum span, lower center, geometric signature',
                 p_values_used_for_membership=False, missing_seeds='omitted; no imputation',
                 per_seed={str(seed): dict(input=len(frames[seed]), retained=int((nodes.retained & (nodes.seed == seed)).sum()))
                           for seed in seeds})
    if adaptive:
        audit.update(algorithm='nearest-per-seed constrained centroid clustering with joint scale spans v3',
                     distance_bp=None, centroid_radius_bp=None, max_spans_bp=max_spans_bp,
                     scale_fitness_fraction=scale_fitness_fraction,
                     scale_assignment='shortest scale with positive directional fitness >= fraction of maximum',
                     component_span_rule='minimum member span limit; no additional centroid-radius constraint',
                     per_scale_input=nodes.loc[nodes.max_member_span_bp.notna(), 'clustering_scale'].value_counts().sort_index().to_dict(),
                     no_positive_directional_fitness=int(nodes.max_member_span_bp.isna().sum()),
                     zero_fitness_policy='Retain original scores in discarded nodes; exclude from spatial grouping because no scale qualifies',
                     fitness_used_for_scale_assignment=True,
                     representative='nearest available eligible peak per seed at each span tier; ties use geometry',
                     eligibility='member span limit >= proposal span tier')
    return tables, audit
