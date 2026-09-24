"""Seed consensus for independently detected loci, with native fixed-center refitting.

Clustering ports the pipeline's score-blind, nearest-per-seed greedy centroid
algorithm. Its radius is half the median observed event width in each track.
A mean member q is descriptive; it is not a calibrated component FDR.
"""
from math import fsum
import numpy as np
import pandas as pd
from spice.scale_modes import SCALES, activate_scale, validate_table_mode

FITNESS = [f'fitness_{s}_{d}' for s in SCALES for d in ('gain', 'loss')]
COMPONENT_COLUMNS = ['component_id', 'chrom', 'length_scale', 'type', 'pos', 'start', 'end',
                     'width', 'n_seeds', 'support', 'seeds', 'mean_q', 'mean_p',
                     'median_event_width', 'cluster_radius', 'max_centroid_distance',
                     'detection_scale_mode'] + FITNESS
MEMBER_COLUMNS = ['component_id', 'seed', 'peak_row', 'chrom', 'length_scale', 'type',
                  'pos', 'start', 'end', 'p_value', 'q_value']


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


def cluster_loci(frames, median_widths):
    """Partition all raw peaks, allowing one representative per seed and singletons.

    ``median_widths`` maps (chromosome, scale, OG/TSG) to median model event width.
    Every member is within radius=median/2 of the final centroid; span <= median.
    Competing peaks remain available for later components. Scores never choose
    members or break ties. Frames must be independent-mode pre-filter tables.
    """
    seeds = sorted(frames)
    if len(seeds) < 2:
        raise ValueError('Clustering requires at least two distinct seeds')
    chunks = []
    for seed in seeds:
        frame = frames[seed]
        validate_table_mode(frame, 'independent')
        required = MEMBER_COLUMNS[3:] + FITNESS
        if not frame.index.is_unique or not set(required).issubset(frame):
            raise ValueError(f'Seed {seed}: invalid peak schema or duplicate row IDs')
        numeric = ['pos', 'start', 'end', 'p_value', 'q_value'] + FITNESS
        if (not np.isfinite(frame[numeric].to_numpy(float)).all()
                or not frame.type.isin(['OG', 'TSG']).all()
                or not frame.length_scale.isin(SCALES).all()
                or not frame[['p_value', 'q_value']].apply(lambda v: v.between(0, 1).all()).all()
                or (frame.start > frame.pos).any() or (frame.pos > frame.end).any()):
            raise ValueError(f'Seed {seed}: invalid peak coordinates, fitness or scores')
        for scale in SCALES:
            other = [col for col in FITNESS if not col.startswith(f'fitness_{scale}_')]
            if (frame.loc[frame.length_scale == scale, other] != 0).any().any():
                raise ValueError('Independent loci contain fitness in another scale')
        part = frame[required].copy()
        part['seed'], part['peak_row'] = seed, frame.index
        part['_row_key'] = frame.index.map(repr)
        part['_width'] = part.end - part.start
        chunks.append(part)
    nodes = pd.concat(chunks, ignore_index=True).sort_values(
        ['chrom', 'length_scale', 'type', 'pos', '_width', 'start', 'end', 'seed', '_row_key']).reset_index(drop=True)
    components, members, tracks = [], [], []
    for (chrom, scale, direction), group in nodes.groupby(['chrom', 'length_scale', 'type'], sort=True):
        median = float(median_widths.get((chrom, scale, direction), np.nan))
        if not np.isfinite(median) or median <= 0:
            raise ValueError(f'Missing positive median event width: {chrom}/{scale}/{direction}')
        radius = median / 2
        available = group.index.to_numpy()
        extraction = 0
        while len(available):
            proposals = _proposals(nodes, available, median, radius, 1)
            if not proposals:
                raise ValueError('Singleton-inclusive clustering left unassigned peaks')
            _, chosen = min(proposals, key=lambda entry: entry[0])
            extraction += 1
            cid = f'{chrom}_{scale}_{direction}_{extraction:04d}'
            selected = nodes.loc[list(chosen)]
            center = fsum(sorted(selected.pos)) / len(selected)
            components.append(dict(component_id=cid, chrom=chrom, length_scale=scale,
                type=direction, pos=center, start=selected.start.min(), end=selected.end.max(),
                width=selected.end.max()-selected.start.min(), n_seeds=len(selected),
                support=len(selected), seeds=','.join(map(str, sorted(selected.seed))),
                mean_q=selected.q_value.mean(), mean_p=selected.p_value.mean(),
                median_event_width=median, cluster_radius=radius,
                max_centroid_distance=max(abs(selected.pos-center)),
                detection_scale_mode='independent',
                **selected[FITNESS].mean().to_dict()))
            for row in selected.to_dict('records'):
                members.append(dict(component_id=cid, **{k: row[k] for k in MEMBER_COLUMNS[1:]}))
            available = available[~np.isin(available, chosen)]
        tracks.append(dict(chrom=chrom, length_scale=scale, type=direction,
                           median_event_width=median, cluster_radius=radius,
                           input_peaks=len(group), components=extraction))
    return (pd.DataFrame(components, columns=COMPONENT_COLUMNS),
            pd.DataFrame(members, columns=MEMBER_COLUMNS),
            dict(algorithm='nearest-per-seed constrained centroid v1', tracks=tracks,
                 input_peaks=len(nodes), components=len(components), seeds=seeds,
                 min_support=1, scores_used_for_membership=False,
                 width_definition='member location interval envelope',
                 missing_seeds='omitted from mean_q; no imputation'))


def fit_fixed_loci(frame, data, chrom, seed, iterations=50_000, blocks=2, stream='components'):
    """Refit each scale with gain/loss coupled; retain all geometry and score columns.

    Return the fitted table, eight-track selection points and per-scale loss audit.
    Each optimization block is accepted only if the exact native loss improves.
    """
    from spice.random_state import derive_seed, seed_task
    from spice.tsg_og.simulation import SelectionPoints, convolution_simulation_per_ls
    from spice.tsg_og.detection import _optimize_selection_points, calc_mse_loss
    validate_table_mode(frame, 'independent')
    if iterations < 1 or blocks < 1:
        raise ValueError('iterations and blocks must be positive')
    if len(frame) and set(frame.chrom) != {chrom}:
        raise ValueError('Fixed-locus fitting takes exactly one chromosome')
    fitted = frame.copy().reset_index(drop=True)
    fitness = fitted[FITNESS].to_numpy(float)
    if not np.isfinite(fitness).all():
        raise ValueError('Nonfinite initial fitness')
    signs = np.where(fitted.type.to_numpy()[:, None] == 'OG', [1, -1]*4, [-1, 1]*4)
    fitness = signs * np.maximum(signs * fitness, 0)
    for j, scale in enumerate(SCALES):
        rows = fitted.length_scale.to_numpy() == scale
        other = [i for i in range(8) if i//2 != j]
        if (fitness[np.ix_(rows, other)] != 0).any():
            raise ValueError('Independent loci contain fitness in another scale')
    points = [[SelectionPoints(loci=[(pos, fitness[j, i])]) for j, pos in enumerate(fitted.pos)]
              for i in range(8)]
    audit = []
    for scale in SCALES:
        indices = np.flatnonzero(fitted.length_scale.to_numpy() == scale)
        if not len(indices):
            continue
        selected = [[track[j] for j in indices] for track in points]
        active = activate_scale(data, scale)
        curve = convolution_simulation_per_ls(chrom, active, selected)
        loss = float(calc_mse_loss(active, curve))
        initial_loss = loss
        history = []
        for block in range(blocks):
            seed_task(derive_seed(stream, seed, chrom, scale, block))
            proposal, reported_loss, accepted = _optimize_selection_points(
                iterations, list(zip(*selected)), active, chrom, best_loss=loss,
                allow_pos_change=False, max_pos_change=0,
                up_down_order=(fitted.iloc[indices].type == 'OG').tolist(),
                allowed_fitness_change=np.ones((8, len(indices)), dtype=bool),
                max_fitness=max(1000., float(np.max(np.abs(fitness), initial=0))*1.1),
                N_iterations_base=0, max_deviation=1e-5)
            proposal = [list(track) for track in zip(*proposal)]
            actual_loss = float(calc_mse_loss(active, convolution_simulation_per_ls(chrom, active, proposal)))
            np.testing.assert_allclose(reported_loss, actual_loss, rtol=1e-10, atol=1e-10)
            improved = np.isfinite(actual_loss) and actual_loss < loss
            history.append(dict(block=block, input_loss=loss, output_loss=actual_loss,
                                accepted_updates=len(accepted), kept=bool(improved)))
            if improved:
                selected, loss = proposal, actual_loss
        for i in range(8):
            for j, point in zip(indices, selected[i]):
                if point[0].pos != fitted.iloc[j].pos:
                    raise ValueError('A fixed component center moved')
                points[i][j] = point
                fitted.loc[j, FITNESS[i]] = point[0].fitness
        audit.append(dict(length_scale=scale, loci=len(indices), initial_loss=initial_loss,
                          final_loss=loss, history=history))
    return fitted, points, audit
