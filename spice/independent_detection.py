"""Run the native cascade independently for each gain/loss pair."""
from pathlib import Path
from spice.scale_modes import SCALES, activate_scale
from spice.utils import save_pickle
from spice.random_state import derive_seed, seed_task


def run_independent_scales(detector, params):
    root = Path(params['loci_results_dir']) / 'detection' / params['cur_chrom']
    # Never leave an old aggregate looking complete while its scale models change.
    for name in ('final_selection_points', 'final_loci_widths', 'final_locus_scales'):
        (root / f'{name}.pickle').unlink(missing_ok=True)
    results = {}
    for i, scale in enumerate(SCALES):
        args = dict(params, _active_length_scale=scale)
        # Preprocessing is shared; overwrite it once, before the first model.
        args['overwrite_preprocessing'] = params['overwrite_preprocessing'] and i == 0
        results[scale] = detector(**args)
    # Only publish a complete final model when the requested stages reach final widths.
    # A partial/resumed stage must not make a stale aggregate look newly complete.
    if not all(result.get('final_loci_widths') is not None for result in results.values()):
        for name in ('final_selection_points', 'final_loci_widths', 'final_locus_scales'):
            (root / f'{name}.pickle').unlink(missing_ok=True)
        return {'scales': results}
    points = [[] for _ in range(8)]
    widths, labels = [], []
    for scale in SCALES:
        result = results[scale]
        for track, values in zip(points, result['final_selection_points']):
            track.extend(values)
        widths.extend(result['final_loci_widths'])
        labels.extend([scale] * len(result['final_loci_widths']))
    for name, value in [('final_selection_points', points), ('final_loci_widths', widths),
                        ('final_locus_scales', labels)]:
        save_pickle(value, str(root / f'{name}.pickle'))
    return dict(scales=results, final_selection_points=points, final_loci_widths=widths)


def refit_independent_scales(optimizer, chrom, points, data, kept_scales, removed_scales, iterations):
    """Refit only changed scales, preserving the order of the combined table."""
    for scale in SCALES:
        indices = [i for i, value in enumerate(kept_scales) if value == scale]
        if scale not in removed_scales or not indices:
            continue
        seed_task(derive_seed('loci_detection', chrom, scale, 'post_filter_refit'))
        subset = [[track[i] for i in indices] for track in points]
        fitted, _ = optimizer(cur_chrom=chrom, final_selection_points=subset,
                              data_per_length_scale=activate_scale(data, scale),
                              n_neighbors_optimization=10,
                              N_iterations_optimization=iterations, max_pos_change=1e5)
        for track, fitted_track in zip(points, fitted):
            for i, value in zip(indices, fitted_track):
                track[i] = value
    return points
