"""Fixed-position post-filter fitness fitting, with a joint chromosome model."""
import numpy as np

REFIT_METHODS = ('joint', 'neighborhood')


def iteration_unit(method):
    if method not in REFIT_METHODS:
        raise ValueError(f'post_filter_refit_method must be one of {REFIT_METHODS}, got {method!r}')
    return 'per_changed_model' if method == 'joint' else 'per_locus_neighborhood'


def joint_optimization_step(cur_chrom, final_selection_points, data_per_length_scale,
                            N_iterations_optimization=100_000,
                            n_neighbors_optimization=10, max_pos_change=0):
    """One global annealing call, retaining the individual-fit sign/zero constraints.

    A model is one chromosome.
    Positions are always fixed. Accept the returned state only if its recomputed
    native loss improves on the starting model, as in each neighborhood fit.
    The unused neighborhood/position arguments allow the existing scale adapter
    to call either optimizer without altering its skip/order/seed behavior.
    """
    from spice.tsg_og.detection import _optimize_selection_points, calc_mse_loss
    from spice.tsg_og.simulation import copy_list_of_selection_points, convolution_simulation_per_ls

    if N_iterations_optimization < 1:
        raise ValueError('Post-filter refit iterations must be positive')
    points = copy_list_of_selection_points(final_selection_points)
    if not points[0]:
        return points, []
    fitness = np.array([[p[0].fitness for p in track] for track in points])
    positions = np.array([[p[0].pos for p in track] for track in points])
    if not np.isfinite(fitness).all():
        raise ValueError('Nonfinite starting fitness')
    allowed = fitness != 0
    up = (fitness[::2] > 0).any(axis=0)
    initial_loss = float(calc_mse_loss(data_per_length_scale,
        convolution_simulation_per_ls(cur_chrom, data_per_length_scale, points)))
    if not np.isfinite(initial_loss):
        raise ValueError('Nonfinite starting refit loss')
    proposal, reported_loss, _ = _optimize_selection_points(
        N_iterations_optimization, list(zip(*points)), data_per_length_scale, cur_chrom,
        best_loss=initial_loss, loci_to_optimize=None, N_iterations_base=0,
        allow_pos_change=False, max_pos_change=0, up_down_order=up.tolist(),
        allowed_fitness_change=allowed, max_fitness=1000, max_deviation=0.0001)
    proposal = [list(track) for track in zip(*proposal)]
    new_positions = np.array([[p[0].pos for p in track] for track in proposal])
    new_fitness = np.array([[p[0].fitness for p in track] for track in proposal])
    if not np.array_equal(new_positions, positions):
        raise ValueError('A fixed locus position moved during post-filter refitting')
    if not np.isfinite(new_fitness).all():
        return points, [initial_loss, initial_loss]
    if (new_fitness[~allowed] != 0).any():
        raise ValueError('A locked zero fitness changed during post-filter refitting')
    # Existing values can predate sign constraints; only enforce the optimizer's
    # direction on entries it actually changed, preserving untouched input fits.
    changed = new_fitness != fitness
    signs = np.where(up, 1, -1)[None, :] * np.array([1, -1]*4)[:, None]
    if (new_fitness[changed] * signs[changed] < 0).any():
        raise ValueError('A locus changed direction during post-filter refitting')
    loss = float(calc_mse_loss(data_per_length_scale,
        convolution_simulation_per_ls(cur_chrom, data_per_length_scale, proposal)))
    if not np.isfinite(loss):
        return points, [initial_loss, initial_loss]
    np.testing.assert_allclose(reported_loss, loss, rtol=1e-10, atol=1e-10)
    if loss < initial_loss:
        return proposal, [initial_loss, loss]
    return points, [initial_loss, initial_loss]
