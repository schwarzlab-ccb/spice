"""Joint refitting preserves fixed positions, locked zeros, direction and loss."""
from copy import deepcopy
import numpy as np
import pytest
from spice.length_scales import LENGTH_SCALE_NAMES as SCALES, DEFAULT_SEGMENT_SIZE_DICT as SEG
from spice.random_state import seed_task
from spice.tsg_og import detection, loci, simulation

@pytest.fixture
def model(monkeypatch):
    """Analytic convolution; native fitting/filtering decisions remain under test."""
    chrom = 'chr21'
    size = int(detection.CHROM_LENS.loc[chrom])
    data = {}
    for i, (scale, direction) in enumerate((s,d) for s in SCALES for d in ('gain','loss')):
        grid = np.arange(size // SEG[scale]) * SEG[scale]
        center = 22e6 + (i//2)*5e6
        signal = 10 + (8 if direction == 'gain' else -3)*np.exp(-((grid-center)/1e6)**2)
        data[(scale,direction)] = dict(chrom=chrom, signals=signal, length_scale=scale,
            type=direction, length_scale_i=i, cur_widths=np.array([2e6]),
            loci_width=max(4,int(2e6/SEG[scale])), kernel=np.ones(3),
            non_centromere_index=np.arange(len(signal)), cur_loss_norm=10.,
            height_multiplier=np.ones(len(signal)), centromere_values={},
            signal_bounds=(signal-.1,signal+.1), signal_upsampling=SEG[scale]/SEG['small'])
    def convolve(cur_chrom, selection_points, cur_widths, cur_length_scale, cur_signal,
                 segment_size=None, **kwargs):
        grid = np.arange(len(cur_signal))*(segment_size or SEG[cur_length_scale])
        result = np.full(len(grid),10.)
        for locus in selection_points:
            result += locus.fitness*np.exp(-((grid-locus.pos)/1e6)**2)
        return result
    for module in (simulation,detection,loci):
        monkeypatch.setattr(module,'convolution_simulation',convolve)
    return chrom,data

def points(scale, positions=(30e6,), fitness=2.):
    i=SCALES.index(scale)*2
    return [[simulation.SelectionPoints(loci=[(p,fitness if j==i else -fitness/2 if j==i+1 else 0.)])
             for p in positions] for j in range(8)]


@pytest.mark.parametrize('scale', SCALES)
def test_joint_refit_fixed_geometry_masks_and_nonincreasing_loss(model, scale):
    from spice.post_filter_refit import joint_optimization_step
    chrom, data = model
    initial = points(scale, positions=(22e6, 26e6), fitness=2.)
    # One zero in an active track must remain locked, not just inactive scales.
    i = SCALES.index(scale)*2
    initial[i+1][0] = simulation.SelectionPoints(loci=[(22e6, 0.)])
    selected = data
    before = deepcopy(initial)
    seed_task(731)
    fitted, losses = joint_optimization_step(chrom, initial, selected, N_iterations_optimization=100)
    old = np.array([[(p[0].pos, p[0].fitness) for p in tr] for tr in before])
    new = np.array([[(p[0].pos, p[0].fitness) for p in tr] for tr in fitted])
    np.testing.assert_array_equal(new[:,:,0], old[:,:,0])
    assert (new[:,:,1][old[:,:,1]==0] == 0).all()
    assert (new[::2,:,1] >= 0).all() and (new[1::2,:,1] <= 0).all()
    assert losses[-1] <= losses[0]
    assert losses[-1] == pytest.approx(detection.calc_mse_loss(selected,
        simulation.convolution_simulation_per_ls(chrom, selected, fitted)))
    np.testing.assert_array_equal(np.array([[(p[0].pos,p[0].fitness) for p in tr] for tr in initial]), old)
    # The same joint model and stream reproduce the result.
    perturbed = deepcopy(selected)
    seed_task(731)
    again, _ = joint_optimization_step(chrom, initial, perturbed, N_iterations_optimization=100)
    np.testing.assert_array_equal(np.array([[(p[0].pos,p[0].fitness) for p in tr] for tr in again]), new)


@pytest.mark.parametrize('bad', ['worse', 'nonfinite'])
def test_joint_refit_rejects_bad_returned_state(model, monkeypatch, bad):
    from spice.post_filter_refit import joint_optimization_step
    chrom, data = model
    selected = data
    initial = points('small', positions=(22e6,), fitness=2.)
    def proposal(*args, **kw):
        result = deepcopy(args[1])
        value = 999. if bad == 'worse' else float('nan')
        result[0] = list(result[0])
        result[0][0] = simulation.SelectionPoints(loci=[(22e6, value)])
        loss = detection.calc_mse_loss(selected,
            simulation.convolution_simulation_per_ls(chrom, selected, list(zip(*result))))
        return result, loss, []
    monkeypatch.setattr(detection, '_optimize_selection_points', proposal)
    fitted, losses = joint_optimization_step(chrom, initial, selected, N_iterations_optimization=3)
    assert fitted[0][0][0].fitness == 2.
    assert losses[0] == losses[-1]


def test_refit_method_validation_and_defaults():
    import spice
    from spice.post_filter_refit import iteration_unit
    assert spice.default_config['loci_detection']['post_filter_refit_method'] == 'joint'
    assert iteration_unit('joint') == 'per_changed_model'
    assert iteration_unit('neighborhood') == 'per_locus_neighborhood'
    with pytest.raises(ValueError, match='post_filter_refit_method'):
        iteration_unit('individual')
