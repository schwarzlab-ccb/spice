"""Keep final tables, plots and CI inputs on the same fit after filtering."""
from argparse import Namespace
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

import spice
from spice import cli, main_loci_functions as main
from spice.length_scales import LENGTH_SCALE_NAMES
from spice.utils import save_pickle

FITNESS = [f'fitness_{ls}_{d}' for ls in LENGTH_SCALE_NAMES for d in ['gain', 'loss']]


@pytest.fixture
def combined_run(tmp_path, monkeypatch):
    root = tmp_path / 'loci'
    rows = []
    for chrom, count in [('chr1', 2), ('chr2', 1)]:
        points = [[(SimpleNamespace(fitness=float(i + 1)),) for i in range(count)] for _ in FITNESS]
        save_pickle(points, str(root / 'detection' / chrom / 'final_selection_points.pickle'))
        save_pickle([(i, i + 1) for i in range(count)], str(root / 'detection' / chrom / 'final_loci_widths.pickle'))
        save_pickle({}, str(root / 'data_per_length_scale' / f'{chrom}.pickle'))
        for i in range(count):
            row = dict(chrom=chrom, rank_on_chrom=i, pos=100 * (i + 1), type='OG')
            row.update({col: float(i + 1) for col in FITNESS})
            row.update(p_value_raw=0.01, p_value=0.01 if i == 0 else 0.9)
            for ls in LENGTH_SCALE_NAMES:
                row[f'p_value_raw_{ls}'] = row['p_value_raw']
                row[f'p_value_{ls}'] = row['p_value']
            rows.append(row)
    frame = pd.DataFrame(rows)
    monkeypatch.setattr(main, 'build_final_loci_df', lambda **kwargs: frame.copy(deep=True))
    monkeypatch.setattr(main, 'assign_p_values', lambda loci, null, **kwargs: loci.copy(deep=True))

    def refit(**kwargs):
        points = deepcopy(kwargs['final_selection_points'])
        for track in points:
            for locus in track:
                locus[0].fitness += 10
        return points, None

    optimizer = Mock(side_effect=refit)
    monkeypatch.setattr(main, 'final_optimization_step', optimizer)
    return root, optimizer


@pytest.mark.parametrize('cutoff', [1.01, 0.95])
def test_keep_all_never_refits(combined_run, cutoff):
    root, optimizer = combined_run
    final, points, widths, original = main.combine_loci(
        str(root), calculate_p_value=True, p_value_threshold=cutoff,
        permutation_null=pd.DataFrame({'stat': [1.0]}))
    optimizer.assert_not_called()
    pd.testing.assert_frame_equal(final, original)
    assert [p[0].fitness for p in points['chr1'][0]] == [1.0, 2.0]
    assert len(widths['chr1']) == 2


@pytest.mark.parametrize('cutoff,floor', [(0.05, None), (1.01, 1.5)])
def test_refits_only_chromosomes_with_removed_peaks(combined_run, cutoff, floor):
    root, optimizer = combined_run
    final, points, widths, original = main.combine_loci(
        str(root), calculate_p_value=True, p_value_threshold=cutoff,
        mean_fitness_threshold=floor, permutation_null=pd.DataFrame({'stat': [1.0]}))
    assert [call.kwargs['cur_chrom'] for call in optimizer.call_args_list] == ['chr1']
    before = 1.0 if floor is None else 2.0
    assert points['chr1'][0][0][0].fitness == before + 10
    assert len(widths['chr1']) == 1
    row = final[final.chrom == 'chr1'].iloc[0]
    assert all(row[col] == before + 10 for col in FITNESS)
    assert original[FITNESS[0]].tolist() == [1.0, 2.0, 1.0]
    assert row.q_value == (0.01 if floor is None else 0.9)
    assert len(points['chr2'][0]) == (1 if floor is None else 0)
    if floor is None:
        assert points['chr2'][0][0][0].fitness == 1.0


def test_removing_every_peak_never_calls_optimizer(combined_run):
    root, optimizer = combined_run
    final, points, widths, _ = main.combine_loci(
        str(root), calculate_p_value=True, p_value_threshold=0,
        permutation_null=pd.DataFrame({'stat': [1.0]}))
    optimizer.assert_not_called()
    assert final.empty
    assert all(not track for tracks in points.values() for track in tracks)
    assert all(not values for values in widths.values())


@pytest.mark.parametrize('single', [False, True])
@pytest.mark.parametrize('combined', [False, True])
def test_plot_curve_and_markers_use_final_fit(tmp_path, monkeypatch, single, combined):
    from spice import plot
    from spice.tsg_og import detection
    from matplotlib import pyplot as plt

    cfg = deepcopy(spice.config)
    cfg['name'] = 'plot_test'
    cfg['directories'].update(results_dir=str(tmp_path / 'results'),
                              log_dir=str(tmp_path / 'logs'), plot_dir=str(tmp_path / 'plots'))
    monkeypatch.setattr(spice, 'config', cfg)
    monkeypatch.setattr(spice, 'load_config', lambda _: None)
    monkeypatch.setattr(cli, '_apply_seed', lambda *args: None)
    root = tmp_path / 'results' / cfg['name'] / 'loci_of_selection'
    save_pickle('data', str(root / 'data_per_length_scale/chr1.pickle'))
    save_pickle('original fit', str(root / 'detection/chr1/final_selection_points.pickle'))
    if combined:
        save_pickle({'chr1': 'refitted survivors'}, str(root / 'detection/final_loci_detection_filtered.pickle'))
    # The survivor's original rank is 4, but its position in the compact fit is 0.
    pd.DataFrame([dict(chrom='chr1', rank_on_chrom=4)], index=[12]).to_csv(
        root.parent / 'final_loci_detection.tsv', sep='\t')
    convolution = Mock(return_value='curve')
    render = Mock()
    monkeypatch.setattr(detection, 'convolution_simulation_per_ls', convolution)
    monkeypatch.setattr(plot, 'plot_tsg_og_results', render)
    monkeypatch.setattr(plt, 'subplots', lambda **kwargs: (Mock(), None))
    args = Namespace(config_path='fixture', debug=False, log='terminal', seed=9,
                     plot_events_per_sample=None, plot_events_per_id=None,
                     plot_loci_on_chrom=None if single else 'chr1',
                     plot_single_locus=12 if single else None, loci_mode='detection')
    cli.main_plotting(args)
    expected = 'refitted survivors' if combined else 'original fit'
    convolution.assert_called_once_with('chr1', 'data', expected)
    assert render.call_args.kwargs['final_selection_points'] == expected
    assert render.call_args.kwargs['simulated_conv'] == 'curve'
    if single:
        assert render.call_args.kwargs['cluster_i'] == (0 if combined else 4)


def test_empty_combined_fit_does_not_fall_back_to_removed_peaks(tmp_path):
    save_pickle({'chr1': [[] for _ in FITNESS]}, str(tmp_path / 'detection/final_loci_detection_filtered.pickle'))
    points, combined = cli._load_plot_selection_points(str(tmp_path), 'detection', 'chr1')
    assert combined and points == [[] for _ in FITNESS]
    with pytest.raises(ValueError, match='recombine'):
        cli._load_plot_selection_points(str(tmp_path), 'detection', 'chr2')
