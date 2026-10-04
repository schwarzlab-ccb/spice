"""Fixed local prominence searches retain neighbor and chromosome boundaries."""
import numpy as np
import pandas as pd
import pytest

from spice.tsg_og.loci import process_locus_prominence


def prominence(signal, positions, kernel_size, interval_width=2, segment_size=1):
    frame = pd.DataFrame({
        'pos': np.array(positions) * segment_size,
        'start': (np.array(positions) - interval_width / 2) * segment_size,
        'end': (np.array(positions) + interval_width / 2) * segment_size,
        'type': 'TSG',
    })
    result = process_locus_prominence(
        frame, signal, 'large', 'loss', kernel_size, 10, {'large': segment_size})
    return result['prominence_TSG_large'].to_numpy()


@pytest.mark.parametrize('segment_size', [1, 200_000])
def test_overlapping_candidates_do_not_claim_maximum_outside_local_windows(segment_size):
    x = np.arange(250)
    signal = 100 * np.exp(-0.5 * ((x - 120) / 20) ** 2)
    # Local windows [98,118) and [128,148) both exclude the maximum at bin 120.
    result = prominence(signal, [108, 138], 104, segment_size=segment_size)
    assert np.all(result == 0)


@pytest.mark.parametrize('center,peak', [(3, 8), (96, 91)])
def test_local_window_at_chromosome_edge_keeps_correct_argmax(center, peak):
    x = np.arange(100)
    signal = 50 * np.exp(-0.5 * ((x - peak) / 2) ** 2)
    assert prominence(signal, [center], 40)[0] > 49


def test_search_does_not_claim_peak_outside_kernel():
    x = np.arange(150)
    signal = 100 * np.exp(-0.5 * ((x - 90) / 5) ** 2)
    assert prominence(signal, [40], 40)[0] == 0


def test_neighbor_boundary_still_separates_distinct_peaks():
    x = np.arange(150)
    signal = (20 * np.exp(-0.5 * ((x - 40) / 2) ** 2) +
              100 * np.exp(-0.5 * ((x - 65) / 2) ** 2))
    values = prominence(signal, [40, 65], 104)
    assert values[0] == pytest.approx(20, abs=.01)
    assert values[1] == pytest.approx(100, abs=.01)


def test_wide_position_interval_retains_existing_search_bounds():
    x = np.arange(150)
    signal = 100 * np.exp(-0.5 * ((x - 80) / 5) ** 2)
    assert prominence(signal, [50], 104, interval_width=40)[0] == 0
