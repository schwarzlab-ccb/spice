"""
Use this script to perform bootstrap sampling of signals for a given chromosome.
This involves resampling events with replacement and recalculating signals
based on the resampled events. The results are saved as pickle files for further analysis.
"""

import argparse
import itertools
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from spice import config, directories
from spice.length_scales import DEFAULT_SEGMENT_SIZE_DICT, DEFAULT_LENGTH_SCALE_BOUNDARIES
from spice.logging import log_debug
from spice.random_state import derive_seed, np_rng, seed_task
from spice.segmentation import create_events_in_segmentation
from spice.utils import get_logger, open_pickle, CALC_NEW

logger = get_logger(__name__)

DATA_LOADERS_DIR = os.path.join(directories['results_dir'], 'data_loaders')
CHROMS_ = ['chr' + str(x) for x in range(1, 23)] + ['chrX']


@CALC_NEW()
def get_signal_bootstrap_bounds(cur_chrom, data_loaders_dir, N_bootstrap=1_000):
    signal_bootstrap = open_pickle(os.path.join(data_loaders_dir, 'signal_bootstrap', f'{cur_chrom}_N_{N_bootstrap}.pickle'))
    signal_bootstrap_bounds = [(
        np.quantile(cur_signal_bootstrap, 0.025, axis=0),
        np.quantile(cur_signal_bootstrap, 0.975, axis=0))
        for cur_signal_bootstrap in signal_bootstrap]
    return signal_bootstrap_bounds


@CALC_NEW()
def bootstrap_sampling_of_signal(
        cur_chrom,
        final_events_df,
        segment_size_dict=DEFAULT_SEGMENT_SIZE_DICT,
        length_scale_boundaries=DEFAULT_LENGTH_SCALE_BOUNDARIES,
        N_bootstrap=1_000,
        filter_plateaus=True, disable_tqdm=True):
    """
    Perform bootstrap sampling for a single chromosome and return the results.

    Args:
        cur_chrom (str): Chromosome to process.
        final_events_df (pd.DataFrame): DataFrame containing event data.
        segment_size_dict (dict): Dictionary containing segment size information.
        N_bootstrap (int): Number of bootstrap iterations.

    Returns:
        list: List of bootstrap signals for the specified chromosome.
    """

    if filter_plateaus:
        final_events_df = final_events_df.query('plateau == "neither_left_nor_right"').copy().reset_index(drop=True)

    # Seed off the same identity the CALC_NEW cache is keyed on (chromosome + N_bootstrap), so the
    # bootstrap signals are the same whether they came from cache or were just recomputed, and are
    # unaffected by whatever else drew from this thread's stream first.
    seed_task(derive_seed('signal_bootstrap', cur_chrom, N_bootstrap))

    all_bootstrap_signals = []
    for iteration in tqdm(range(N_bootstrap), disable=disable_tqdm):
        log_debug(logger, f"Bootstrap iteration {iteration + 1}/{N_bootstrap} for chromosome {cur_chrom}")
        cur_bootstrap_signals = []
        for cur_length_scale, cur_type in itertools.product(['small', 'mid1', 'mid2', 'large'], ['gain', 'loss']):
            cur_length_scale_border = length_scale_boundaries[cur_length_scale]
            cur_events = final_events_df.query('pos == "internal" and type == @cur_type and chrom == @cur_chrom and width > @cur_length_scale_border[0] and width <= @cur_length_scale_border[1]').reset_index().copy()

            # Actual bootstrap sampling with replacement
            cur_events = cur_events.loc[np_rng().choice(np.arange(len(cur_events)), replace=True, size=len(cur_events))].reset_index().copy()

            signals_bootstrap = (create_events_in_segmentation(
                cur_events, bin_df=segment_size_dict[cur_length_scale], skip_tqdm=True)
                .loc[cur_chrom].sum(axis=1).values)
            cur_bootstrap_signals.append(signals_bootstrap)
        all_bootstrap_signals.append(cur_bootstrap_signals)

    all_bootstrap_signals = list(zip(*all_bootstrap_signals))
    all_bootstrap_signals = [np.stack(x) for x in all_bootstrap_signals]

    return all_bootstrap_signals
