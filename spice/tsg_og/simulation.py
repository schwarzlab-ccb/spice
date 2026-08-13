import os
from collections import namedtuple
from copy import copy, deepcopy
import pickle
from joblib import Parallel, delayed

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from spice.random_state import np_rng, seed_task, spawn_seeds
from spice.utils import get_logger, open_pickle, save_pickle, CALC_NEW
from spice.length_scales import DEFAULT_SEGMENT_SIZE_DICT
from spice import data_loaders, directories, config

logger = get_logger('simulation')

CHROM_LENS = data_loaders.load_chrom_lengths()
CENTROMERES = data_loaders.load_centromeres()
CENTROMERES_OBSERVED = data_loaders.load_centromeres(extended=False, observed=True)
TELOMERES_OBSERVED = data_loaders.load_telomeres_observed()

Locus = namedtuple('Locus', ['pos', 'fitness'])
Plateau = namedtuple('Plateau', ['start', 'end', 'fitness'])
PLATEAU_WIDTH = 10e5

def is_locus(selection):
    return isinstance(selection, Locus)

def is_plateau(selection):
    return isinstance(selection, Plateau)

def combine_selection_points(selection_points):
    return SelectionPoints(loci=sum([x.loci for x in selection_points], []),
                        plateaus=sum([x.plateaus for x in selection_points], []))

def copy_list_of_selection_points(selection_points_list):
    return pickle.loads(pickle.dumps(selection_points_list, -1));

class SelectionPoints:
    def __init__(self, loci=None, plateaus=None):
        self.loci = list([x if is_locus(x) else Locus(*x) for x in loci]) if loci is not None else []
        self.plateaus = list([x if is_plateau(x) else Plateau(*x) for x in plateaus]) if plateaus is not None else []

    def __len__(self):
        return len(self.loci) + len(self.plateaus)
    
    def __getitem__(self, i):
        if i < len(self.loci):
            return self.loci[i]
        return self.plateaus[i - len(self.loci)]
    
    def get_entries(self):
        return self.loci + self.plateaus

    def add_locus(self, locus):
        self.loci.append(locus if is_locus(locus) else Locus(*locus))
        return self

    def add_plateau(self, plateau):
        self.plateaus.append(plateau if is_plateau(plateau) else Plateau(*plateau))
        return self

    def replace_locus(self, i, locus):
        if 0 <= i < len(self.loci):
            self.loci[i] = locus
        else:
            raise IndexError("Locus index out of range")

    def replace_plateau(self, i, plateau):
        if 0 <= i < len(self.plateaus):
            self.plateaus[i] = plateau
        else:
            raise IndexError("Plateau index out of range")
        
    def remove_entry(self, i):
        if i < 0 or i >= len(self.loci) + len(self.plateaus):
            raise IndexError("Entry index out of range")
        if i < len(self.loci):
            del self.loci[i]
        else:
            del self.plateaus[i - len(self.loci)]
        return self

    def replace_entry(self, i, new_entry):
        """Replaces the i-th entry in the combined list of loci and plateaus."""
        if i < 0 or i >= len(self.loci) + len(self.plateaus):
            raise IndexError("Entry index out of range")

        if i < len(self.loci):
            if is_locus(new_entry):
                self.loci[i] = new_entry
            else:
                assert len(new_entry) == 2, 'Wrong size for locus entry'
                self.loci[i] = Locus(*new_entry)
        else:
            if is_plateau(new_entry):
                self.plateaus[i - len(self.loci)] = new_entry
            else:
                assert len(new_entry) == 3, "Wrong size for plateau entry"
                self.plateaus[i - len(self.loci)] = Plateau(*new_entry)
        return self

    def copy(self, deep=True):
        return deepcopy(self) if deep else copy(self)

    # def add(self, other, copy=False):
    #     try:
    #         new_selection_points = SelectionPoints(self.loci + other.loci, self.plateaus + other.plateaus)
    #     except AttributeError:
    #         if is_locus(other):
    #             new_selection_points = SelectionPoints(self.loci + [other], self.plateaus)
    #         elif is_plateau(other):
    #             new_selection_points = SelectionPoints(self.loci, self.plateaus + [other])
    #         else:
    #             raise ValueError(f'{type(other)} is not a valid entry type')
    #     if copy:
    #         new_selection_points = new_selection_points.copy()
    #     return new_selection_points

    def __add__(self, other):
        try:
            return SelectionPoints(self.loci + other.loci, self.plateaus + other.plateaus)
        except AttributeError:
            if is_locus(other):
                return SelectionPoints(self.loci + [other], self.plateaus)
            elif is_plateau(other):
                return SelectionPoints(self.loci, self.plateaus + [other])
            else:
                raise ValueError(f'{type(other)} is not a valid entry type')
    
    def __str__(self):
        loci_str = " | ".join(f"{p.pos}: {p.fitness}" for i, p in enumerate(self.loci))
        plateaus_str = " | ".join(f"{pl.start} - {pl.end}: {pl.fitness}" for i, pl in enumerate(self.plateaus))
        
        return f"loci: {loci_str if loci_str else ' None'} /// Plateaus: {plateaus_str if plateaus_str else ' None'}"

    def __repr__(self):
        return str(self)
    
    def __eq__(self, other) -> bool:
        return str(self) == str(other)
    
    def __iter__(self):
        return iter(self.loci + self.plateaus)
    
    def __deepcopy__(self, memodict={}):
        return SelectionPoints(loci=deepcopy(self.loci), plateaus=deepcopy(self.plateaus))

    def random_entry(self):
        combined = self.loci + self.plateaus
        assert len(combined) > 0, "No entries in the selection points"
        
        index = np_rng().randint(0, len(combined)) if len(combined) > 1 else 0
        entry = combined[index]

        return index, entry


def resimulate_events(cur_widths, selection_points=None, chrom_size=None, baseline_fitness=0, segment_size=100e3,
                    length_scale=None, seed=None, cur_chrom=None, n_random_values=100, remove_centromere_bound=True, scale=1,
                    normalize_from_signal=False, cur_signal=None):
    
    assert selection_points is not None or baseline_fitness > 0, 'Either genes with fitness values or a baseline fitness value > 0 is needed'
    if seed is not None:
        seed_task(seed)
    if chrom_size is None:
        chrom_size = CHROM_LENS.loc[cur_chrom]

    if cur_chrom is None:
        chrom_start = 0
        chrom_end = chrom_size
    else:
        chrom_start = TELOMERES_OBSERVED.loc[cur_chrom, length_scale]['chrom_start']
        chrom_end = TELOMERES_OBSERVED.loc[cur_chrom, length_scale]['chrom_end']

    assert np.max(cur_widths) < chrom_size

    if remove_centromere_bound:
        if length_scale is None:
            centromeres = CENTROMERES
        else:
            centromeres = CENTROMERES_OBSERVED[length_scale]
        assert cur_chrom is not None
        centro_start = centromeres.loc[cur_chrom, 'centro_start']
        centro_end = centromeres.loc[cur_chrom, 'centro_end']

        if cur_chrom in ['chr13', 'chr14', 'chr15', 'chr21', 'chr22']:
            chrom_start = centro_end

    chrom_size_real = chrom_end - chrom_start
    cur_starts = chrom_start + np_rng().uniform(0, 1, (len(cur_widths), n_random_values)) * (chrom_size_real - cur_widths)[:, None]
    cur_ends = cur_starts + cur_widths[:, None]

    event_fitness = np.ones_like(cur_starts) * baseline_fitness

    if selection_points is not None and len(selection_points.loci) > 0:
        cur_loci_array = np.array(selection_points.loci)
        overlap_with_loci = ((cur_starts[:, :, None] < cur_loci_array[:, 0]) & (cur_ends[:, :, None] > cur_loci_array[:, 0]))
        event_fitness += (overlap_with_loci * cur_loci_array[:, 1]).sum(axis=2)
        event_fitness = np.maximum(0, event_fitness)

    if remove_centromere_bound:
        is_centromere_bound = np.logical_or(
            np.logical_and(cur_starts > centro_start, cur_starts < centro_end),
            np.logical_and(cur_ends > centro_start, cur_ends < centro_end)
        )
        assert not is_centromere_bound.all(axis=1).any(), "for a given simulated event, all options were centromere-bound! Increase `n_random_values` to avoid this"

        # Mask the event_fitness array to exclude entries where is_centromere_bound is True
        event_fitness = np.where(is_centromere_bound, 0, event_fitness)

    event_fitness = event_fitness / (event_fitness.sum(axis=1, keepdims=True) + 1e-9)

    # Inverse transform sampling
    random_values = np_rng().rand(len(event_fitness))[:, None]
    cumulative_probs = np.cumsum(event_fitness, axis=1)
    selected_events_indices = (random_values < cumulative_probs).argmax(axis=1)

    selected_starts = cur_starts[np.arange(len(cur_starts)), selected_events_indices]
    selected_ends = cur_ends[np.arange(len(cur_ends)), selected_events_indices]
    
    # Overlap with bins
    overlap_bins = np.arange(int(chrom_size/segment_size)) * segment_size
    overlap_bins[-1] = chrom_size
    bin_starts = np.arange(0, chrom_size, segment_size)[:-1]
    bin_ends = np.append(bin_starts[1:], chrom_size)  # Ensure the last bin ends at chrom_size
    start_bins = np.searchsorted(bin_ends, selected_starts, side='right')
    end_bins = np.searchsorted(bin_starts, selected_ends, side='left')
    overlap_bin_counts = np.bincount(np.hstack([np.arange(s, e) for s, e in zip(start_bins, end_bins)]), minlength=len(bin_starts))
    overlap_bin_counts = overlap_bin_counts.astype(float)

    if selection_points is not None and len(selection_points.plateaus) > 0:
        overlap_bin_counts_sum = overlap_bin_counts.sum()
        for plateau in selection_points.plateaus:
            overlap_bin_counts[int(plateau.start//segment_size):int(plateau.end//segment_size)] += scale * plateau.fitness
            left, right = int((plateau.start-PLATEAU_WIDTH)//segment_size), int(plateau.start//segment_size)
            length = right - left
            if length > 0:
                overlap_bin_counts[left:right] += scale * np.arange(int(PLATEAU_WIDTH//segment_size) - length, PLATEAU_WIDTH//segment_size) / (PLATEAU_WIDTH//segment_size) * plateau.fitness
            left, right = int(plateau.end//segment_size), int((plateau.end+PLATEAU_WIDTH)//segment_size)
            length = right - left
            if length > 0:
                overlap_bin_counts[left:right] += scale * np.arange(int(PLATEAU_WIDTH//segment_size) - length, PLATEAU_WIDTH//segment_size)[::-1]/(PLATEAU_WIDTH//segment_size) * plateau.fitness
        overlap_bin_counts = overlap_bin_counts / overlap_bin_counts.sum() * overlap_bin_counts_sum
    
    if normalize_from_signal:
        assert cur_signal is not None, 'cur_signal is required if normalize_from_signal is True'
        overlap_bin_counts = overlap_bin_counts / overlap_bin_counts.sum() * cur_signal.sum()

    return overlap_bins, overlap_bin_counts


def resimulate_events_multiple(cur_chrom, data_per_length_scale, final_selection_points,
                            N_sims=250, segment_size_dict=DEFAULT_SEGMENT_SIZE_DICT, n_cores=4,
                            normalize_from_signal=True):

    if final_selection_points is None:
        final_selection_points = 8 * [[SelectionPoints()]]

    # One stream per sim, drawn from the caller's stream. Both halves matter: the workers are
    # separate processes whose RNG would otherwise start from OS entropy (unreproducible), and this
    # function is called many times per run (bootstrap, resim nulls), so the sims must still differ
    # between calls -- which a seed derived from `sim_index` alone would not do.
    sim_seeds = spawn_seeds(N_sims)

    def simulate_single_sim(sim_index):
        """Simulate a single iteration across all length scales."""
        seed_task(sim_seeds[sim_index])
        return [
            resimulate_events(
                data['cur_widths'], combine_selection_points(cur_sp), chrom_size=CHROM_LENS.loc[cur_chrom], baseline_fitness=1,
                remove_centromere_bound=True, length_scale=data['length_scale'], cur_chrom=cur_chrom,
                segment_size=segment_size_dict[data['length_scale']],
                normalize_from_signal=normalize_from_signal, cur_signal=data['signals'])[1]
                for data, cur_sp in zip(data_per_length_scale.values(), final_selection_points)]

    if n_cores is not None:
        simulated_resim = Parallel(n_jobs=n_cores)(
            delayed(simulate_single_sim)(sim_index) for sim_index in range(N_sims))
    else:
        simulated_resim = [
            simulate_single_sim(sim_index) for sim_index in range(N_sims)]

    simulated_resim =  [np.stack(x) for x in list(zip(*simulated_resim))]
    return simulated_resim


def create_convolution_kernel(cur_widths, segment_size=1e5, n_widths_for_kernel=100_000, seed=None, which='locus'):
    if which == 'locus':
        # This is required to match the kernel size to the other functions
        kernel_size = int(np.ceil(2*np.max(cur_widths)/segment_size)*segment_size) + 1
        central_locus = SelectionPoints(loci=[(kernel_size//2, 1)])

        _, kernel = resimulate_events(
            np.repeat(cur_widths, max(1, n_widths_for_kernel/len(cur_widths))), central_locus, chrom_size=kernel_size, baseline_fitness=0, seed=seed,
            remove_centromere_bound=False, segment_size=segment_size)
    elif which == "edge":
        kernel_size = 4*np.max(cur_widths)
        empty_locus = SelectionPoints()

        _, kernel = resimulate_events(
            np.repeat(cur_widths, max(1, n_widths_for_kernel/len(cur_widths))), empty_locus, chrom_size=kernel_size, baseline_fitness=1, seed=seed,
            remove_centromere_bound=False, segment_size=segment_size)
        kernel = kernel[:int(np.max(cur_widths)/segment_size)+1]
    else:
        raise ValueError(f'Invalid kernel type: {which}')

    return kernel / kernel.max()


def create_centromere_values(cur_chrom, length_scale, cur_widths, segment_size=100e3):
    centro_width = int(np.max(cur_widths)/segment_size)+1
    centromere_values = {
        'left_centromere_outer_bound': max(0, (int(CENTROMERES_OBSERVED[length_scale].loc[cur_chrom, 'centro_start']/segment_size) - centro_width)),
        'right_centromere_outer_bound': min(int(CHROM_LENS.loc[cur_chrom]//segment_size), (int(CENTROMERES_OBSERVED[length_scale].loc[cur_chrom, 'centro_end']/segment_size) + centro_width)),
        'left_centromere_bound': max(0, (int(CENTROMERES_OBSERVED[length_scale].loc[cur_chrom, 'centro_start']/segment_size))),
        'right_centromere_bound': min(int(CHROM_LENS.loc[cur_chrom]//segment_size), (int(CENTROMERES_OBSERVED[length_scale].loc[cur_chrom, 'centro_end']/segment_size))),
        'centro_width': CENTROMERES_OBSERVED[length_scale].loc[cur_chrom].diff().iloc[1],
    }
    return centromere_values


def create_height_multiplier(cur_widths, cur_chrom, cur_length_scale, cur_type, loci_width,
                            segment_size_dict=DEFAULT_SEGMENT_SIZE_DICT,
                            n_widths=100_000, n_sims=100, show_progress=False):
    assert cur_widths is not None and cur_chrom is not None and cur_length_scale is not None and cur_type is not None, f'All parameters are required, some are None: {cur_widths}, {cur_chrom}, {cur_length_scale}, {cur_type}'

    height_multiplier = []
    for _ in tqdm(range(n_sims), disable=not show_progress):
        _, cur_sim = resimulate_events(np.repeat(cur_widths, max(1, n_widths/len(cur_widths))), SelectionPoints(), baseline_fitness=1, cur_chrom=cur_chrom,
                                        segment_size=segment_size_dict[cur_length_scale], remove_centromere_bound=True, length_scale=cur_length_scale)
        height_multiplier.append(cur_sim)
    height_multiplier = np.stack(height_multiplier).mean(axis=0)

    # Reduce the ruggedness in the center of the arms
    boundary_tel_left = np.where(height_multiplier>0)[0][0] + 2*loci_width
    boundary_tel_right = np.where(height_multiplier>0)[0][-1] - 2*loci_width
    boundary_cen_left = int(CENTROMERES_OBSERVED.loc[cur_chrom].loc[cur_length_scale]['centro_start'] / segment_size_dict[cur_length_scale]) - 2*loci_width
    boundary_cen_right = int(CENTROMERES_OBSERVED.loc[cur_chrom].loc[cur_length_scale]['centro_end'] / segment_size_dict[cur_length_scale]) + 2*loci_width
    if boundary_tel_left < boundary_cen_left:
        height_multiplier[:boundary_tel_left] = (
            height_multiplier[:boundary_tel_left] / height_multiplier[boundary_tel_left-1])
            # height_multiplier[:boundary_tel_left] / height_multiplier[:boundary_tel_left].max())
        height_multiplier[boundary_tel_left:boundary_cen_left] = 1
        if not (boundary_tel_right > boundary_cen_right):
            height_multiplier[boundary_cen_left:] = (
                height_multiplier[boundary_cen_left:] / height_multiplier[boundary_cen_left])
                # height_multiplier[boundary_cen_left:] / height_multiplier[boundary_cen_left:].max())
    if boundary_tel_right > boundary_cen_right:
        height_multiplier[boundary_tel_right:] = (
            height_multiplier[boundary_tel_right:] / height_multiplier[boundary_tel_right])
            # height_multiplier[boundary_tel_right:] / height_multiplier[boundary_tel_right:].max())
        height_multiplier[boundary_cen_right:boundary_tel_right] = 1
        if not (boundary_tel_left < boundary_cen_left):
            height_multiplier[:boundary_cen_right] = (
                height_multiplier[:boundary_cen_right] / height_multiplier[boundary_cen_right-1])
                # height_multiplier[:boundary_cen_right] / height_multiplier[:boundary_cen_right].max())
    if (boundary_tel_left < boundary_cen_left) and (boundary_tel_right > boundary_cen_right):
        height_multiplier[boundary_cen_left:boundary_cen_right] = (
            height_multiplier[boundary_cen_left:boundary_cen_right] / height_multiplier[boundary_cen_left:boundary_cen_right].max()
        )
    if not (boundary_tel_left < boundary_cen_left) and not (boundary_tel_right > boundary_cen_right):
        height_multiplier = height_multiplier / height_multiplier.max()

    return height_multiplier


def convolution_simulation(cur_widths, selection_points=None, chrom_size=None, cur_type=None, cur_chrom=None,
                        kernel=None, kernel_edge=None, baseline_fitness=1, legacy_height_multiplier=False,
                        height_multiplier=None, segment_size=100e3, return_None_if_zero=False,
                        n_widths_for_kernel=100_000, correct_centromeres=True, cur_length_scale=None,
                        centromere_values=None, normalize_from_signal=True, cur_signal=None):

    # Kind of awkward but I want to allow running without specific chromosome for testing
    if cur_chrom is None:
        assert chrom_size is not None, 'chrom_size is required if cur_chrom is None'
        chrom_start = 0
        chrom_end = chrom_size
    else:
        chrom_size = CHROM_LENS[cur_chrom]
        if cur_length_scale is None:
            chrom_start = 0
            chrom_end = CHROM_LENS[cur_chrom]
        else:
            chrom_start = TELOMERES_OBSERVED.loc[cur_chrom, cur_length_scale]['chrom_start']
            chrom_end = TELOMERES_OBSERVED.loc[cur_chrom, cur_length_scale]['chrom_end']
    chrom_left_pad = int(chrom_start//segment_size)
    chrom_right_pad = int((chrom_size - chrom_end)//segment_size)

    if len(cur_widths) == 0:
        return None if return_None_if_zero else np.zeros(int(chrom_size//segment_size))

    if selection_points is None:
        selection_points = SelectionPoints()

    # This is for example used in the baseline plots
    if kernel is None:
        kernel = create_convolution_kernel(cur_widths, segment_size, n_widths_for_kernel, which='locus')
    else:
        assert len(kernel) == np.ceil(int(2*np.max(cur_widths))/segment_size), f'Kernel size {len(kernel)} does not match the expected size {np.ceil(int(2*np.max(cur_widths))/segment_size)}'

    final_length = int(chrom_size//segment_size)
    pad_width = int(np.max(cur_widths)//segment_size)
    point = np.ones(final_length+2*pad_width) * baseline_fitness / np.sum(kernel)
    for gene in selection_points.loci:
        point[int(gene[0]/segment_size)+pad_width] += gene[1]
    convolution = np.convolve(kernel, point, mode='full')

    # Adjust length so it matches the real data, required because of padding
    cur_length = len(convolution)
    convolution = convolution[int(np.floor((cur_length - final_length)/2)):-int(np.ceil((cur_length - final_length)/2))]

    if correct_centromeres and not cur_chrom in ['chr13', 'chr14', 'chr15', 'chr21', 'chr22']:
        if centromere_values is None:
            centromere_values = create_centromere_values(cur_chrom, cur_length_scale, cur_widths, segment_size)
        left_centromere_outer_bound = centromere_values['left_centromere_outer_bound']
        right_centromere_outer_bound = centromere_values['right_centromere_outer_bound']
        left_centromere_bound = centromere_values['left_centromere_bound']
        right_centromere_bound = centromere_values['right_centromere_bound']

        # Force a constant value at the centromere
        convolution[left_centromere_bound:right_centromere_bound] = np.mean(convolution[left_centromere_bound:right_centromere_bound])

    if height_multiplier is None:
        if legacy_height_multiplier:
            raise DeprecationWarning("legacy_height_multiplier is deprecated, please provide height_multiplier directly")
            if kernel_edge is None:
                raise ValueError("If legacy_height_multiplier is True, kernel_edge must be provided or created")
                # kernel_edge = create_convolution_kernel(cur_widths, segment_size, n_widths_for_kernel, which='edge')
            # pad the kernel to account for the fact that events are not observed close to telomere
            kernel_edge_left = np.append(np.zeros(chrom_left_pad), kernel_edge)
            kernel_edge_right = np.append(kernel_edge[::-1], np.zeros(chrom_right_pad))

            # Height multiplier for centromeres and telomeres
            height_multiplier = np.ones_like(convolution)

            # Correct for centromeres
            if correct_centromeres and not cur_chrom in ['chr13', 'chr14', 'chr15', 'chr21', 'chr22']:
                assert cur_chrom is not None and cur_length_scale is not None, f'The following cannot be None if correcting centroermes: {"cur_chrom" if cur_chrom is None else ""} {"cur_length_scale" if cur_length_scale is None else ""}'
                centro_width = centromere_values['centro_width']
                centromere_factor = np.mean((cur_widths > centro_width) * (1-(centro_width/cur_widths)))
                height_multiplier[left_centromere_bound:right_centromere_bound] = 0
                height_multiplier[right_centromere_bound:right_centromere_outer_bound] = kernel_edge[:(right_centromere_outer_bound-right_centromere_bound)]
                height_multiplier[left_centromere_outer_bound:left_centromere_bound] = kernel_edge[:(left_centromere_bound - left_centromere_outer_bound)][::-1]
                height_multiplier[left_centromere_outer_bound:right_centromere_outer_bound] *= 1-centromere_factor
                height_multiplier[left_centromere_outer_bound:right_centromere_outer_bound] += centromere_factor
            
            # Telomere adjustement
            height_multiplier = height_multiplier.copy()
            if cur_chrom in ['chr13', 'chr14', 'chr15', 'chr21', 'chr22']:
                if centromere_values is None:
                    centro_end = CENTROMERES.loc[cur_chrom, 'centro_end']
                else:
                    centro_end = centromere_values['right_centromere_bound']
                height_multiplier[:int(centro_end//segment_size)] = 0
                height_multiplier[int(centro_end//segment_size):int(centro_end//segment_size)+len(kernel_edge_left)] = kernel_edge_left
            else:
                height_multiplier[:len(kernel_edge_left)] *= kernel_edge_left
            height_multiplier[-len(kernel_edge_right):] *= kernel_edge_right

        else:
            raise ValueError("Need to provide height_multiplier if legacy_height_multiplier is set to False")
    else:
        assert not legacy_height_multiplier, "If height_multiplier is provided, legacy_height_multiplier must be False"
    convolution = convolution * height_multiplier

    # Adjust sum to the expected one
    if normalize_from_signal:
        assert cur_signal is not None, 'cur_signal is required if normalize_from_signal is True'
        convolution = convolution / convolution.sum() * cur_signal.sum()
    else:
        convolution = convolution / convolution.sum() * np.sum(cur_widths/segment_size + 1)

    # This is not used anymore
    # for plateau in selection_points.plateaus:
    #     convolution[int(plateau.start//segment_size):int(plateau.end//segment_size)] += plateau.fitness
    #     left, right = int((plateau.start-PLATEAU_WIDTH)//segment_size), int(plateau.start//segment_size)
    #     length = right - left
    #     if length > 0:
    #         convolution[left:right] += np.arange(int(PLATEAU_WIDTH//segment_size) - length, PLATEAU_WIDTH//segment_size) / (PLATEAU_WIDTH//segment_size) * plateau.fitness               
    #     left, right = int(plateau.end//segment_size), int((plateau.end+PLATEAU_WIDTH)//segment_size)
    #     length = right - left
    #     if length > 0:
    #         convolution[left:right] += np.arange(int(PLATEAU_WIDTH//segment_size) - length, PLATEAU_WIDTH//segment_size)[::-1]/(PLATEAU_WIDTH//segment_size) * plateau.fitness
    
    # convolution = convolution / convolution.sum() * np.sum(cur_widths/segment_size + 1)

    # shift by one
    convolution = np.append(convolution[1:], 0)

    if return_None_if_zero and np.sum(convolution) == 0:
        return None

    return convolution


def convolution_simulation_per_ls(cur_chrom, data_per_length_scale, cur_selection_points,
                                segment_size_dict=DEFAULT_SEGMENT_SIZE_DICT, normalize_from_signal=True,
                                legacy_height_multiplier=False):

    assert all([x['chrom']==cur_chrom for x in data_per_length_scale.values()]), f'Wrong data_per_length_scale for current chrom {cur_chrom}'

    if cur_selection_points is None:
        cur_selection_points = 8 * [[SelectionPoints()]]
    assert len(cur_selection_points) == 8, len(cur_selection_points)

    return [convolution_simulation(
                cur_chrom=cur_chrom, selection_points=combine_selection_points(cur_sp), cur_widths=data['cur_widths'],
                kernel=data['kernel'], chrom_size=None, kernel_edge=data.get('kernel_edge', None), cur_length_scale=data['length_scale'],
                segment_size=segment_size_dict[data['length_scale']], centromere_values=data['centromere_values'],
                normalize_from_signal=normalize_from_signal, cur_signal=data['signals'],
                legacy_height_multiplier=legacy_height_multiplier,
                height_multiplier=None if legacy_height_multiplier else data['height_multiplier'])
            for data, cur_sp in zip(data_per_length_scale.values(), cur_selection_points)]
