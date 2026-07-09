#!/usr/bin/env python
"""Unit tests for empty-bucket handling in spice/main_loci_functions.py.

A "bucket" here is one (chromosome, length scale, direction) combination. SPICE
detects loci per bucket, then full_filter_by_p_values() combines/filters everything
by p-value. When a bucket is empty -- no candidate loci at all, or loci that all
fail the p < 0.05 cut -- the combine step used to crash instead of just reporting
zero loci for that bucket, which aborted the whole `spice loci_detection` run
before final_loci_detection.tsv was written.
"""

import numpy as np
import pandas as pd
import pytest

from spice import main_loci_functions as mlf


class _StubPoint:
    """Minimal stand-in for a real SelectionPoint: only .fitness/.pos are read
    on the empty-bucket code paths under test."""

    def __init__(self, fitness=0.0, pos=0):
        self.fitness = fitness
        self.pos = pos


def _make_selection_points(n_loci):
    # 8 length-scale/direction lists, each with one cluster per locus.
    return [[[_StubPoint()] for _ in range(n_loci)] for _ in range(8)]


class TestFullFilterByPValuesEmptyBuckets:
    def test_no_candidate_loci_returns_empty_result(self, monkeypatch, tmp_path):
        # Regression test: previously
        #   len(loci_df.query(...)) / len(loci_df)
        # raised ZeroDivisionError whenever a bucket had zero candidate loci
        # (e.g. no events of that type/size on that chromosome).
        empty_loci_df = pd.DataFrame(columns=['chrom', 'rank_on_chrom', 'p_value'])
        monkeypatch.setattr(mlf, 'assign_p_values', lambda loci_df, **kwargs: empty_loci_df)

        filtered_selection_points, filtered_loci_widths, final_p_values = mlf.full_filter_by_p_values(
            all_selection_points={},
            all_loci_widths={},
            all_data_per_length_scale={},
            output_dir=str(tmp_path),
            loci_df=empty_loci_df,
        )

        assert filtered_selection_points == {}
        assert filtered_loci_widths == {}
        assert len(final_p_values) == 0

    def test_no_loci_survive_p_value_cut_returns_empty_result(self, monkeypatch, tmp_path):
        # Regression test: when candidate loci exist but none pass p < threshold,
        # filtered_selection_points[chrom] ends up with empty per-length-scale
        # lists. The post-filter optimization loop then did
        #   max([y[0].fitness for y in x])
        # on an empty list, raising ValueError. Nothing should survive here, and
        # the function must return an empty result instead of raising.
        loci_df = pd.DataFrame({
            'chrom': ['chr1', 'chr1'],
            'rank_on_chrom': [0, 1],
            'p_value': [0.2, 0.8],  # both >= 0.05 threshold
        })
        monkeypatch.setattr(mlf, 'assign_p_values', lambda loci_df, **kwargs: loci_df)

        all_selection_points = {'chr1': _make_selection_points(2)}
        all_loci_widths = {'chr1': [1_000, 1_000]}

        filtered_selection_points, filtered_loci_widths, final_p_values = mlf.full_filter_by_p_values(
            all_selection_points=all_selection_points,
            all_loci_widths=all_loci_widths,
            all_data_per_length_scale={'chr1': {}},
            output_dir=str(tmp_path),
            loci_df=loci_df,
            p_value_threshold=0.05,
        )

        assert filtered_loci_widths == {'chr1': []}
        assert all(ls_x == [] for ls_x in filtered_selection_points['chr1'])
        assert len(final_p_values) == 0

    def test_one_bucket_empty_others_significant_does_not_raise(self, monkeypatch, tmp_path):
        # Regression test for the "some buckets empty, others fine" case: chr1 has
        # a significant locus (proceeds into the optimization loop), chr2 has none
        # (must be skipped rather than crashing the whole combine step).
        loci_df = pd.DataFrame({
            'chrom': ['chr1', 'chr2'],
            'rank_on_chrom': [0, 0],
            'p_value': [0.01, 0.9],
        })
        monkeypatch.setattr(mlf, 'assign_p_values', lambda loci_df, **kwargs: loci_df)
        # Also stub out the optimization internals so this stays a fast unit test
        # focused on the empty-bucket control flow, not the real optimizer.
        monkeypatch.setattr(mlf, 'convolution_simulation_per_ls', lambda *a, **k: 'conv')
        monkeypatch.setattr(mlf, 'calc_mse_loss', lambda *a, **k: 0.0)
        monkeypatch.setattr(
            mlf, '_optimize_selection_points',
            lambda *a, **k: (list(zip(*_make_selection_points(1))), None, None)
        )

        all_selection_points = {
            'chr1': _make_selection_points(1),
            'chr2': _make_selection_points(1),
        }
        all_loci_widths = {'chr1': [1_000], 'chr2': [1_000]}

        filtered_selection_points, filtered_loci_widths, final_p_values = mlf.full_filter_by_p_values(
            all_selection_points=all_selection_points,
            all_loci_widths=all_loci_widths,
            all_data_per_length_scale={'chr1': {}, 'chr2': {}},
            output_dir=str(tmp_path),
            loci_df=loci_df,
            p_value_threshold=0.05,
        )

        assert filtered_loci_widths['chr1'] == [1_000]
        assert filtered_loci_widths['chr2'] == []
        assert len(final_p_values) == 1


class TestBuildFinalLociDfEmptyInput:
    def test_returns_empty_dataframe_with_chrom_column(self):
        # Regression test: create_loci_df() raises ValueError('No loci found...')
        # on empty input, which used to propagate out of build_final_loci_df and
        # abort combine_loci() right after full_filter_by_p_values legitimately
        # reported zero significant loci.
        result = mlf.build_final_loci_df(
            all_selection_points={},
            all_loci_widths={},
            final_events_df=pd.DataFrame(),
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        # combine_loci() immediately does loci_df["chrom"].nunique() for logging.
        assert 'chrom' in result.columns
