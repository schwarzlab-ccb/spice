#!/usr/bin/env python
"""Unit tests for spice/preprocessing/preprocessing.py, targeting bugs found in code review."""

import pandas as pd

import spice
from spice.preprocessing.preprocessing import get_or_infer_xy_status


class TestGetOrInferXyStatus:
    def test_xy_status_file_is_loaded_not_silently_ignored(self, tmp_path, monkeypatch):
        # Regression test: get_or_infer_xy_status previously read
        # config['input_files']['xy_samples'], but every shipped config (and
        # the README) uses the key 'xy_status' -- so a user-provided
        # sex-status file was always silently ignored in favor of re-inferring
        # XY status from chrY presence.
        xy_file = tmp_path / 'xy_status.tsv'
        xy_file.write_text("sample_id\txy\nsample1\tTrue\nsample2\tFalse\n")

        monkeypatch.setitem(spice.config, 'input_files', {'xy_status': str(xy_file)})

        # Neither sample has any chrY segments, so if the file were still
        # (silently) ignored, inference would mark both samples as XX (False),
        # masking the bug -- sample1 would incorrectly come back False.
        data = pd.DataFrame({
            'sample_id': ['sample1', 'sample2'],
            'chrom': ['chr1', 'chr1'],
            'cn_a': [1, 1],
            'cn_b': [1, 1],
        })

        xy_status = get_or_infer_xy_status(data)
        assert bool(xy_status.loc['sample1']) is True
        assert bool(xy_status.loc['sample2']) is False

    def test_falls_back_to_inference_when_no_file_configured(self, monkeypatch):
        monkeypatch.setitem(spice.config, 'input_files', {})

        data = pd.DataFrame({
            'sample_id': ['sample1', 'sample2'],
            'chrom': ['chr1', 'chrY'],
            'cn_a': [1, 1],
            'cn_b': [1, 0],
        })

        xy_status = get_or_infer_xy_status(data)
        assert bool(xy_status.loc['sample1']) is False
        assert bool(xy_status.loc['sample2']) is True
