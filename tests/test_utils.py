#!/usr/bin/env python
"""Unit tests for spice/utils.py, targeting bugs found in code review."""

import logging
import os

import pytest

import spice
from spice.utils import (
    get_sister_allele,
    save_pickle,
    open_pickle,
    add_filename_suffix,
    CALC_NEW,
)
from spice.cli_functions import save_fail_reports


class TestGetSisterAllele:
    def test_swaps_cn_a_to_cn_b(self):
        assert get_sister_allele('sample1:chr1:cn_a') == 'sample1:chr1:cn_b'

    def test_swaps_cn_b_to_cn_a(self):
        assert get_sister_allele('sample1:chr1:cn_b') == 'sample1:chr1:cn_a'

    def test_preserves_trailing_chain_number(self):
        assert get_sister_allele('sample1:chr1:cn_a:3') == 'sample1:chr1:cn_b:3'
        assert get_sister_allele('sample1:chr1:cn_b:12') == 'sample1:chr1:cn_a:12'

    def test_does_not_corrupt_sample_id_containing_cn_a_substring(self):
        # A sample_id that happens to contain the literal substring "cn_a" must
        # not have that substring swapped -- only the trailing allele token.
        assert get_sister_allele('cn_a_patient1:chr1:cn_a') == 'cn_a_patient1:chr1:cn_b'

    def test_raises_for_malformed_id(self):
        with pytest.raises(ValueError):
            get_sister_allele('sample1:chr1:total_cn')

    def test_raises_for_id_with_no_allele_suffix(self):
        with pytest.raises(ValueError):
            get_sister_allele('sample1:chr1')


class TestSavePickle:
    def test_bare_filename_with_no_directory_component(self, tmp_path, monkeypatch):
        # Regression test: previously os.path.dirname('foo.pickle') == '' and
        # os.makedirs('', exist_ok=True) raised FileNotFoundError.
        monkeypatch.chdir(tmp_path)
        save_pickle({'a': 1}, 'bare_filename.pickle')
        assert os.path.exists('bare_filename.pickle')
        assert open_pickle('bare_filename.pickle') == {'a': 1}

    def test_round_trip_with_directory(self, tmp_path):
        filename = os.path.join(str(tmp_path), 'nested', 'dir', 'data.pickle')
        save_pickle([1, 2, 3], filename)
        assert open_pickle(filename) == [1, 2, 3]


class TestAddFilenameSuffix:
    def test_tsv_extension(self):
        assert add_filename_suffix('a/b.tsv', '_split') == 'a/b_split.tsv'

    def test_non_tsv_extension_preserved(self):
        assert add_filename_suffix('a/b.csv', '_split') == 'a/b_split.csv'
        assert add_filename_suffix('a/b.txt', '_split') == 'a/b_split.txt'

    def test_no_extension_falls_back_to_tsv(self):
        assert add_filename_suffix('a/b', '_split') == 'a/b_split.tsv'

    def test_distinct_from_input_regardless_of_extension(self):
        # This is the actual bug: a naive '.tsv' replace was a no-op (and thus
        # produced a path identical to the input) whenever the input path
        # didn't literally contain '.tsv'.
        for path in ['data.csv', 'data.txt', 'data', 'data.tsv']:
            assert add_filename_suffix(path, '_split') != path


class TestCalcNewLogger:
    def test_logger_name_participates_in_shared_spice_config(self):
        calc_new = CALC_NEW(verbose=True)
        # configure_logging() only touches loggers whose name contains 'spice'
        # (see spice/logging.py) -- previously this logger was a bare
        # logging.getLogger('CALC_NEW') and was invisible to --log/--debug.
        assert 'spice' in calc_new.logger.name.lower()

    def test_verbose_sets_debug_level(self):
        # log_debug() only emits when logger.level == logging.DEBUG exactly,
        # so verbose=True must result in DEBUG, not INFO.
        calc_new = CALC_NEW(verbose=True)
        assert calc_new.logger.level == logging.DEBUG

    def test_non_verbose_sets_warning_level(self):
        calc_new = CALC_NEW(verbose=False)
        assert calc_new.logger.level == logging.WARNING

    def test_calc_new_verbose_kwarg_sets_debug_level(self):
        calc_new = CALC_NEW(filename=None, verbose=False)

        @calc_new
        def f(x):
            return x * 2

        f(3, calc_new_verbose=True)
        assert calc_new.logger.level == logging.DEBUG


class TestSaveFailReports:
    def test_explicit_results_dir_does_not_raise_unbound_local_error(self, tmp_path, monkeypatch):
        # Regression test: `from spice import directories, config` used to be
        # nested inside `if results_dir is None`, making `config` a
        # function-local name that was unbound whenever a caller passed an
        # explicit results_dir.
        monkeypatch.setitem(spice.config, 'name', 'test_project_for_save_fail_reports')
        events_dir = os.path.join(str(tmp_path), 'test_project_for_save_fail_reports', 'events')
        os.makedirs(events_dir, exist_ok=True)

        result = save_fail_reports([], results_dir=str(tmp_path))
        assert result is not None
        assert os.path.exists(os.path.join(events_dir, 'failed_reports.tsv'))
