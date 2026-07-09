#!/usr/bin/env python
"""Unit tests for spice/data_loaders.py, targeting bugs found in code review."""

import os

import pytest

import spice
from spice.data_loaders import load_sv_data, resolve_copynumber_file


class TestLoadSvDataChromNormalization:
    def _write_sv_file(self, tmp_path, rows):
        sv_file = tmp_path / 'sv_data.tsv'
        header = "sample_id\tchrom\tsvclass\tstart\tend\n"
        body = "".join(f"{s}\t{c}\t{cls}\t{start}\t{end}\n" for s, c, cls, start, end in rows)
        sv_file.write_text(header + body)
        return str(sv_file)

    def test_non_chr_prefixed_chromosomes_are_normalized(self, tmp_path):
        sv_data_file = self._write_sv_file(tmp_path, [
            ('sample1', '1', 'DUP', 1000, 2000),
            ('sample1', '2', 'DEL', 3000, 4000),
        ])
        sv_data = load_sv_data(sv_data_file)
        assert set(sv_data['chrom_id']) == {'sample1:chr1', 'sample1:chr2'}

    def test_chrom_id_matches_copynumber_convention_for_filtering(self, tmp_path):
        # This is the concrete failure scenario from the bug report: an SV file
        # using bare chromosome numbers ("1") must still be matched by a
        # chrom_id built from the "chr1"-style convention used elsewhere in the
        # pipeline (see split_input.py's _process_group), or SV constraints for
        # that chromosome would be silently dropped.
        sv_data_file = self._write_sv_file(tmp_path, [
            ('sample1', '1', 'DUP', 1000, 2000),
        ])
        sv_data = load_sv_data(sv_data_file, chrom_id='sample1:chr1')
        assert len(sv_data) == 1

    def test_already_chr_prefixed_chromosomes_still_work(self, tmp_path):
        sv_data_file = self._write_sv_file(tmp_path, [
            ('sample1', 'chr1', 'DUP', 1000, 2000),
        ])
        sv_data = load_sv_data(sv_data_file, chrom_id='sample1:chr1')
        assert len(sv_data) == 1


class TestResolveCopynumberFileSplitPath:
    def test_non_tsv_input_produces_distinct_split_path(self, tmp_path, monkeypatch):
        # Regression test: a naive '.tsv' substring replace was a no-op for
        # any input extension other than '.tsv', making the "split" output
        # path identical to the raw input path -- which meant split_tsv_file
        # would silently overwrite the user's raw input file.
        data_dir = tmp_path / 'data'
        data_dir.mkdir()
        monkeypatch.setitem(spice.config, 'name', 'test_project_resolve')
        monkeypatch.setitem(spice.config, 'input_files', {'copynumber': 'raw_input.csv'})
        monkeypatch.setitem(spice.config, 'directories', {'data_dir': str(data_dir)})
        monkeypatch.setitem(spice.directories, 'base_dir', str(tmp_path))
        monkeypatch.setitem(spice.directories, 'data_dir', str(data_dir))

        raw_path = resolve_copynumber_file(return_raw=True)
        split_path = resolve_copynumber_file(return_raw=False)
        assert raw_path != split_path
        assert split_path.endswith('_split.csv')

    def test_tsv_input_still_produces_correct_split_path(self, tmp_path, monkeypatch):
        data_dir = tmp_path / 'data'
        data_dir.mkdir()
        monkeypatch.setitem(spice.config, 'name', 'test_project_resolve_tsv')
        monkeypatch.setitem(spice.config, 'input_files', {'copynumber': 'raw_input.tsv'})
        monkeypatch.setitem(spice.config, 'directories', {'data_dir': str(data_dir)})
        monkeypatch.setitem(spice.directories, 'base_dir', str(tmp_path))
        monkeypatch.setitem(spice.directories, 'data_dir', str(data_dir))

        split_path = resolve_copynumber_file(return_raw=False)
        assert split_path.endswith('_split.tsv')
        assert 'raw_input_split.tsv' in split_path
