"""Observed boundaries must be measured after the same static event filters as loci."""
import os
import subprocess
import sys

import pandas as pd
import pytest

import spice
from spice import data_loaders


def raw_events():
    rows = []
    for i, (start, end) in enumerate([
        (80_000_000, 80_500_000), (160_000_000, 160_500_000),
        (117_000_000, 117_500_000),  # inside the extended centromere's 5 Mb padding
        (123_000_000, 123_500_000),  # centromere-bound after static reclassification
        (5_000_000, 50_000_000),    # above the maximum event width
        (90_000_000, 90_050_000),   # below the minimum event width
    ]):
        rows.append(dict(sample=f's{i}', id=f's{i}:chr1', chrom='chr1',
                         start=start, end=end, width=end-start, type='gain',
                         diff='010', telomere_bound=False, whole_chrom=False,
                         whole_arm=False, events_per_chrom=1, chrom_length=249250621,
                         pos='internal'))
    return pd.DataFrame(rows)


def read_tables(output):
    return tuple(pd.read_csv(output / f'{stem}_observed.tsv', sep='\t',
                             header=[0, 1], index_col=0)
                 for stem in ['centromeres', 'telomeres'])


def test_generation_without_existing_tables_in_fresh_process(tmp_path, repo_root_dir):
    """Also catches import-time dependencies on the tables being constructed."""
    source = tmp_path / 'events.tsv'
    raw_events().to_csv(source, sep='\t', index=False)
    code = '''
import sys
import pandas as pd
import spice
spice.config['input_files'] = {}
spice.directories['results_dir'] = sys.argv[2]
from spice.data_loaders import create_observed_centromeres_and_telomeres
create_observed_centromeres_and_telomeres(pd.read_csv(sys.argv[1], sep='\\t', dtype={'diff': str}))
assert 'spice.tsg_og.detection' not in sys.modules
'''
    env = dict(os.environ, PYTHONPATH=repo_root_dir)
    result = subprocess.run([sys.executable, '-c', code, str(source), str(tmp_path)],
                            cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    cen, tel = read_tables(tmp_path / 'data_loaders')
    assert list(cen.index) == ['chr1']
    assert tuple(cen.loc['chr1', 'small']) == (80_500_000, 160_000_000)
    assert tuple(tel.loc['chr1', 'small']) == (80_000_000, 160_500_000)


@pytest.mark.parametrize('remove_plateaus,expected_end', [(True, 170_500_000), (False, 175_500_000)])
def test_generation_honors_plateau_filter(tmp_path, monkeypatch, remove_plateaus, expected_end):
    events = raw_events()
    extra = events.iloc[[1]].copy()
    extra['id'] = 'plateau'; extra['sample'] = 'plateau'
    extra['start'] = 175_000_000; extra['end'] = 175_500_000
    # Keep an unmasked event on this arm; the masked event would extend the telomere bound.
    events.loc[1, ['start', 'end']] = [170_000_000, 170_500_000]
    events = pd.concat([events, extra], ignore_index=True)
    plateaus = tmp_path / 'plateaus.tsv'
    pd.DataFrame([dict(chrom='chr1', start=175_000_000, end=175_500_000)]).to_csv(
        plateaus, sep='\t', index=False)
    monkeypatch.setitem(spice.config, 'input_files', {'plateaus': str(plateaus)})
    monkeypatch.setitem(spice.config, 'loci_detection', {'remove_plateaus': remove_plateaus})
    monkeypatch.setitem(spice.directories, 'results_dir', str(tmp_path))
    data_loaders.create_observed_centromeres_and_telomeres(events)
    _, tel = read_tables(tmp_path / 'data_loaders')
    assert tel.loc['chr1', ('small', 'chrom_end')] == expected_end


def test_generation_handles_arm_without_filtered_events(tmp_path, monkeypatch):
    monkeypatch.setitem(spice.config, 'input_files', {})
    monkeypatch.setitem(spice.directories, 'results_dir', str(tmp_path))
    data_loaders.create_observed_centromeres_and_telomeres(raw_events().iloc[[0]])
    cen, tel = read_tables(tmp_path / 'data_loaders')
    assert not cen.isna().any().any()
    assert tuple(tel.loc['chr1', 'small']) == (80_000_000, 80_500_000)
