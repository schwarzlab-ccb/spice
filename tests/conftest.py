#!/usr/bin/env python
"""Pytest configuration and shared fixtures for SPICE tests."""

import os
import sys
import pytest

# Add the project root to the Python path
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, repo_root)

# ---------------------------------------------------------------------------------------------
# Point spice at the fixture observed tables, BEFORE any test module imports spice.tsg_og.
#
# spice/tsg_og/simulation.py loads centromeres_observed / telomeres_observed at MODULE IMPORT, and
# since 2026-09-07 those are cohort data with no packaged fallback (data_loaders._require_observed
# raises rather than borrowing another dataset's centromere definition). So every test that touched
# anything under spice.tsg_og died during COLLECTION with FileNotFoundError, whatever it was testing
# -- 14 of 25 in test_loci_cli.py + test_permutation_null.py, plus all of test_detection_empty_loci.
#
# conftest is imported before the test modules, so injecting here is enough for in-process imports.
# Subprocess CLI tests pass their own --config, which wins over anything set here, so those configs
# carry the same two keys themselves (see test_loci_cli.py's temp_workspace_with_loci).
#
# The fixture pair is spice's own pre-2026-09-07 packaged hg19 table -- see tests/objects/README.md.
# It is here to make imports succeed, not because its values mean anything for a real cohort.
import spice  # noqa: E402

TEST_OBSERVED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'objects')
TEST_OBSERVED_FILES = {
    'centromeres_observed': os.path.join(TEST_OBSERVED_DIR, 'centromeres_observed.tsv'),
    'telomeres_observed': os.path.join(TEST_OBSERVED_DIR, 'telomeres_observed.tsv'),
}
for _stem, _path in TEST_OBSERVED_FILES.items():
    assert os.path.exists(_path), f"missing test fixture {_path}"
spice.config.setdefault('input_files', {}).update(TEST_OBSERVED_FILES)


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )


@pytest.fixture(scope="session")
def repo_root_dir():
    """Return the repository root directory."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="session")
def example_data_exists(repo_root_dir):
    """Check if example data files exist."""
    example_data = os.path.join(repo_root_dir, 'data', 'example_data.tsv')
    return os.path.exists(example_data)


@pytest.fixture(scope="session")
def knn_train_data_exists(repo_root_dir):
    """Check if KNN training data exists."""
    knn_data = os.path.join(repo_root_dir, 'spice', 'objects', 'train_events_sv_and_unamb.pickle')
    return os.path.exists(knn_data)


@pytest.fixture(scope='session')
def prepared_segmentations(tmp_path_factory):
    """Self-contained test references; production grids are prepared externally."""
    import hashlib
    import json
    import numpy as np
    import pandas as pd
    from spice.data_loaders import load_chrom_lengths
    root=tmp_path_factory.mktemp('segmentation_reference')
    folder=root/'hg19';folder.mkdir()
    lengths=load_chrom_lengths()
    manifest=dict(schema=1,assembly='hg19',coordinate_system='0-based inclusive',
                  chromosome_lengths={c:int(n) for c,n in lengths.items()},files={})
    for size in [4000,20000,40000,100000,200000]:
        rows=[]
        for chrom,length in lengths.items():
            edges=np.append(np.arange(0,length,size)[:-1],length)
            rows.append(pd.DataFrame(dict(chrom=chrom,start=edges[:-1],end=edges[1:]-1)))
        frame=pd.concat(rows,ignore_index=True)
        path=folder/f'segmentation_{size}.tsv';frame.to_csv(path,sep='\t',index=False)
        manifest['files'][path.name]=dict(bin_size=size,bins=len(frame),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (folder/'manifest.json').write_text(json.dumps(manifest))
    return root
