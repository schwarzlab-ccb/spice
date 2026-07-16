#!/usr/bin/env python
"""Smoke tests that the installed environment itself is sound.

These guard against dependency-resolution regressions such as the one fixed
alongside this file: relaxing the `ortools` pin in setup.py let pip resolve a
release that requires numpy>=2, which silently upgraded numpy over the
conda-forge numpy<2 build that pandas/medicc2 were compiled against. Every
import then failed with:
    ValueError: numpy.dtype size changed, may indicate binary incompatibility

Each check below shells out to a fresh interpreter so it reflects what a user
actually gets when they run `spice`, rather than relying on imports already
cached by other test modules in this session.
"""

import os
import subprocess
import sys


def _run(code):
    return subprocess.run(
        [sys.executable, '-c', code],
        capture_output=True,
        text=True,
    )


def test_numpy_pandas_scipy_import_without_abi_error():
    """pandas/scipy must load against the numpy ABI they were built for."""
    result = _run('import numpy, scipy, pandas')
    assert result.returncode == 0, result.stderr


def test_numpy_major_version_is_pinned_below_2():
    """medicc2's compiled extensions and the conda-forge pandas build require numpy<2.

    See the header comment in environment.yml and the `ortools<9.15` pin in
    setup.py -- ortools>=9.15 requires numpy>=2 and will silently break this.
    """
    import numpy
    major = int(numpy.__version__.split('.')[0])
    assert major < 2, (
        f"numpy {numpy.__version__} is installed; medicc2/pandas in this env "
        "require numpy<2. Check for an unpinned dependency (e.g. ortools) "
        "that resolved to a release requiring numpy>=2."
    )


def test_fst_engine_imports():
    """fstlib/medicc ship inside the medicc2 bioconda package and are required
    by spice.event_inference; a numpy ABI break or missing medicc2 install
    breaks these imports."""
    result = _run('import fstlib, medicc')
    assert result.returncode == 0, result.stderr


def test_ortools_importable_alongside_pinned_numpy():
    """ortools must not require a numpy newer than what's installed."""
    result = _run('import numpy, ortools')
    assert result.returncode == 0, result.stderr


def test_spice_package_imports_cleanly():
    """Reproduces the exact import chain behind the `spice` console-script
    entry point (spice.cli:main -> spice.utils -> pandas)."""
    result = _run('from spice.cli import main; from spice.utils import save_pickle')
    assert result.returncode == 0, result.stderr


def test_spice_cli_entry_point_runs():
    """The installed `spice` console script must actually start up end-to-end.

    Resolved next to sys.executable rather than via bare `spice` on PATH: a
    machine can have several conda envs each with their own `spice` install,
    and PATH order would silently test the wrong one instead of the env
    actually under test.
    """
    spice_bin = os.path.join(os.path.dirname(sys.executable), 'spice')
    result = subprocess.run(
        [spice_bin, '--help'],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert 'SPICE' in result.stdout
