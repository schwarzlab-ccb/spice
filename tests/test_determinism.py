#!/usr/bin/env python
"""Tests for spice.random_state: one seed has to pin the whole run down.

These cover the seeding contract itself (same seed -> same draws, per-thread streams, identity-keyed
task seeds surviving processes) rather than any one pipeline step, because that contract is what the
call sites in detection / p_values / event inference rely on.
"""

import os
import subprocess
import sys
import threading

import numpy as np
import pytest

from spice.random_state import (
    DEFAULT_SEED, SEED_ENV_VAR, derive_seed, get_seed, np_rng, py_rng, seed_task, set_seed,
    spawn_seeds)


def _draw(n=5):
    return list(np_rng().uniform(size=n)) + [py_rng().random() for _ in range(n)]


class TestSetSeed:
    def test_same_seed_same_draws(self):
        set_seed(1234)
        first = _draw()
        set_seed(1234)
        assert _draw() == first

    def test_different_seed_different_draws(self):
        set_seed(1234)
        first = _draw()
        set_seed(1235)
        assert _draw() != first

    def test_seeds_the_numpy_and_random_globals_too(self):
        # Un-migrated call sites and third-party code still draw from the globals.
        set_seed(7)
        first = (np.random.uniform(), __import__('random').random())
        set_seed(7)
        assert (np.random.uniform(), __import__('random').random()) == first

    def test_falls_back_to_env_then_default(self, monkeypatch):
        monkeypatch.setenv(SEED_ENV_VAR, '99')
        assert set_seed(None) == 99
        monkeypatch.delenv(SEED_ENV_VAR, raising=False)
        assert set_seed(None) == DEFAULT_SEED

    def test_exports_seed_for_worker_processes(self):
        set_seed(4321)
        assert os.environ[SEED_ENV_VAR] == '4321'

    def test_get_seed_reports_the_base_seed(self):
        set_seed(11)
        assert get_seed() == 11


class TestDeriveSeed:
    def test_stable_for_the_same_key(self):
        set_seed(1)
        assert derive_seed('step', 'chr3', 7) == derive_seed('step', 'chr3', 7)

    def test_distinct_keys_give_distinct_seeds(self):
        set_seed(1)
        seeds = {derive_seed('step', f'chr{i}') for i in range(1, 23)}
        assert len(seeds) == 22

    def test_tracks_the_base_seed(self):
        set_seed(1)
        one = derive_seed('step', 'chr3')
        set_seed(2)
        assert derive_seed('step', 'chr3') != one

    def test_not_affected_by_hash_randomisation(self):
        # str.__hash__ is salted per process, so derive_seed must not be built on hash(). Two
        # interpreters with different PYTHONHASHSEED must agree, or results would differ per run.
        code = (
            'from spice.random_state import set_seed, derive_seed;'
            'set_seed(5); print(derive_seed("all_solutions", "sample_1", "chr17"))'
        )
        outs = []
        for hash_seed in ('0', '1'):
            env = dict(os.environ, PYTHONHASHSEED=hash_seed)
            env.pop(SEED_ENV_VAR, None)
            outs.append(subprocess.run([sys.executable, '-c', code], env=env, check=True,
                                       capture_output=True, text=True).stdout.strip())
        assert outs[0] == outs[1]


class TestSpawnSeeds:
    def test_reproducible_under_the_same_base_seed(self):
        set_seed(3)
        first = spawn_seeds(8)
        set_seed(3)
        assert spawn_seeds(8) == first

    def test_successive_calls_differ(self):
        # Repeated calls (e.g. one resimulate_events_multiple per bootstrap iteration) must keep
        # drawing fresh randomness rather than repeating the same children.
        set_seed(3)
        assert spawn_seeds(8) != spawn_seeds(8)

    def test_children_are_distinct(self):
        set_seed(3)
        assert len(set(spawn_seeds(64))) == 64


class TestThreadIsolation:
    def test_each_thread_has_its_own_stream(self):
        set_seed(10)
        results = {}
        barrier = threading.Barrier(2)

        def work(name):
            seed_task(derive_seed('task', name))
            barrier.wait()          # force the two threads to interleave their draws
            values = _draw(20)
            barrier.wait()
            results[name] = values

        threads = [threading.Thread(target=work, args=(n,)) for n in ('a', 'b')]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Interleaved, but each thread's draws match what it gets on its own.
        for name in ('a', 'b'):
            seed_task(derive_seed('task', name))
            assert _draw(20) == results[name]
        assert results['a'] != results['b']


class TestParallelPattern:
    """The pattern every parallel call site uses: parent derives, worker calls seed_task."""

    @staticmethod
    def _task(key):
        seed_task(derive_seed('unit_test_task', key))
        return _draw(4)

    def test_process_workers_match_serial(self):
        joblib = pytest.importorskip('joblib')
        set_seed(2024)
        keys = [f'chr{i}' for i in range(1, 9)]
        serial = [self._task(k) for k in keys]
        parallel = joblib.Parallel(n_jobs=4, backend='loky')(
            joblib.delayed(self._task)(k) for k in keys)
        assert parallel == serial

    def test_thread_workers_match_serial(self):
        joblib = pytest.importorskip('joblib')
        set_seed(2024)
        keys = [f'chr{i}' for i in range(1, 9)]
        serial = [self._task(k) for k in keys]
        threaded = joblib.Parallel(n_jobs=4, backend='threading')(
            joblib.delayed(self._task)(k) for k in keys)
        assert threaded == serial

    def test_result_does_not_depend_on_batching(self):
        # What lets the pipeline scatter samples into chunks: a task's draws depend on its key, not
        # on which batch it landed in or what ran before it.
        set_seed(2024)
        alone = self._task('chr7')
        for other in ('chr1', 'chr2', 'chr3'):
            self._task(other)
        assert self._task('chr7') == alone
