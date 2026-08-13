"""Central control of every random number SPICE draws, so a run is a function of one seed.

SPICE's inference is stochastic throughout: MCMC over event orders, resimulation-based nulls,
bootstrap resampling, randomised tie-breaks. Left on the global `numpy.random` / `random` state a
run is reproducible only by accident -- the state at interpreter start comes from OS entropy, each
joblib worker is a fresh process with its own state, and under a threading backend the interleaving
of threads decides which thread gets which draw.

This module replaces that with three rules:

1. `set_seed(seed)` fixes the base seed for the process. The CLI calls it once per command from
   `params.seed` in the config (or `--seed`).
2. Every draw goes through `np_rng()` / `py_rng()`, which return the *calling thread's* generators
   rather than the shared globals. Two threads therefore never share a stream, and their
   interleaving cannot change either one's draws.
3. Parallel sections give each task its own stream. The parent either derives a seed from stable
   task identity (`derive_seed('all_solutions', sample_id)`) or draws child seeds from its own
   stream (`spawn_seeds(n)`); the worker calls `seed_task(seed)` before doing any work. Both are
   reproducible, and `derive_seed` additionally makes a task's result independent of how tasks are
   batched or ordered -- which is what lets the pipeline scatter samples into chunks without
   changing per-sample results.

Two things this cannot fix, documented rather than silently half-solved:

* **Wall-clock limits.** `time_limit_all_solutions` / `time_limit_mcmc` / CP-SAT's
  `max_time_in_seconds` make the answer depend on machine speed and load. Determinism holds only
  while no such limit binds; leave them unset for reproducible runs.
* **`PYTHONHASHSEED`.** Python randomises the hash of strings per process, which reorders iteration
  over sets of strings -- no seed can reach that. Where such an ordering decided results it has been
  made canonical instead (see the sorted() calls in events_from_graph, and the sorted os.listdir()
  calls that fix file order); `PYTHONHASHSEED=0` is a cheap extra guard, not a requirement.
"""

import hashlib
import os
import random
import threading

import numpy as np

# Used when neither the config nor --seed nor $SPICE_SEED says otherwise.
DEFAULT_SEED = 42

# Carries the base seed into joblib/loky worker processes, which start as fresh interpreters and so
# do not inherit module state -- only the environment.
SEED_ENV_VAR = 'SPICE_SEED'

# numpy's legacy RandomState takes seeds in [0, 2**32).
_MAX_SEED = 2 ** 32

_base_seed = None
_local = threading.local()


def set_seed(seed=None):
    """Fix the base seed for this process. Returns the seed actually used.

    `seed=None` falls back to $SPICE_SEED, then DEFAULT_SEED. Besides the per-thread generators this
    also seeds the `numpy.random` / `random` globals: third-party code draws from those, and leaving
    them on OS entropy would make them the one non-reproducible part of a run.
    """
    global _base_seed
    if seed is None:
        seed = os.environ.get(SEED_ENV_VAR, DEFAULT_SEED)
    seed = int(seed) % _MAX_SEED
    _base_seed = seed
    os.environ[SEED_ENV_VAR] = str(seed)
    _seed_this_thread(seed)
    np.random.seed(seed)
    random.seed(seed)
    return seed


def get_seed():
    """The process's base seed, initialising it from $SPICE_SEED / DEFAULT_SEED if unset."""
    if _base_seed is None:
        set_seed(None)
    return _base_seed


def seed_task(seed):
    """Seed the calling thread (or worker process) for one unit of work. Returns the seed used.

    Call this at the top of anything run through joblib, with a seed the parent derived
    (`derive_seed`) or drew (`spawn_seeds`) -- never one taken from the worker's own state.
    """
    return _seed_this_thread(seed)


def derive_seed(*parts):
    """A stable seed for `parts` under the current base seed.

    Same base seed + same parts -> same seed, in any process, in any order, on any machine (blake2b,
    unlike `hash()`, is not per-process randomised). Use for tasks with a natural identity -- a
    sample id, a chromosome, a (cluster, iteration) pair -- so their results survive being
    re-batched, re-ordered or re-run in isolation.
    """
    key = '|'.join([str(get_seed())] + [str(p) for p in parts]).encode('utf-8')
    return int.from_bytes(hashlib.blake2b(key, digest_size=4).digest(), 'big')


def spawn_seeds(n):
    """Draw `n` child seeds from the calling thread's stream.

    Use where tasks have no stable identity and each call is meant to see *fresh* randomness (e.g.
    the resimulations behind one bootstrap): the parent's stream is itself deterministic, so the
    children are reproducible, while repeated calls still get different draws.
    """
    return [int(x) for x in np_rng().randint(0, _MAX_SEED, size=int(n))]


def np_rng():
    """This thread's `numpy.random.RandomState` (drop-in for the `np.random.*` functions)."""
    if getattr(_local, 'np_state', None) is None:
        _seed_this_thread(get_seed())
    return _local.np_state


def py_rng():
    """This thread's `random.Random` (drop-in for the `random.*` functions).

    Kept separate from `np_rng()` because the two APIs are not interchangeable: `random.choice`
    takes any sequence (a list of tuples included), while `RandomState.choice` needs 1-D array-like.
    """
    if getattr(_local, 'py_state', None) is None:
        _seed_this_thread(get_seed())
    return _local.py_state


def _seed_this_thread(seed):
    seed = int(seed) % _MAX_SEED
    _local.seed = seed
    _local.np_state = np.random.RandomState(seed)
    _local.py_state = random.Random(seed)
    return seed
