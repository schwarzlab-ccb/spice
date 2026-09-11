"""Central control of every random number SPICE draws, so a run is a function of one seed.

SPICE's inference is stochastic throughout: MCMC over event orders, resimulation-based nulls,
bootstrap resampling, randomised tie-breaks. Left on the global `numpy.random` / `random` state, a
run is reproducible only by accident: interpreter start uses OS entropy, each joblib worker is a
fresh process, and thread interleaving decides which thread gets which draw.

This module replaces that with three rules:

1. `set_seed(seed)` fixes the process's base seed. The CLI calls it once per command from
   `params.seed` (or `--seed`).
2. Every draw goes through `np_rng()` / `py_rng()`, returning the *calling thread's* generator
   instead of the shared globals -- two threads never share a stream, so interleaving can't change
   either one's draws.
3. Parallel sections give each task its own stream: the parent derives a seed from stable task
   identity (`derive_seed('all_solutions', sample_id)`) or draws child seeds from its stream
   (`spawn_seeds(n)`), and the worker calls `seed_task(seed)` before starting work. Both are
   reproducible; `derive_seed` also makes results independent of batching/ordering, so the pipeline
   can scatter samples into chunks without changing per-sample results.

Two things this cannot fix, documented rather than silently half-solved:

* **Wall-clock limits.** `time_limit_all_solutions` / `time_limit_mcmc` / CP-SAT's
  `max_time_in_seconds` make the answer depend on machine speed and load -- leave them unset for
  reproducible runs.
"""

import hashlib
import os
import random
import threading

import numpy as np

# Used when neither the config nor --seed nor $SPICE_SEED says otherwise.
DEFAULT_SEED = 42

# Carries the base seed into joblib/loky workers, which start as fresh interpreters and inherit
# only the environment, not module state.
SEED_ENV_VAR = 'SPICE_SEED'

# numpy's legacy RandomState takes seeds in [0, 2**32).
_MAX_SEED = 2 ** 32

_base_seed = None
_local = threading.local()


def set_seed(seed=None):
    """Fix the base seed for this process; returns the seed used.

    `seed=None` falls back to $SPICE_SEED, then DEFAULT_SEED. Also seeds the `numpy.random` /
    `random` globals, since third-party code draws from those and leaving them on OS entropy would
    make them the one non-reproducible part of a run.
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
    """Seed the calling thread/worker for one unit of work; returns the seed used.

    Call at the top of anything run through joblib, with a seed the parent derived (`derive_seed`)
    or drew (`spawn_seeds`) -- never one from the worker's own state.
    """
    return _seed_this_thread(seed)


def derive_seed(*parts):
    """A stable seed for `parts` under the current base seed.

    Same base seed + same parts -> same seed, in any process/order/machine (blake2b, unlike
    `hash()`, isn't per-process randomised). Use for tasks with a natural identity -- a sample id,
    chromosome, (cluster, iteration) pair -- so results survive re-batching, re-ordering, or
    re-running in isolation.
    """
    key = '|'.join([str(get_seed())] + [str(p) for p in parts]).encode('utf-8')
    return int.from_bytes(hashlib.blake2b(key, digest_size=4).digest(), 'big')


def spawn_seeds(n):
    """Draw `n` child seeds from the calling thread's stream.

    Use where tasks have no stable identity and need *fresh* randomness each call (e.g.
    resimulations behind one bootstrap): the parent's stream is deterministic, so children are
    reproducible, but repeated calls still draw differently.
    """
    return [int(x) for x in np_rng().randint(0, _MAX_SEED, size=int(n))]


def np_rng():
    """This thread's `numpy.random.RandomState` (drop-in for the `np.random.*` functions)."""
    if getattr(_local, 'np_state', None) is None:
        _seed_this_thread(get_seed())
    return _local.np_state


def py_rng():
    """This thread's `random.Random` (drop-in for the `random.*` functions).

    Kept separate from `np_rng()`: the APIs aren't interchangeable -- `random.choice` takes any
    sequence (e.g. tuples), while `RandomState.choice` needs 1-D array-like.
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
