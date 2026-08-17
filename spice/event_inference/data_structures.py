from collections import namedtuple

ChromData = namedtuple(
    'ChromData',
    ['id', 'sample', 'chrom', 'allele', 'cn_profile', 'string', 'dist', 'n_events', 'has_wgd', 'copynumber_file'])
Diff = namedtuple(
    'Diff',
    ['diff', 'is_gain', 'wgd'])
FullPaths = namedtuple(
    'FullPaths',
    ['id', 'sample', 'chrom', 'allele', 'cn_profile', 'n_solutions', 'n_events', 'is_wgd', 'solved', 'events',
     'solutions'])


class McmcGuardExceeded(RuntimeError):
    """A configured ceiling stopped an MCMC work unit before it could run away.

    Lives here (a leaf module with no spice imports) so both the MCMC and the CP-SAT layer can raise
    it without an import cycle.

    Raised as a normal exception ON PURPOSE: `_run_batch` already wraps every unit in try/except and
    records failures to `failed_reports.tsv`, so a ceiling hit costs one reported unit instead of the
    whole chunk. That is the entire point — an unbounded solve inside CP-SAT (C++) eventually dies of
    SIGSEGV or the cgroup OOM killer, and neither can be caught per-unit, so it takes the process and
    every other sample in the chunk with it (CNSistent chunk_0024, unit SP124441:chr9:cn_a).

    Equally deliberate: the ceilings RAISE rather than degrade. A CP-SAT solve that silently gave up
    at its time limit would return "no LOH solution", flipping a filter decision and quietly changing
    the reconstruction; a truncated iteration budget would quietly return a worse optimum. A reported
    failure is honest, a silently different answer is not.
    """
