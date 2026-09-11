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

    Lives here (leaf module, no spice imports) so both MCMC and CP-SAT can raise it without a cycle.

    Raised rather than degraded on purpose: an unbounded solve can SIGSEGV/OOM the whole process,
    so `_run_batch` catches this per-unit and logs to `failed_reports.tsv` instead of losing the
    chunk. A silent fallback (e.g. "no LOH solution") would quietly corrupt the reconstruction.
    """
