"""
Measure how the genome-wide fit improves as loci are added one at a time, in decreasing order
of significance (permutation.fitness_statistic).

Standalone, opt-in diagnostic -- NOT part of `spice loci_detection` / `loci_assignment` /
`components` and never runs automatically. The ONLY required input is which pipeline's output to
read; everything else -- the combined loci table, selection points, and data_per_length_scale --
is loaded straight from that pipeline's own conventional on-disk layout (see
spice/tsg_og/add_loci_one_by_one.py for the exact paths per mode):

  loci_inference  -- a `spice loci_detection` run that reached the 'combine' step
  loci_assignment -- a `spice loci_assignment` run
  components      -- a `spice components` run

Usage:
    python scripts/add_loci_one_by_one.py <mode> [--config PATH] [--cores N] [--overwrite]

Example:
    python scripts/add_loci_one_by_one.py loci_inference --config configs/pcawg.yaml --cores 8
    python scripts/add_loci_one_by_one.py components --config configs/components_example.yaml
"""
import argparse
import os

import spice


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('mode', choices=['loci_inference', 'loci_assignment', 'components'],
                         help="Which pipeline's output to load everything else from disk for.")
    parser.add_argument('--config', dest='config_path', default=None,
                         help='Path to the SPICE config.yaml for the run to evaluate. Defaults to '
                              'the SPICE_CONFIG environment variable (the same discovery spice '
                              'itself uses) -- only needed on the command line if that is not set.')
    parser.add_argument('--components-label', choices=['all', 'filtered'], default='filtered',
                         help="Only used with mode=components: which components table/fit to use "
                              "(default: filtered)")
    parser.add_argument('--chrom', nargs='+', default=None,
                         help='Restrict to these chromosomes (default: every chromosome present '
                              "in the mode's own output)")
    parser.add_argument('--cores', '-j', type=int, default=1,
                         help='Parallel workers across chromosomes (default: 1)')
    parser.add_argument('--n-iterations-base', type=int, default=3_000,
                         help='Base optimizer iteration count for a newly-added locus (scaled by '
                              'sqrt(N_loci) internally, default: 3000)')
    parser.add_argument('--overwrite', action='store_true',
                         help="Recompute a chromosome's result even if it is already cached")
    parser.add_argument('--debug', action='store_true', help='Enable DEBUG logging')
    args = parser.parse_args()

    config_path = args.config_path or os.environ.get('SPICE_CONFIG')
    if config_path is None:
        raise ValueError('No config found: pass --config or set the SPICE_CONFIG environment '
                          'variable to the run you want to evaluate.')
    spice.load_config(config_path)
    from spice import config
    from spice.logging import configure_logging, get_logger
    from spice.tsg_og.add_loci_one_by_one import run_add_loci_one_by_one

    if 'name' not in config or not config['name']:
        raise ValueError("Config file must specify a 'name' field.")

    log_level = 'DEBUG' if args.debug else config['params'].get('logging_level', 'INFO')
    configure_logging(log_mode='terminal', log_dir=config['directories']['log_dir'],
                       config_name=config['name'], level=log_level)
    logger = get_logger('SPICE', spice_prefix=False)

    logger.info(f"Running add_loci_one_by_one in '{args.mode}' mode for project '{config['name']}'")
    _, summary_df = run_add_loci_one_by_one(
        mode=args.mode,
        config=config,
        chroms=args.chrom,
        N_iterations_base=args.n_iterations_base,
        cores=args.cores,
        overwrite=args.overwrite,
        components_label=args.components_label,
    )
    logger.info(f'Done. {len(summary_df)} rows written.')


if __name__ == '__main__':
    main()
