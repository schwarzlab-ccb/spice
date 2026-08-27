import os
import re
from io import StringIO
import sys

import pandas as pd
import numpy as np
from tqdm.auto import tqdm

from spice import config, directories
from spice.length_scales import DEFAULT_SEGMENT_SIZE_DICT, DEFAULT_LENGTH_SCALE_BOUNDARIES
from spice.utils import (CALC_NEW, open_pickle, save_pickle, add_filename_suffix)

# Use importlib.resources for accessing package data (works with installed packages)
if sys.version_info >= (3, 9):
    from importlib.resources import files
else:
    try:
        from importlib_resources import files
    except ImportError:
        files = None
from spice.logging import log_debug, get_logger


logger = get_logger('data_loaders')
CHROMS = ['chr' + str(x) for x in range(1, 23)] + ['chrX', 'chrY']
DATA_LOADERS_DIR = os.path.join(directories['results_dir'], 'data_loaders')

# --- Reference assembly -------------------------------------------------------------------------
# SPICE's packaged coordinate tables (chromosome lengths, centromeres, and the observed
# centromere/telomere positions) were hg19-only, and nothing checked it: an hg38 cohort produced
# plausible, wrong output rather than an error. The build is now a config parameter,
# `params.assembly`, defaulting to hg19 so every existing config and cohort is unchanged.
#
# Why a config key and not an env var or a CLI flag: cli.py imports nothing assembly-dependent at
# module level -- every heavy module is imported INSIDE its main_* function, after
# spice.load_config() -- so by the time any `CHROM_LENS = load_chrom_lengths()` module constant is
# evaluated, the config is already loaded. That makes a config key work with no change to those
# constants or their call sites. It is also the only option that gets RECORDED: the pipeline stages an
# immutable copy of its config per run, so the build a table was produced under stays visible
# afterwards, which an env var never would be.
#
# hg19 keeps the original filenames, so its tables are byte-identical and no file was renamed.
SUPPORTED_ASSEMBLIES = ('hg19', 'hg38')
DEFAULT_ASSEMBLY = 'hg19'


def get_assembly():
    """The reference build for this run, from `params.assembly` (default hg19)."""
    # Read the LIVE binding rather than the `config` imported at module scope: load_config()
    # reassigns spice.config, so a stale reference would silently return the default.
    import spice
    cfg = getattr(spice, 'config', None)
    name = DEFAULT_ASSEMBLY
    if isinstance(cfg, dict):
        name = (cfg.get('params') or {}).get('assembly', DEFAULT_ASSEMBLY) or DEFAULT_ASSEMBLY
    if name not in SUPPORTED_ASSEMBLIES:
        raise ValueError(
            f"params.assembly = {name!r} is not supported (expected one of {SUPPORTED_ASSEMBLIES})")
    return name


def _assembly_filename(stem):
    """objects/<stem>.tsv on hg19 (unchanged), objects/<stem>_<assembly>.tsv otherwise."""
    assembly = get_assembly()
    return f'{stem}.tsv' if assembly == DEFAULT_ASSEMBLY else f'{stem}_{assembly}.tsv'


def _read_object_tsv(filename, **read_csv_kwargs):
    """Read a packaged objects/*.tsv, with an error that names the generator when it is missing."""
    if files is None:
        raise FileNotFoundError(f"importlib.resources unavailable for {filename}")
    try:
        content = files('spice').joinpath('objects', filename).read_text()
    except (TypeError, ImportError, AttributeError, FileNotFoundError) as exc:
        raise FileNotFoundError(
            f"Could not find {filename} in spice/objects/. Assembly-specific tables for a "
            f"non-default build are generated outside this package: the static ones "
            f"(chrom_lengths, centromeres, centromeres_ext) by "
            f"pipeline-peak-detection/src/spice/make_assembly_tables.py, and the observed ones "
            f"(centromeres_observed, telomeres_observed) by "
            f"create_observed_centromeres_and_telomeres() on that cohort's own final_events.tsv "
            f"(see its docstring -- it needs event inference to have run first)."
        ) from exc
    return pd.read_csv(StringIO(content), sep='\t', **read_csv_kwargs)


def resolve_copynumber_file(return_raw=False) -> str:
    """Resolve the chromosome segments file path. """
    from spice import config, directories
    name = config.get('name')
    data_dir = config['directories']['data_dir']
    orig = config['input_files']['copynumber']
    processed = os.path.join(data_dir, f"{name}_processed.tsv")
    if not return_raw:
        orig = add_filename_suffix(orig, '_split')
        processed = os.path.join(data_dir, f"{name}_processed_split.tsv")

    # Prefer processed split file if available, else original
    cur_file = orig
    if processed and os.path.exists(processed):
        cur_file = processed
    
    if not os.path.isabs(cur_file):
        cur_file = os.path.join(directories['base_dir'], cur_file)
    log_debug(logger, f"Resolved chrom_segments_file: {cur_file}")
    return cur_file


def _resolve_optional_input_file(path):
    if path is None or isinstance(path, bool):
        return None
    if isinstance(path, str):
        if path.strip() == '' or path.strip().lower() == 'none':
            return None
    if os.path.isabs(path):
        return path
    base_dir = config.get('directories', {}).get('base_dir', None)
    if base_dir is None:
        return path
    return os.path.join(base_dir, path)


def load_sv_data(sv_data_file, chrom_id=None):
    """Load optional SV data file and return chromosome-specific rows.

    If chrom_id is None, returns the full dataframe without filtering.
    Accepts files with a pre-built 'chrom_id' column or with separate
    'sample_id' and 'chrom' columns, in which case chrom_id is constructed
    as ``sample_id + ':' + chrom``.
    """
    sv_data_file = _resolve_optional_input_file(sv_data_file)
    if sv_data_file is None:
        return None

    sv_data = pd.read_csv(sv_data_file, sep=None, engine='python')
    if 'chrom_id' not in sv_data.columns:
        assert {'sample_id', 'chrom'}.issubset(sv_data.columns), (
            f'SV data must have a "chrom_id" column or both "sample_id" and "chrom" columns. '
            f'Got: {sorted(sv_data.columns)}')
        # Normalize chromosome labels the same way as copy-number data (see
        # load_raw_copy_number_data), so chrom_id joins against copy-number
        # chrom_ids succeed regardless of the SV file's chromosome naming convention.
        sv_data['chrom'] = format_chromosomes(sv_data['chrom'])
        sv_data['chrom_id'] = sv_data['sample_id'].astype(str) + ':' + sv_data['chrom'].astype(str)
    required_columns = {'chrom_id', 'svclass', 'start', 'end'}
    missing_columns = required_columns - set(sv_data.columns)
    assert not missing_columns, f'Missing required SV columns: {sorted(missing_columns)}'
    log_debug(logger, f'Loaded SV data from {sv_data_file} with {len(sv_data)} rows')
    if chrom_id is not None:
        sv_data = sv_data.query('chrom_id == @chrom_id')
    return sv_data


def load_final_events():
    if 'final_events' in config['input_files']:
        filename = config['input_files']['final_events']
        logger.info(f"Loading final events from config path: {filename}")
    else:
        results_dir = os.path.join(directories['results_dir'], config['name'])
        if not os.path.exists(os.path.join(results_dir, 'final_events.tsv')):
            raise FileNotFoundError(f"final_events.tsv not found in dir {results_dir}. Run SPICE event inference first")
        filename = os.path.join(results_dir, 'final_events.tsv')
        logger.info(f"Loading final events from results dir: {filename}")
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Final events file not found at {filename}. Run SPICE event inference first")
    final_events_df = pd.read_csv(filename, sep='\t', dtype={'cn': str, 'diff': str})
    verify_assembly(final_events_df)
    return final_events_df


def verify_assembly(events_df, raise_on_mismatch=True):
    """Cross-check the configured assembly against the coordinates actually present.

    Without this, a wrong (or silently-ignored) `params.assembly` is undetectable: every downstream
    number stays plausible. An event running past its chromosome's length in the configured build is
    proof the build is wrong, so it is a hard error rather than a warning. Note it can only catch a
    build whose chromosomes are SHORTER than the data -- hg38 read as hg19 trips 8 chromosomes
    (chr18 +2.30 Mb, chr17 +2.06 Mb, ...), while hg19 read as hg38 does not, so the log line below
    matters as much as the assertion.
    """
    assembly = get_assembly()
    if events_df is None or len(events_df) == 0 or 'chrom' not in events_df or 'end' not in events_df:
        return
    lengths = load_chrom_lengths()
    observed = events_df.groupby('chrom')['end'].max()
    over = {c: (int(v), int(lengths.loc[c])) for c, v in observed.items()
            if c in lengths.index and v > lengths.loc[c]}
    n_exact = sum(1 for c, v in observed.items() if c in lengths.index and v == lengths.loc[c])
    logger.info(f"Reference assembly: {assembly} "
                f"({len(observed)} chromosomes in the events, {n_exact} reaching the exact "
                f"chromosome length)")
    if over:
        detail = '; '.join(f"{c}: max end {v:,} > {assembly} length {l:,}" for c, (v, l) in over.items())
        msg = (f"Events exceed {assembly} chromosome lengths on {len(over)} chromosome(s) -- "
               f"params.assembly is wrong for this cohort. {detail}")
        if raise_on_mismatch:
            raise ValueError(msg)
        logger.warning(msg)


def load_segmentation(size=None, data_loaders_dir_top=DATA_LOADERS_DIR):
    # import here to avoid circular imports
    from spice.segmentation import create_segmentation
    cur_filename = os.path.join(data_loaders_dir_top, 'segmentations', f'segmentation_{int(size)}.pickle')
    if not os.path.exists(cur_filename):
        logger.info(f'Creating segmentation with size {size}')
        if size is not None:
            segmentation = create_segmentation(size)
            save_pickle(segmentation, cur_filename)
        else:
            raise ValueError('Segmentation file not found and size is None')
    else:
        segmentation = open_pickle(cur_filename, fail_if_nonexisting=True)

    return segmentation


def load_raw_copy_number_data(input_file, alleles=['cn_a', 'cn_b']):
    data = pd.read_csv(input_file, sep='\t')
    data = (data
            .infer_objects()
            .rename(
                {
                    'chr': 'chrom',
                    'sample': 'sample_id',
                    'major_cn': 'cn_a',
                    'minor_cn': 'cn_b',
                    'cn': 'total_cn',
                    'total': 'total_cn',
                },
                axis=1,
            ))

    required_cols = ['sample_id', 'chrom', 'start', 'end'] + list(alleles)
    missing_cols = [col for col in required_cols if col not in data.columns]
    if len(missing_cols) > 0:
        raise ValueError(f'Missing required columns in input file {input_file}: {missing_cols}')

    data = data[required_cols]
    for allele in alleles:
        data[allele] = data[allele].astype('int64')
        data.loc[data[allele] > 8, allele] = 8

    data['chrom'] = format_chromosomes(data['chrom'])
    # data = data.set_index(['sample_id', 'chrom', 'start', 'end'])
    # data = data.sort_index()

    data['width'] = data.eval('end - start')

    return data


def load_centromeres(extended=True, observed=False, pad=None):
    '''Create file using create_observed_centromeres_and_telomeres'''

    assert not (extended and observed), 'Cannot have both extended and observed centromeres'
    stem = 'centromeres_ext' if extended else (
        'centromeres_observed' if observed else 'centromeres'
    )
    centromeres = _read_object_tsv(
        _assembly_filename(stem),
        header=[0, 1] if observed else [0],
        index_col=0,
    )

    if pad is not None:
        centromeres['centro_start'] = np.maximum(centromeres['centro_start'] - pad, 0)
        centromeres['centro_end'] = centromeres['centro_end'] + pad
    return centromeres


def load_telomeres_observed():
    '''Create file using create_observed_centromeres_and_telomeres'''
    telomeres_observed = _read_object_tsv(
        _assembly_filename('telomeres_observed'),
        header=[0, 1],
        index_col=0,
    )

    return telomeres_observed


def create_observed_centromeres_and_telomeres(final_events_df, segment_size_dict=DEFAULT_SEGMENT_SIZE_DICT,
                                              length_scale_boundaries=DEFAULT_LENGTH_SCALE_BOUNDARIES):
    # import here to avoid circular imports
    centromeres = load_centromeres(extended=False)

    actual_centro_pos = pd.DataFrame(index=CHROMS[:-1], columns=pd.MultiIndex.from_product([['small', 'mid1', 'mid2', 'large'], ['centro_start', 'centro_end']]))
    actual_telomere_pos = pd.DataFrame(index=CHROMS[:-1], columns=pd.MultiIndex.from_product([['small', 'mid1', 'mid2', 'large'], ['chrom_start', 'chrom_end']]))
    for cur_chrom in tqdm(CHROMS[:-1]):
        for cur_length_scale in ['small', 'mid1', 'mid2', 'large']:

            cur_length_scale_border = length_scale_boundaries[cur_length_scale]
            cur_events = final_events_df.query('pos == "internal" and chrom == @cur_chrom').copy()
            centro_center = centromeres.loc[cur_chrom].mean()

            if centromeres.loc[cur_chrom, 'centro_start'] == 0 or cur_chrom in ['chr13', 'chr14', 'chr15', 'chr21', 'chr22']:
                cur_start = 0
            else:
                cur_start = cur_events.query('end < @centro_center')['end'].max()
                if np.isnan(cur_start):
                    cur_start = centromeres.loc[cur_chrom, 'centro_start']
                else:
                    cur_start = int(np.floor(cur_start/segment_size_dict[cur_length_scale])*segment_size_dict[cur_length_scale])
            actual_centro_pos.loc[cur_chrom, (cur_length_scale, 'centro_start')] = cur_start

            cur_end = cur_events.query('start > @centro_center')['start'].min()
            cur_end = int(np.ceil(cur_end/segment_size_dict[cur_length_scale])*segment_size_dict[cur_length_scale])
            actual_centro_pos.loc[cur_chrom, (cur_length_scale, 'centro_end')] = cur_end

            actual_telomere_pos.loc[cur_chrom, (cur_length_scale, 'chrom_start')] = cur_events['start'].min()
            actual_telomere_pos.loc[cur_chrom, (cur_length_scale, 'chrom_end')] = cur_events['end'].max()

    output_dir = os.path.join(directories['results_dir'], 'data_loaders')
    os.makedirs(output_dir, exist_ok=True)
    actual_telomere_pos.to_csv(os.path.join(output_dir, 'telomeres_observed.tsv'), sep='\t')
    actual_centro_pos.to_csv(os.path.join(output_dir, 'centromeres_observed.tsv'), sep='\t')


def load_chrom_lengths():
    """Chromosome lengths for the configured assembly (see get_assembly)."""
    return _read_object_tsv(_assembly_filename('chrom_lengths')).set_index('chrom')['chrom_length']


def format_chromosomes(ds):
    '''copied from medicc.tools'''

    ds = ds.astype('str')
    pattern = re.compile(r"(chr|chrom)?(_)?(0)?((\d+)|X|Y)", flags=re.IGNORECASE)
    matches = ds.apply(pattern.match)
    matchable = ~matches.isnull().any()
    if matchable:
        newchr = matches.apply(lambda x:f"chr{x[4].upper():s}")
        numchr = matches.apply(lambda x:int(x[5]) if x[5] is not None else -1)
        chrlevels = np.sort(numchr.unique())
        chrlevels = np.setdiff1d(chrlevels, [-1])
        chrcats = [f"chr{i}" for i in chrlevels]
        if 'chrX' in list(newchr):
            chrcats += ['chrX',]
        if 'chrY' in list(newchr):
            chrcats += ['chrY',]
        newchr = pd.Categorical(newchr, categories=chrcats)
    else:
        logger.warning("Could not match the chromosome labels. Rename the chromosomes according chr1, "
                      "chr2, ... to avoid potential errors."
                      "Current format: {}".format(ds.unique()))
        newchr = pd.Categorical(ds, categories=ds.unique())
    assert not newchr.isna().any(), "Could not reformat chromosome labels. Rename according to chr1, chr2, ..."
    return newchr
