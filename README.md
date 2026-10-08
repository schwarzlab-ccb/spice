# SPICE: Selection Patterns In somatic Copy-number Events

![](doc/logo_banner.png)

**SPICE**, Selection Patterns In somatic Copy-number Events, is a framework that
1) infers discrete copy-number events from allele-specific profiles,
2) detects loci of selection in the copy-number data and 
3) can assign loci of selection to copy-number data

See the [accompanying BioRxiv preprint](https://www.biorxiv.org/content/10.64898/2026.03.01.708809v1) for more information.

## 0. Installation

**Conda Quick start (recommended)** 
```bash
conda env create -f environment.yml     # creates the `spice` env (run from this directory)
conda activate spice
```
This installs `medicc2` (which supplies the required `fstlib` + `openfst`), pins `numpy < 2` for the
FST ABI, and installs scna-spice in editable mode. **Do not `pip install` numpy / scipy / pandas
into this env afterwards** — layering a pip numpy 2.x over the conda stack corrupts numpy and breaks
scipy. The manual steps 0.1–0.4 below are the alternative if you'd rather assemble the env yourself.

### 0.1. Prerequisites
- Python >= 3.8
- medicc2 (including openfst)

Install MEDICC2 using conda/mamba as it requires compilation of source files
```bash
conda install -c bioconda -c conda-forge medicc2
```

Or better directly create a new conda environment with MEDICC2 inside of it 
```bash
conda create -n spice_env -c conda-forge -c bioconda medicc2
conda activate spice_env
```

### 0.2. Install from pip (recommended)

After installing MEDICC2 through conda/mamba simply install spice using pip

```bash
pip install scna-spice
```

### 0.3 Install from source

Clone the repository:
```bash
git clone git@github.com:schwarzlab-ccb/spice.git
cd spice
```

And then install in development mode:
```bash
pip install -e .
```


### 0.4 Optional Dependencies

To use the extra preprocessing also install CNSistent:
```bash
pip install CNSistent
```

## 1. Configuration

SPICE uses a configuration file for each run which are specified using the `--config` flag.
This means you can keep multiple configs (e.g., in `configs/`) and select them at runtime.

Parameters and directories not specified in the provided config file are taken from the default config file `default_config.yaml`.
Each config must specify `name` and `directories.base_dir`.

### 1.1 Minimal `config.yaml` override example
Each config must contain a name, a base directory, and the location of the input copy-number file like so:
```yaml
name: example_run
directories:
   base_dir: /path/to/project
input_files:
   copynumber: data/example_data.tsv
```

For other parameters that can be modified, see [the default configuration](spice/objects/default_config.yaml).

### 1.2 Relative vs absolute paths

- `directories.*` entries (e.g., `data_dir`, `results_dir`, `log_dir`) as well as input files can be given as relative or absolute paths.
   - If relative, SPICE resolves them against `directories.base_dir`.
   - If absolute, SPICE uses them as-is.

### 1.3 Reproducibility (`params.seed`)

SPICE analysis uses stochastic steps (MCMC over event orders, resimulated nulls, bootstrap resampling,
randomised tie-breaks). Detection and event inference derive their random streams from `params.seed` (default 42),
overridable per command with `--seed`; the seed in use is written to the log at startup. Re-running
the same command with identical inputs, settings and seed reproduces its random streams; change the seed to get
an independent replicate. Component model preparation uses its own
`components.model_seed`; component fitness optimization uses `params.seed`.

The seed is threaded through parallel work as well, so results do not depend on `--cores`: each task
(a sample, a chromosome, a bootstrap iteration, a resimulation) gets its own stream keyed on *what
it is* rather than on when it ran. Detecting loci on `chr7` alone therefore gives the same answer as
detecting it as part of a whole-genome run, which is what makes scattering the work over a cluster
safe. See `spice/random_state.py` for the mechanism.

`spice permute` uses the same base seed as loci detection. `--index` selects a permutation
under that seed; it does not replace `--seed`. Use the same config and base seed for every
scattered unit and the pooling command:

```bash
spice permute --config cohort_loci.yaml --seed 7 --index 3 --chrom chr7
spice permute --config cohort_loci.yaml --seed 7 --pool
```

Run all required `(index, chromosome)` units before pooling. Omit `--seed` to use `params.seed`.
Rerunning a permutation unit invalidates its combined table and the pooled null. Pool again
once all units finish. `spice permute --config <config> --pool --overwrite` also forces
recombination of existing per-chromosome results, without rerunning detection.

Production supports only the `rotate` null and `combined_fitness` (A+B) with
`zpool_chrom` calibration. These settings apply to observed loci and their permutation null.

The CLI/config default is `loci_detection.N_bootstrap: 100` for the signal's
2.5% and 97.5% bootstrap quantiles. This is a runtime/storage compromise; use
`1000` for more stable tail estimates when resources permit. Signal resampling
work and stored bootstrap arrays grow approximately linearly with this count,
so 1000 costs about ten times as much as 100 for that stage, not for the entire
pipeline. `N_bootstrap_for_widths` remains 10: increasing the signal count does
not increase the number of width-fitting optimizations. Signal bounds also
participate in detection/filtering, so changing the count can change fitted peaks
and downstream runtime, not just the reported CI score.

Apply a changed count to a fresh run/output directory with matching observed and
null settings. The bootstrap filename contains the count, but the derived
`data_per_length_scale/<chrom>.pickle` filename does not; resuming old caches
could retain the old bounds. Existing saved runs and frozen source snapshots
keep their original settings.

Loci preprocessing and each fitting stage use separate random streams. With the same
inputs, parameters, and seed, rebuilding a stage gives the same result whether preceding
stages were computed, cached, or loaded during a resumed run. When changing the seed,
inputs, or parameters, use a new run name/directory so earlier caches are not reused.

One thing falls outside the seed: **Wall-clock limits.** `params.time_limit_all_solutions` / `time_limit_mcmc` (and CP-SAT's internal time limit) make the result depend on machine speed and load. Record these limits alongside the seed when reproducing an event-inference run.

## 2. Usage Overview

The main analysis commands are:
- **event_inference**: Infer discrete copy-number events from allele-specific profiles
- **loci_detection**: Detect recurrent copy-number loci across samples
- **permute**: Build or pool the rotate permutation null used for locus scoring
- **loci_assignment**: Fit cohort-level fitness at predefined locus positions
- **components**: Group scored loci across detection seeds, select components and jointly refit all/filtered sets
- **plotting**: Generate visualizations of inferred events and detected loci


### 2.1 Top-level execution examples

For event inference the example config `configs/events_example.yaml` can be used.
For loci detection and assignment the example config `configs/loci_example.yaml` can be used.

```bash
# Event inference
spice event_inference --config configs/events_example.yaml

# Loci detection
spice loci_detection --config configs/loci_example.yaml

# Loci assignment
spice loci_assignment --config configs/loci_example.yaml

# Components across independent detection seeds
spice components --config configs/components_example.yaml

# Plotting
spice plotting --config <path/to/config> --plot-events-per-sample <SAMPLE_ID>
```

---

## 3. Event Inference

Event inference infers discrete copy-number events from allele-specific copy-number profiles by enumerating valid evolutionary paths through the copy-number landscape and selecting the most likely path using k-nearest neighbors or MCMC sampling.

**Note that spice automatically deletes previous runs of the same name when it is rerun.**

### 3.1 Pipeline Overview

The event inference pipeline runs 6 steps:
- `preprocessing`: Extra preprocessing (filling telomeres, phasing, etc.)
- `split`: Split haplotypes and preprocess input
- `all_solutions`: Enumerate all valid evolutionary paths
- `disambiguate`: Select best path using k-nearest neighbors
- `large_chroms`: Use MCMC sampling for chromosomes with many events
- `combine`: Combine all events into the final output

For each step, nonWGD and WGD samples are treated separately and samples are split by chromosome and allele to give the file IDs "sample:chrom:allele". For each step, each ID is calculated separately and stored as separate files.

Intermediate files can be removed using
```bash
spice event_inference --clean --config <path/to/config>
```

### 3.2 Expected Input

SPICE expects tab-separated input files with copy-number segments. See example file `data/example_data.tsv`.

**Required columns:**
- `sample_id`: Sample identifier
- `chrom`: Chromosome name
- `start`: Segment start position
- `end`: Segment end position
- `cn_a`: Copy number for allele A (haplotype-specific)
- `cn_b`: Copy number for allele B (haplotype-specific)

**Optional files:**
- `wgd_status`: TSV with WGD status per sample (see section 3.2.2)
- `xy_status`: TSV with sex status per sample (see section 3.2.3)
- `sv`: Pickle file (`.pickle`) with SV calls used for SV-constrained event matching (see section 3.2.4)

Total copy-number mode can be enabled by setting `params.total_cn: True` in the config file.

#### 3.2.1 Total copy-number mode (`params.total_cn`)

Set `params.total_cn: True` to run event inference on single-channel total copy-number input.

**Required column changes in this mode:**
- Use `total_cn` instead of `cn_a`/`cn_b` in the input TSV
- Keep the same segment metadata columns: `sample`, `chrom`, `start`, `end`

Note that in the output the total copy-number will be displayed as allele `cn_a`

#### 3.2.2 WGD Detection

SPICE supports two ways to determine WGD (whole genome duplication) status per sample. The pipeline branches on WGD status and uses different FSTs and neutral CN values accordingly.

- Provided status via `wgd_status` file:
   - Set `input_files.wgd_status` in your config to a TSV file.
   - The file must have two columns: first column is the sample identifier (used as index), second column named `wgd` with boolean values (`True`/`False`).
   - Example:
      ```tsv
      sample_id	wgd
      SA123	True
      SA456	False
      ```

- Inferred WGD status:
   - If `input_files.wgd_status` is missing or empty, SPICE infers WGD using copy-number data and the method specified by `params.wgd_inference_method`.
   - Supported values:
      - `major_cn`: heuristic whether at least half of the major copy-number is greater or equal to 2
      - `ploidy_loh`: PCAWG-style rule combining ploidy and LOH fraction

Notes
- WGD status impacts neutral CN values and constraint solving throughout the pipeline, so ensure this is set or inferred correctly.
- For haplotype-specific data, neutral CN is 1 (noWGD) vs 2 (WGD); for total CN, 2 vs 4 respectively.

#### 3.2.3 Sex (XY/XX) Detection

SPICE supports resolving sample sex (XY vs XX) either via a provided file or automatic inference. This affects handling of `chrX` and `chrY` in preprocessing and splitting.

- Provided status via `xy_status` file:
   - Set `input_files.xy_status` in your config to a TSV file.
   - The file must have two columns: first column is the sample identifier (used as index), second column named `xy` with boolean values (`True`/`False`) indicating XY (male) vs XX (female).
   - Example:
      ```tsv
      sample_id	xy
      SA123	True
      SA456	False
      ```

- Inferred XY status:
   - If `input_files.xy_status` is missing or empty, SPICE infers XY by checking if any segments exist on chromosome `chrY` for a sample.

Effects
- For XY samples with haplotype-specific CN, the minor copy number of `chrX` and `chrY` is set to 0 during preprocessing and splitting.
- For XX samples, `chrY` is excluded (no segments on `chrY`).

#### 3.2.4 Structural Variant (SV) Input

SV support in `event_inference` is optional and enabled by setting the config key `input_files.sv` to the path of the structural variant calls in the config.

**Expected columns**
- `sample_id`: Sample identifier
- `chrom`: Chromosome name
- `start`: Segment start position
- `end`: Segment end position
- `svclass`: Type of SV, must be either "DUP" or "DEL"


### 3.3 Expected Output

Results are saved in `results/{name}/`

**Main outputs:**

- `final_events.tsv`: Summary of inferred events per sample/chromosome/allele with event types, coordinates, and validation metrics
- `events_summary.tsv`: Summary statistics for each ID (sample, chromosome, allele combination), including number of events and path selection method

**Intermediate files** (with separate directories for WGD and non-WGD profiles):
- `chrom_data_full/`: Preprocessed chromosome data
- `full_paths_single_solution/`: Chromosomes with unique solutions
- `full_paths_multiple_solutions/`: Chromosomes requiring kNN selection
- `knn_solved_chroms/`: Results from kNN selection
- `mcmc_solved_chroms_large/`: Results from MCMC sampling

Intermediate files can be removed using

```bash
spice event_inference --clean --config <path/to/config>
```

### 3.4 Preprocessing Step Details

The preprocessing step runs only when `--run-preprocessing` is provided and prepares the input for robust event inference. It performs:

- Data normalization: ensures chromosome names use `chr` prefix; converts starts/ends to integers and adjusts starts to 0-based.
- CN capping and filtering: caps copy numbers at 8; removes segments shorter than 1kb.
- WGD resolution: loads from `wgd_status.tsv` or infers as described in section 3.2.2.
- Sex resolution: loads from `xy_status.tsv` or infers by presence of `chrY`; for XY samples with haplotype-specific CN, sets minor CN of `chrX` and `chrY` to 0.
- Neighbor merging: merges adjacent segments with identical CNs to reduce fragmentation.
- Telomeres and centromeres: fills telomeric regions and optionally bins/unifies centromeres (can be skipped with `--pre-skip-centromeres`).
- MEDICC2 phasing: optional phasing of haplotypes; can be skipped with `--pre-skip-phasing`.
- Short arms and bounds: handles short arms and aligns segment ends to reference chromosome lengths.

Run control:
- Use `--run-preprocessing` to enable this step (default is to skip and proceed directly to `split`).

### 3.5 Parallel Processing

Use multiple cores for event inference:
```bash
# Use 8 cores
spice event_inference --config <path/to/config> --cores 8
```
While using multiple cores can technically make execution faster (especially in the case when spice takes a long time for single runs), it can also slow down execution when there are many entries to loop over.
We usually recommend to only use multiple cores for the `large_chroms` pipeline step as it takes the longest per sample.

Note that parallel processing will disable logging for the different subprocesses.

### 3.6 Logging Output

Control where logging output is sent with the `--log` flag:

* `--log terminal` (default): Writes logs to terminal only
* `--log file`: Writes logs to file only
* `--log both`: Writes logs to both terminal and file

When using `--log file` or `--log both`, logs are saved to the configured log directory from the config with a filename pattern: `{name}_{timestamp}.log`

---

## 4. Loci Detection

Loci detection identifies recurrently gained or lost copy-number loci across a cohort of samples.

**NOTE that SPICE requires a large cohort for de-novo loci calling and it will likely not produce good results for cohorts with less than 1000 samples**

### 4.1 Pipeline Overview

Prepare chromosome-level signals and kernels from inferred events, detect and
optimize loci across length scales, then combine chromosomes. Score loci against
the permutation null, apply the configured filters, and refit the
retained loci for reporting.

### 4.2 Expected Input

Loci detection requires:

- **Static segmentation grids**: set `input_files.segmentations` to a directory
  containing `hg19/` and/or `hg38/`, each with a `manifest.json` and
  `segmentation_<size>.tsv` tables (`chrom`, `start`, `end`; zero-based inclusive).
  The pipeline repository tracks these under `data/segmentations/` and supplies
  `src/data/make_segmentations.py` for explicit offline preparation. SPICE chooses
  `params.assembly`, validates file checksums and chromosome lengths, and caches
  parsed grids in memory. It never creates missing grids during analysis.
- **Event inference results**: `final_events.tsv` produced by the event-inference pipeline.
- **Observed centromere and telomere tables**: set `input_files.centromeres_observed` and
  `input_files.telomeres_observed` to tables derived from that same cohort. These are
  cohort measurements, not interchangeable assembly reference files. Use the same pair
  for observed detection, permutations, pooling, assignment, and loci plotting.
  Generate them with `data_loaders.create_observed_centromeres_and_telomeres(final_events_df)`
  after loading the cohort's loci config. The generator applies the loci event filters,
  including static centromere classification, the 5 Mb padding exclusion, width limits,
  duplicates and configured plateau filtering. Observed-centromere refinement is skipped
  during construction to avoid a circular dependency. Each length scale uses its own
  filtered events (`lower < width <= upper`, matching detection), with centromere
  coordinates rounded to that scale's segment size. Missing arms use static centromere
  boundaries; an entirely empty scale uses static assembly bounds with a warning.

For example, add these paths to your `cohort_loci.yaml` (relative paths use `directories.base_dir`):

```yaml
input_files:
  final_events: data/final_events.tsv
  segmentations: data/segmentations
  centromeres_observed: data/centromeres_observed.tsv
  telomeres_observed: data/telomeres_observed.tsv
```

The default permutation mode, `rotate`, applies a shared circular offset to internal
events within each sample/chromosome/arm. It chooses a cut uniformly from integer
positions in gaps or at event boundaries, preserving widths, overlaps, and circular
spacing without splitting events. Dense groups have fewer legal cuts; an arm-spanning
event prevents its group from moving. Events outside the observed arm bounds remain fixed.

The fitness statistic is A+B: mean positive same-direction fitness plus mean
negative opposite-direction magnitude across the four scales. `zpool_chrom`
standardizes within chromosome/direction using null-only mean and sample standard
deviation, pools standardized null draws, and uses an inclusive upper tail with
add-one correction. BH adjustment covers all observed loci in the cohort/seed.
Missing null strata score p=1; zero variance uses a denominator of one.

To combine previously detected chromosomes, use
`spice loci_detection --config <config> --loci-steps combine`. If no pooled permutation
null exists, SPICE builds one using the complete `loci_detection.loci_steps` recipe
from the config (`fast`, `full`, or a complete stage list). Keep that recipe in the
config and select combine-only or resume stages on the command line. `--overwrite`
also rebuilds an existing null; for large cohorts, build it with scattered `spice permute`
commands before combining.

### 4.3 Expected Output

Results are saved in `results/{name}`

**Main outputs:**

- `final_loci_detection.tsv`: Filtered loci with coordinates, scores and refitted fitness
- `final_loci_detection_unfiltered.tsv`: All scored candidates, used as component inputs
- `within_ci_detection.tsv`: Per-chromosome fit coverage within bootstrap bounds

Intermediate models and fits are saved in `results/{name}/loci_of_selection/`

---

## 5. Loci Assignment

Loci assignment assigns predetermined loci to a cohort. This is recommended for smaller cohorts where de-novo loci detection is prohibited.

### 5.1 Pipeline Overview

Prepare cohort signals from inferred events and optimize fitness at the supplied
reference locus positions. Combine the chromosome fits into a cohort-level locus
table.

### 5.2 Expected Input

Loci assignment requires:
- **Event inference results**: `final_events.tsv` produced by the event_inference pipeline
- **Reference loci**: defaults to `spice/reference_loci/all_460_loci.tsv`, the reference loci set created on TCGA data

### 5.3 Expected Output

Results are saved in `results/{name}/`

**Main outputs:**

- `final_loci_assignment.tsv`: Reference loci with fitted cohort-level fitness and locus statistics

---

## 6. Components

Components group recurrent loci across independent detection seeds and estimate
joint fitness for both the complete and selected component sets.

```bash
spice components --config configs/components_example.yaml
```

### 6.1 Pipeline Overview

1. Load scored, unfiltered locus tables from independent detections of the same cohort.
2. Cluster loci on the same chromosome and in the same direction, allowing one
   member per seed. Default maximum member-center spans are 1, 2, 4 and 8 Mb
   for the small, mid1, mid2 and large scales. Each group obeys its tightest
   member's span limit; scale assignment uses positive same-direction fitness.
3. Select components by seed support and equal-weight Stouffer combination of
   member q-values. Defaults require all ten seeds and a score below 0.05.
4. Prepare fresh kernels, boundary corrections, signals and bootstrap bounds
   from the cohort events after clustering. No reference detection seed is needed.
5. Jointly optimize component fitness at fixed positions, separately for the all
   and filtered sets. All eight fitness tracks can change, including initial
   zeros; membership, intervals and selection scores remain fixed.

The Stouffer score is a descriptive selection score, not a calibrated component
p-value or FDR. Component intervals enclose member positional uncertainty; they
do not measure signal extent. Candidates without positive same-direction fitness
are excluded from grouping and recorded separately.

### 6.2 Expected Input

Components require:

- **Seed locus tables**: `input_files.component_loci` maps integer seed IDs to
  `final_loci_detection_unfiltered.tsv` files. The number of tables must equal
  `components.n_seeds`. Use scored A+B tables from the same cohort and the same
  permutation null, with all eight signed fitness columns and p/q values.
- **Cohort inputs**: inferred `input_files.final_events` (an absolute path),
  cohort-specific observed centromere/telomere tables, matching plateau inputs
  and static segmentation grids, as described in section 4.2. Use the same
  assembly and event-preprocessing settings as the individual detections.
- **Component settings**: seed count, support and score thresholds, clustering
  spans and fitting budget in the YAML configuration.

The [example configuration](configs/components_example.yaml) supplies the input
layout. The component defaults are:

```yaml
components:
  n_seeds: 10
  model_seed: 0
  N_bootstrap: 1000
  N_kernel: 1000
  selection:
    method: stouffer_q
    min_support: 10
    threshold: 0.05
  max_member_spans_mb: {small: 1, mid1: 2, mid2: 4, large: 8}
  scale_fitness_fraction: 0.25
  refit_iterations: 100000
```

`components.model_seed` controls fresh model preparation; `params.seed` or
`--seed` controls fitness optimization. Omit `input_files.component_model_dir`
and `input_files.cohort_model_dir` to build the fresh component model. Detection
seeds run independently before this command; it consumes their completed tables.

### 6.3 Expected Output

Results are saved in `directories.results_dir/<name>/components/`.

**Main outputs:**

- `components_all.tsv` and `components_filtered.tsv`: Separately fitted component
  catalogs with coordinates, fitness, A and A+B summaries, member/support counts
  and `stouffer_q_score`. Component p/q fields are empty.
- `component_members.tsv` and `discarded_loci.tsv`: Seed and row identities for
  component members and excluded candidates.
- `component_model/`: Fresh models, bootstrap draws and their manifest.
- `fits/{all,filtered}/<chrom>/`: Fitted signals and selection-point pickles.
- `config.yaml` and `audit.json`: Effective settings, input hashes, model identity,
  fit losses and within-CI fractions, calculated on non-centromere bins using
  strict bounds and including zero-signal bins.

Use a new name/results directory for each run. `--chrom chr21` restricts the job
to one chromosome; separate chromosome jobs require distinct output names and
component IDs must be made unique when merging their catalogs.

---

## 7. Plotting

Plotting generates visualizations of inferred events and detected loci to aid in manual inspection and interpretation of results.

### 7.1 Event Visualization

Plotting inferred events can be done on the sample or ID (sample, chromosome, allele) level.

```bash
# Plot inferred events per sample
spice plotting --config <path/to/config> --plot-events-per-sample <SAMPLE_ID>
spice plotting --config <path/to/config> --plot-events-per-sample <SAMPLE_ID> --plot-unit-size

# Plot per ID (format: sample:chr:cn_a|cn_b)
spice plotting --config <path/to/config> --plot-events-per-id <sample:chr:allele>
```

**Requirements:**
- Event plotting uses the event-inference config and `final_events.tsv`; cohort observed-centromere
  and observed-telomere tables are not required.
- Output PNGs are saved to `plot_dir/{name}/` (see `directories.plot_dir` in config; defaults to `plots/`).
- `--plot-unit-size` switches per-sample plots to unit-size segments.

For interactive exploration, see `notebooks/events_plotting.ipynb`.

### 7.2 Loci Visualization

Plotting detected or assigned loci can be done on the chromosome or loci level.

```bash
# Plot detected/assigned loci for chromosome 1
spice plotting --config <path/to/config> --plot-loci-on-chrom chr1 --loci-mode detection
spice plotting --config <path/to/config> --plot-loci-on-chrom chr1 --loci-mode assignment

# Plot the detected locus "3" (corresponds to the index in the final_loci_detection.tsv file)
spice plotting --config <path/to/config> --plot-single-locus 3 --loci-mode detection
```

**Requirements:**
- Plotting requires `final_loci_detection.tsv` or `final_loci_assignment.tsv`.
- Use the same `input_files.centromeres_observed` and `input_files.telomeres_observed`
  tables used for detection or assignment.
- Output PNGs are saved to `plot_dir/{name}/` (see `directories.plot_dir` in config; defaults to `plots/`).

Detection plots use the saved combined fit (`detection/final_loci_detection_filtered.pickle`)
when available, matching the final table and within-CI scores. Before combination,
they use the original chromosome fit. An empty combined fit remains empty in the plot.
Single-locus plots translate original ranks to positions among the surviving peaks.

Combination only reoptimizes chromosomes where q-value or mean-fitness filtering
removed peaks. If every peak is retained, the original fitness is kept. P/q values
remain the scores calculated before this optional refit.

`loci_detection.post_filter_refit_method` selects `joint` (default) or
`neighborhood`. Both keep positions fixed and preserve zero-fitness constraints.
`final_reoptimization_N_iterations` is a total budget per changed chromosome
model for `joint`, or per neighborhood for `neighborhood`.
Joint refitting keeps the starting fit if the returned fit does not improve its
native loss. The native annealer still returns its last accepted state; this
option does not add best-ever tracking or alter detection/null generation.

For interactive exploration, see `notebooks/loci_plotting.ipynb`.

---

## 8. Python API

You can also import and use SPICE functions directly in Python. Note that it is important to run `spice.load_config(config_file)` before any other spice imports
```python
# First import spice and set the config location
config_file = 'configs/events_example.yaml'
import spice
spice.load_config(config_file);

# Then perform any other spice imports
from spice.data_loaders import load_chrom_lengths
...
```

See also the example notebooks for how to use the API.

## 9. Known issues

**SPICE event inference runs for too long / doesn't finish:** This is usually due to the MCMC event inference for large chromosomes (>9 events). Either reduce the paramter `mcmc_n_iterations_scale` which will reduce the total number of iterations to run or set the parameter `time_limit_mcmc` to a time limit (in seconds) which will abort the computation. Note that in the case of `time_limit_mcmc`, no output will be saved.

**Long computation time for single-cell data:** SPICE treats every sample/chromsome pair separately. For single-cell datasets this results in a massive amount of individual calculations. We recommend to first remove duplicate sample/chromosomes and then run SPICE on this reduced dataset.

## 10. Citation

If you use SPICE in your research, please cite the [accompanying BioRxiv preprint](https://www.biorxiv.org/content/10.64898/2026.03.01.708809v1):

> **Deciphering selection patterns of somatic copy-number events** Tom L. Kaufmann, Adam Streck, Florian Markowetz, Peter Van Loo, Roland F. Schwarz. bioRxiv 2026; doi: https://doi.org/10.64898/2026.03.01.708809

## 11. License

GNU GENERAL PUBLIC LICENSE

## 12. Contact

For questions and issues, please contact tom.kaufmann@iccb-cologne.org or roland.schwarz@iccb-cologne.org.
