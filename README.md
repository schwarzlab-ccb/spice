# SPICE: Selection Patterns In somatic Copy-number Events

![](doc/logo_banner.png)

**SPICE**, Selection Patterns In somatic Copy-number Events, is a framework that
1) infers discrete copy-number events from allele-specific profiles,
2) detects loci of selection in the copy-number data and 
3) can assign loci of selection to copy-number data

See the [accompanying BioRxiv preprint](https://www.biorxiv.org/content/10.64898/2026.03.01.708809v1) for more information.

## 0. Installation

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
git clone git@bitbucket.org:schwarzlab/spice.git
cd spice
```

And then install in development mode:
```bash
pip install -e .
```


### 0.4 Optional Dependencies

To use SPICE with Snakemake for parallel execution on computing clusters, install snakemake separately:
```bash
conda install bioconda::snakemake
```

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

For other parameters that can be modified, see `default_config.yaml`.

### 1.2 Relative vs absolute paths

- `directories.*` entries (e.g., `data_dir`, `results_dir`, `log_dir`) as well as input files can be given as relative or absolute paths.
   - If relative, SPICE resolves them against `directories.base_dir`.
   - If absolute, SPICE uses them as-is.


## 2. Usage Overview

SPICE has four main modes:
- **event_inference**: Infer discrete copy-number events from allele-specific profiles
- **loci_detection**: Detect recurrent copy-number loci across samples
- **loci_assignment**: Assign loci to samples based on detected loci patterns
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

# Plotting
spice plotting --config <path/to/config> --plot-events-per-sample <SAMPLE_ID>
```

For large datasets, we recommend using Snakemake mode on a computing cluster (see respective sections below).

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
- `wgd_status`: TSV with WGD status per sample (see section 1.3)
- `xy_samples`: TSV with sex status per sample (see section 1.4)
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

- Provided status via `xy_samples` file:
   - Set `input_files.xy_samples` in your config to a TSV file.
   - The file must have two columns: first column is the sample identifier (used as index), second column named `xy` with boolean values (`True`/`False`) indicating XY (male) vs XX (female).
   - Example:
      ```tsv
      sample_id	xy
      SA123	True
      SA456	False
      ```

- Inferred XY status:
   - If `input_files.xy_samples` is missing or empty, SPICE infers XY by checking if any segments exist on chromosome `chrY` for a sample.

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
- WGD resolution: loads from `wgd_status.tsv` or infers as described in section 1.3.
- Sex resolution: loads from `xy_samples.tsv` or infers by presence of `chrY`; for XY samples with haplotype-specific CN, sets minor CN of `chrX` and `chrY` to 0.
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

### 3.6 Snakemake Execution

For parallel execution on computing clusters, use the Snakemake workflow.

**Note:** Snakemake must be installed separately:
```bash
conda install bioconda::snakemake
```

**Coming soon, not fully implemented yet**


**Note:** If you get a `LockException` run `spice --config configs/events_example.yaml --unlock` to remove the lock.

### 3.7 Logging Output

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

Coming soon!

### 4.2 Expected Input

Loci detection requires:
- **Event inference results**: `final_events.tsv` produced by the event_inference pipeline

### 4.3 Expected Output

Results are saved in `results/{name}`

**Main outputs:**
- `detected_loci.tsv`: List of detected recurrent loci with coordinates and occurrence statistics
- `loci_summary.tsv`: Summary statistics for each detected locus

Intermediate files are saved in `results/{name}/events`

---

## 5. Loci Assignment

Loci assignment assigns predetermined loci to a cohort. This is recommended for smaller cohorts where de-novo loci detection is prohibited.

### 5.1 Pipeline Overview

Coming soon!

### 5.2 Expected Input

Loci assignment requires:
- **Event inference results**: `final_events.tsv` produced by the event_inference pipeline
- **Reference loci**: defaults to `spice/reference_loci/all_460_loci.tsv`, the reference loci set created on TCGA data

### 5.3 Expected Output

Results are saved in `results/{name}/`

**Main outputs:**
- `loci_assignments.tsv`: Assignment of loci to samples with presence/absence or quantitative scores
- `loci_sample_matrix.tsv`: Binary or weighted matrix of loci (rows) by samples (columns)

---

## 6. Plotting

Plotting generates visualizations of inferred events and detected loci to aid in manual inspection and interpretation of results.

### 6.1 Event Visualization

Plotting inferred events can be done on the sample or ID (sample, chromosome, allele) level.

```bash
# Plot inferred events per sample
spice plotting --config <path/to/config> --plot-events-per-sample <SAMPLE_ID>
spice plotting --config <path/to/config> --plot-events-per-sample <SAMPLE_ID> --plot-unit-size

# Plot per ID (format: sample:chr:cn_a|cn_b)
spice plotting --config <path/to/config> --plot-events-per-id <sample:chr:allele>
```

**Requirements:**
- Plotting requires `final_events.tsv`.
- Output PNGs are saved to `plot_dir/{name}/` (see `directories.plot_dir` in config; defaults to `plots/`).
- `--plot-unit-size` switches per-sample plots to unit-size segments.

For interactive exploration, see `notebooks/events_plotting.ipynb`.

### 6.2 Loci Visualization

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
- Output PNGs are saved to `plot_dir/{name}/` (see `directories.plot_dir` in config; defaults to `plots/`).

For interactive exploration, see `notebooks/loci_plotting.ipynb`.

---

## 7. Python API

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

## 8. Known issues

**SPICE event inference runs for too long / doesn't finish:** This is usually due to the MCMC event inference for large chromosomes (>9 events). Either reduce the paramter `mcmc_n_iterations_scale` which will reduce the total number of iterations to run or set the parameter `time_limit_mcmc` to a time limit (in seconds) which will abort the computation. Note that in the case of `time_limit_mcmc`, no output will be saved.

**Long computation time for single-cell data:** SPICE treats every sample/chromsome pair separately. For single-cell datasets this results in a massive amount of individual calculations. We recommend to first remove duplicate sample/chromosomes and then run SPICE on this reduced dataset.

## 9. Citation

If you use SPICE in your research, please cite the [accompanying BioRxiv preprint](https://www.biorxiv.org/content/10.64898/2026.03.01.708809v1):

> **Deciphering selection patterns of somatic copy-number events** Tom L. Kaufmann, Adam Streck, Florian Markowetz, Peter Van Loo, Roland F. Schwarz. bioRxiv 2026; doi: https://doi.org/10.64898/2026.03.01.708809

## 10. License

GNU GENERAL PUBLIC LICENSE

## 11. Contact

For questions and issues, please contact tom.kaufmann@iccb-cologne.org or roland.schwarz@iccb-cologne.org.
