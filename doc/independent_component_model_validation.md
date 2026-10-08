# Independent component model validation — TCGA, 2026-10-08

Native SPICE branch `components`, implementation commit `c59c18a`.
This validates fresh component models **after independent seed detection**.
It supersedes the shared-detection experiment as the intended workflow.
The production `fixes` branch and pipeline lock remain unchanged.

## Design

The experiment reuses all ten recovered independent TCGA detections and their
original genome-wide A+B/zpool_chrom p/q values. It reruns grouping and selection,
then builds fresh kernels, boundary corrections and signal bounds from the
original cohort events for component fitting only. Detection seeds have no shared
model and no knowledge of each other before clustering. No reference detection
seed supplies the new component model.

The comparison covers chr1–22 and chrX (hg19), 213814 processed events and 5806
samples. Fresh preparation uses component model seed 0, 1000 kernel samples and
1000 signal bootstrap draws. Both all and selected component sets get separate
100000-step, fixed-position joint fits with optimizer seed 9. All eight fitness
tracks are eligible to change, including initial zeros. Selection remains ten
supports and Stouffer member-q score <0.05, using original 1/2/4/8 Mb spans.

The previous fits use the recovered seed-9 kernels and 1000-draw report bounds.
The fresh model changes both kernel preparation and bootstrap draws. To separate
those effects, both old and new fitted curves are evaluated on both sets of
bounds, without changing the curves during that evaluation.

## Results

| Set | Previous count | Fresh count | Previous CI | Fresh CI | CI change | Fresh fit on previous bounds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| All components | 1289 | 1289 | 91.939% | 91.957% | +0.017 pp | 91.922% |
| Selected components | 422 | 422 | 80.282% | 80.408% | +0.125 pp | 80.367% |

Counts, exact seed/row memberships, positions, intervals, support, selection scores
and initial component fitness are unchanged. All components retain nonzero A+B
fitness in both versions; none disappear through zero fitness. The individual
seed peak tables are byte-for-byte unchanged (8564 unfiltered candidates in total).
This is a component refit comparison, not a new peak-detection run.

CI fractions use strict bounds on non-centromere bins, including zero-signal bins.
Genome-wide aggregation averages the eight tracks per chromosome, then weights
chromosomes by length, matching the native convention. Own-bound columns compare
each fitted curve with its corresponding bootstrap bounds. The last column holds
the previous bounds fixed and changes only the kernel/fitness-generated curve.
Selection scores are descriptive Stouffer combinations, not component p-values.

## Validation and retained evidence

- Native tests: 240 passed, 4 optional integration tests skipped.
- Every chromosome completed native fitting and post-fit validation successfully.
- Exact membership matching recovers original global component IDs.
- Observed signals, centromere masks and loss normalization are identical between
  old and fresh models; geometry, selection scores and initial fitness agree.
- Reconstructed fresh curves agree with native saved fitted signals to 1e-10.
- A saved-model native chr21 control reproduces the previous all/selected fitted
  fitness exactly, confirming the baseline comparison.
- Frozen source, seed table and raw input hashes remain unchanged.

The full experiment is retained in the pipeline workspace under
`scratch/independent_component_model/`:

- `components_all.tsv`, `components_filtered.tsv`: fresh fitted output tables.
- `summary.tsv`, `ci_per_chromosome.tsv`, `ci_tracks.tsv`: numeric comparisons.
- `ci_comparison.png`, `ci_change.png`: absolute coverage and changes by chromosome.
- `fitness_changes.tsv`: per-component fitness changes.
- `validation.json`, `raw_input_validation.json`: checks and final summary.
- `configs/`, `source/`, `source_sha256.json`, `experiment.json`,
  `implementation.json`: frozen settings, code and input identities.
- `results/`: model/bootstrap artifacts, both component fits and comparison curves.
- `logs/`, `jobs.json`, `slurm_accounting.tsv`, `full_tests.log`: execution evidence.

Historical seed tables have no cohort tag; their lineage was verified separately
during backup recovery, whose receipt hash is recorded in `experiment.json`.
Event inference, independent detections and null calibration were not rerun.
This comparison does not establish end-to-end inference reproducibility or
performance across cohorts or multiple component model seeds. Production pipeline
migration remains separate from this native command implementation.

See [component configuration and usage](components.md).
