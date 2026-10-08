# Shared cohort model validation — TCGA chr21, 2026-10-08

Implementation: native SPICE `components` branch, commit `f74ede5`.
The production lock and the `fixes` branch were not changed.

## Scope

Recovered TCGA inferred events and cohort-specific hg19 observed bounds were used
for one chromosome (chr21). Model preparation used independent seed 0, kernel
sampling 1000, detection signal/positional bootstraps 100/100, and report signal
bounds 1000. Ten full detections used seeds 0–9. Sixteen fresh within-arm rotate
permutations produced 104 null loci. Scoring remained A+B, zpool_chrom and BH.

Native `spice components` grouped unfiltered scored candidates, used the original
1/2/4/8 Mb spans, and selected ten supports with descriptive Stouffer member-q
score <0.05. The all and filtered sets each received 100000 fixed-position joint
fitness steps, with optimizer seed 9. There was no reference detection model:
both sets used the shared report model. An identical second component run tested
repeatability.

## Results

| Individual-seed metric | Recovered seed-specific models | Shared model |
| --- | ---: | ---: |
| Mean unfiltered peak count | 8.4 | 9.1 |
| Mean pairwise Jaccard | 0.64336 | 0.66712 |
| Mean CI coverage on the common report model | 86.696% | 87.234% |

Jaccard uses maximum one-to-one matching within 1 Mb and the same direction,
averaged over all 45 seed pairs. Both sets of saved individual fitness values
were evaluated against the new shared report model for the CI comparison.
CI uses strict bounds, excludes centromere bins and includes zero-signal bins.
These comparisons use all detected loci, without q filtering.

| Component set | Components | Initial loss | Fitted loss | CI coverage |
| --- | ---: | ---: | ---: | ---: |
| All | 17 | 8.49562 | 6.10927 | 91.259% |
| Filtered | 4 | 14.43158 | 13.19688 | 83.066% |

The 17 groups contain all 91 detected loci; the four selected groups contain
40 members. The repeated run produced identical all/filtered/member/discarded
TSVs, selection-point pickle hashes, and fit loss/CI audits.

## Checks

- Native test suite: **233 passed, 4 skipped** (optional PCAWG integration data
  absent). This includes tests for model-seed isolation, RNG restoration,
  incompatible inputs/configuration/files/caches, null pooling/scoring identity,
  bootstrap/report separation, and shared-model components without a reference seed.
- Every detection used byte-identical model and detection-bootstrap copies.
- Every report-model field except signal bounds matched the detection model.
  Both 100- and 1000-draw bounds matched their saved bootstrap quantiles exactly.
- The new preparation matched the recovered seed-0 signals, kernels, boundary
  corrections, masks, normalization and bounds on all eight tracks. Seed 0 also
  reproduced its previous detected positions, intervals and fitness to numerical
  tolerance. This is a control for the unchanged detection calculations.
- The regenerated chr21 null matched the recovered chr21 null in positions,
  signed fitness and aggregate/per-scale statistics to numerical tolerance.
  This is expected here: the previous null already used preparation seed 0.
- All Slurm jobs completed successfully. The current native checkout accepted and
  verified the shared artifact's manifest/file hashes.

## Limits and retained evidence

This is a functional and repeatability pilot, not a whole-genome reproduction or
calibration study. The new p/q values use the chr21-only standardized reference
pool and chr21-only BH family; the recovered values were genome-wide. Do not
compare their filtered counts as though the calibration scopes were identical.
The small Jaccard/CI improvements on one chromosome do not establish genome-wide
performance. The full production workflow has not yet adopted the shared model.

The experiment is retained under the pipeline workspace's
`scratch/shared_cohort_v2/`: `validation.json`, `seed_summary.tsv`,
`jaccard_{previous,shared}.tsv`, `seed_coherence.png`, effective `configs/`, frozen
`source/` and hashes, logs, Slurm receipts and all result files. Component tables
are `results/components/components/components_{all,filtered}.tsv`.
`implementation.json` records the small unused-import difference between the
pre-commit runtime snapshot and the committed source. Scientific model sources
are identical. Inputs remain in the separately verified backup recovery.

See [shared model usage](cohort_model.md) for configuration and execution.
