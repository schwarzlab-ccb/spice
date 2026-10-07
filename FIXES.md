# Production fixes (2026-10-07)

Baseline: upstream main `ef416ce`. This branch now supports only the validated
`combined_fitness` (A+B) statistic, `zpool_chrom` calibration and `rotate` null.
Native API, YAML and CLI defaults agree. Unsupported options fail explicitly.

A is the mean positive same-direction fitness across all four scales. B is the
mean magnitude of negative opposite-direction fitness across all four scales.
Observed and null loci use A+B, with paired values kept together. Each
chromosome/direction stratum uses null-only mean and sample standard deviation;
standardized null draws are pooled, the inclusive upper tail receives add-one
correction, and observed p-values receive genome-wide BH within each cohort/seed.
Per-scale diagnostic p-values retain their joint locus-by-scale BH correction.

Rotate uses one circular offset per sample/chromosome/original arm, with cuts
outside event interiors. The validated order remains rotation of raw events,
then observed preprocessing. Neither the RNG nor event placement changed.
Untagged historical rotation units remain readable; explicitly incompatible
mode tags are rejected. Pooled scoring tables must carry `combined_fitness`.
Saved signed null loci can be re-pooled; mean-fitness null statistics cannot be
used directly.

## Commit audit against main

| Commit | Decision |
| --- | --- |
| `cd69a60` | Retain joint fixed-position post-filter fitting, zero/sign locks and loss guard. Only chromosomes changed by filtering are refitted. Neighborhood optimization remains a separate fitting compatibility option, not a scoring/null mode. |
| `f49a219` | Retain chromosome-edge argmax correction and local +/-10-bin prominence fallback. Kernel-width search remains reverted. |
| `6028249` | Retain guards rejecting incompatible independent-scale caches, tables and geometry. |
| `e2f8c11` | Retain within-CI reporting restricted to chromosomes actually fitted. |
| `78cb399`, `9f33daa` | Documentation history only; superseded by the final-report-only settings below. No additional bootstrap algorithm to retain. |
| `3174683` | Retain paired A+B scoring and signed null provenance; remove its mean-only p-value compatibility branch. |

Inherited alternatives removed: mean-fitness p-value scoring; arm-based `zpool`,
raw `pooled` and `perchrom` calibration; `uniform`, `chromosome_hybrid` and
`chromosome_exclusion` permutation implementations, helpers and CLI options.
Tests for retired algorithms were replaced with rejection checks. Main's arm
fallback code is no longer present. A-only fitness summaries remain required by
detection/pruning, component scale assignment and reporting; they are not an
alternative p-value method.

Independent-scale/any-scale scoring and bridge-preserving geometry remain on
historical branches, outside this production implementation. Seed clustering and
Stouffer member-q aggregation belong to the pipeline, not native SPICE.

## Frozen pipeline settings

The pipeline explicitly uses K16, null seed0, seeds0..9 for TCGA/HMF, signal100
and detection-width100 bootstrap samples, full pruning, 1 Mb candidate spacing,
no absolute internal fitness floor and 100,000 joint refit iterations. Native
signal100/width10 defaults are unchanged; pipeline overrides stay explicit.
Separate final-report positional bootstrap uses1000 draws; component confidence
bands use1000 draws. Neither increases detection/null bootstrap to1000.

Validation: saved TCGA A+B pooled null (7007 loci), ten observed seeds and every
per-scale p-value are numerically unchanged by cleanup; rotation is identical
across30 test seeds. The pipeline separately verifies the422-component selection.
See pipeline `docs/FINAL_PRODUCTION.md` and its finalization receipts for checks.

Native suite: 199 passed, 4 skipped (optional PCAWG integration data absent).
Use the SPICE conda bin directory on PATH for subprocess CLI tests.
