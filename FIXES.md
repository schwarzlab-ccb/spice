# Production fixes (updated 2026-10-08)

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
historical branches, outside this production implementation. Native component
grouping and Stouffer member-q selection were integrated on 2026-10-08, as below.

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

## Native components integrated on 2026-10-08

The `components` implementation through `49be88f` was fast-forwarded into `fixes`
without changing its validated scientific code. Run:

```bash
spice components --config configs/components_example.yaml
```

The default workflow keeps peak-detection seeds independent. It groups their
scored tables, selects components, prepares fresh kernels/corrections and signal
bounds from the cohort events, then jointly fits the all and filtered sets
separately. No reference detection seed is needed. YAML defaults use ten seeds,
ten supports, Stouffer member-q score <0.05, original 1/2/4/8 Mb spans, model seed
0, 1000 kernel/bootstrap samples and 100000 component optimization steps.

The optional `cohort_model` command from the earlier shared-detection experiment
is retained as implemented, but is not part of this default workflow. Independent
detection/null configurations omit `cohort_model_dir`; standard component configs
omit both model-directory inputs. Explicit saved-model inputs remain useful for
historical reproduction and controlled comparisons.

Whole-genome TCGA validation preserved exact membership and all 1289 / selected
422 component counts, with no zero-fitness components. Selected CI coverage changed
from 80.282% to 80.408%; on unchanged old bounds it became 80.367%. Native tests:
240 passed, 4 skipped. See [current component usage](README.md#6-components).

This integrates the command into native SPICE only. The external pipeline's
workflow and production lock still target their previous implementation and need
a separate migration; its locked preparers reject this newer native revision.
No existing results were replaced and nothing was pushed.

## Static segmentation references

Analysis now reads explicit assembly-specific TSV grids through
`input_files.segmentations`. Bin creation moved to the pipeline repository's
standalone data-preparation script. Missing, modified or wrong-assembly grids
fail instead of silently populating a size-only cache in the results directory.
Parsed grids are cached in memory without sharing mutable frames with callers.
The production bin coordinates and terminal-bin convention are unchanged.
