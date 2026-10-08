# Shared cohort model

`spice cohort_model --config cohort.yaml` prepares the stochastic preprocessing
once, before detection. Every detection seed and the component refit can then use
identical observed signals, kernels, chromosome-boundary corrections, masks and
loss normalization. There is no reference detection seed in this mode.

Start from [the shared model example](../configs/cohort_model_example.yaml):

```yaml
cohort_model:
  seed: 0
  N_bootstrap_report: 1000
loci_detection:
  N_bootstrap: 100
  N_kernel: 1000
input_files:
  cohort_model_dir: results/cohort/cohort_model
```

`cohort_model.seed` controls preprocessing only. `params.seed` / `--seed` still
controls detection or component optimization. The builder rejects `--seed` to
avoid confusing these roles. A different model seed defines a different model;
it is not selected from the detection seeds or their results.

## Execution

1. Supply inferred `final_events`, cohort-specific `centromeres_observed` and
   `telomeres_observed`, and the matching assembly/plateau configuration. Load the
   configuration before importing SPICE's model modules. Use absolute paths for
   `final_events` (the existing event loader expects them).
2. Run `spice cohort_model --config cohort.yaml`. The builder uses its `name` and
   `directories.results_dir` to write `results/<name>/cohort_model/`. Existing
   output is refused. `--chrom chr21` builds a chromosome-only pilot artifact;
   it cannot serve other chromosomes.
3. Point `input_files.cohort_model_dir` in **every observed, null and component
   config** to that artifact. Give each detection seed a distinct output `name`
   and `params.seed`. Run the usual `spice loci_detection` command. Shared model
   paths resolve against `directories.base_dir`, or may be absolute.
4. Build a fresh null with `spice permute` under a distinct name (typically
   `params.seed: 0`, `p_values_K: 16`). Use the same cohort/model and detection
   settings. Scatter/pool works as before. Copy its `permutation_null.tsv` into
   each observed run's `loci_of_selection/` before running
   `spice loci_detection --loci-steps combine`. A full, unscattered detection
   command can also build its null inline, as before.
5. Supply the scored seed tables in `input_files.component_loci` and run
   `spice components`. Keep `cohort_model_dir`; **omit `component_model_dir`**.
   Seed count, support threshold, Stouffer selection and fit iterations use the
   existing `components` configuration. `reference_seed` is ignored and the
   output audit records it as null. Initial fitness still equally averages members.

The native commands do not schedule all seeds. The external pipeline's production
lock and orchestration have not been migrated to this experimental branch.

## Bootstrap separation and outputs

The artifact contains:

- `data_per_length_scale/<chrom>.pickle`: observed detection model with
  `loci_detection.N_bootstrap` signal bounds.
- `signal_bootstrap/<chrom>_N_<count>.pickle`: detection and reporting bootstrap
  draws, including the shared detection draws used for positional inference.
- `report/data_per_length_scale/<chrom>.pickle`: the same model with only
  `signal_bounds` replaced by `N_bootstrap_report` bounds. Kernels, normalization,
  signals and masks are unchanged. Component fitting uses this version.
- `config.yaml` and `manifest.json`: effective configuration, processed-event
  hashes, source/input/asset hashes, file hashes and model identity.

Detection remains at 100 signal/100 width samples when configured that way.
Component signal CI uses 1000 samples by default. This does not execute the
separate 1000-sample **individual positional report**.

The model is published only when preparation succeeds. Detection copies its
required model/bootstrap files into its normal cache paths so existing width,
combination and plotting routines can consume them. Copies cannot overwrite the
shared source. Model identifiers are recorded per chromosome and in locus tables,
null tables and component audits. Different model identities or unidentified old
caches are rejected; use a fresh output directory. `overwrite_preprocessing` is
not supported for shared caches: rebuild the model under a new name instead.

## Null behavior

A permutation gets its **own signals and bounds from the permuted events** and
its own simulated kernels/corrections, using the fixed model-preparation seed
and the same counts/policy. It never copies the observed signals. Each permutation
cache records its processed-event hash and role. Rotation and detection still use
the null run's normal RNG; model preparation restores that RNG afterward.

The shared model changes the detection procedure, so a null from the previous
seed-specific procedure must be regenerated. A+B scoring, within-arm rotation,
chromosome/direction standardization and BH are unchanged. Pooling and scoring
reject mixed/missing model identities. Model tags verify model compatibility;
they do not certify arbitrary external TSVs or every detection/selection setting.
Keep complete run configurations and provenance.

Historical `component_model_dir` and per-seed preprocessing remain available for
reproducing previous outputs when no `cohort_model_dir` is configured. Historical
results are not relabeled as shared-model results.
