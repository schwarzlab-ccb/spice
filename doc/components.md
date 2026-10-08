# Components from independent detection seeds

The standard workflow is: infer cohort events once, run independent peak-detection
seeds from those events, cluster the resulting scored peak tables, then build a
fresh event-derived model and jointly fit component fitness. No detection seed
supplies the component model, and seeds do not share preprocessing or fitted peaks.

```bash
spice components --config configs/components_example.yaml
```

The command uses the normal SPICE YAML loader and shared `--seed`, `--log` and
`--debug` options. It groups saved unfiltered, scored locus tables, selects
components, prepares their model, and fits the all/filtered sets separately.
It does not run event inference, schedule seed detections, regenerate the null,
recompute member p/q values or build a browser.

## Configuration

[The example YAML](../configs/components_example.yaml) supplies the cohort inputs,
seed tables and these defaults:

```yaml
params:
  seed: 9 # Component optimization RNG; --seed overrides this
components:
  n_seeds: 10
  model_seed: 0 # Fresh model preparation RNG; independent of detection seed IDs
  N_bootstrap: 1000 # Component signal bounds
  N_kernel: 1000 # Component convolution kernels and boundary corrections
  selection:
    method: stouffer_q
    min_support: 10
    threshold: 0.05
  max_member_spans_mb: {small: 1, mid1: 2, mid2: 4, large: 8}
  scale_fitness_fraction: 0.25
  refit_iterations: 100000
```

Supply inferred `input_files.final_events`, cohort-specific
`centromeres_observed` and `telomeres_observed`, the matching `plateaus` input and
assembly. Use the same event-preprocessing settings as the independent detections.
The event loader expects an absolute `final_events` path. Other input paths may
be absolute or resolve under `directories.base_dir`.

`input_files.component_loci` maps integer seed IDs to indexed, unfiltered native
TSVs. Its entry count must equal `n_seeds`; seed IDs need not be consecutive.
Each table needs `chrom`, `type`, `pos`, `start`, `end`, `p_value`, `q_value`,
all eight signed `fitness_<scale>_<gain|loss>` columns, and `p_values_method` set
to `combined_fitness`. Supply repeated detections of the same cohort using the
same null. Each independent detection may use its own kernels and bootstrap draws.

New detection/assignment tables record `cohort_id` when explicit event/bound
inputs are configured. This identifies event/bound hashes, assembly and
preprocessing, without depending on model/detection seeds. Components reject
mismatching or partly tagged cohorts. Historical untagged tables are accepted
with `cohort_validation: legacy_untagged`; their original frozen configs and
provenance must establish cohort compatibility externally. Table/scoring checks
and input hashes do not establish null compatibility on their own.

`reference_seed` is unnecessary and has been removed from defaults. An old
configuration containing that key is accepted, but the key has no effect.

## Grouping and selection

Grouping matches chromosome/direction and allows one member per seed. It assigns
each locus to its shortest positive same-direction scale reaching
`scale_fitness_fraction` times its strongest scale. Each group obeys its tightest
member's span limit; there is no extra centroid-radius constraint. Membership and
tie breaking do not use p/q values. Candidates with no positive directional
fitness are listed in `discarded_loci.tsv`.

All groups, including singletons, remain in the all-component catalog. The
filtered set requires at least `selection.min_support` distinct seeds and a score
strictly below `selection.threshold`. `stouffer_q` is equal-weight Stouffer
combination of member q-values clipped to `[1e-15, 1-1e-15]`. There is no further
BH correction or missing-seed imputation. This is a descriptive selection score,
not a calibrated component p-value or FDR.

Seed count and support threshold are separate settings: for example, five input
seeds require `n_seeds: 5` and `min_support` between 2 and 5.

## Model preparation and fitting

After grouping, SPICE creates one new model per chromosome from the cohort events.
Each gain/loss scale gets an event-length-derived kernel, boundary corrections,
observed signal, centromere mask and loss normalization. `N_bootstrap` supplies
signal bounds. Kernel preparation uses the configured component model seed and
counts; it never reads seed model caches or component fitness. This is a fresh
model and can differ from every detection model. Component positional intervals
are uncertainty envelopes, not event lengths or kernel widths.

Both component sets use that new model, each with its own fixed-position joint
optimization. Initial fitness is the mean signed member fitness projected onto
the component direction. All eight tracks can change, including initial zeros.
The fit minimizes normalized signal MSE outside centromeres; bootstrap bounds
are used for subsequent CI reporting. Only a finite, improved fit is retained.
The optimizer resets to `params.seed` / `--seed` for each chromosome/set.

Fitting preserves positions, directions, membership, intervals and selection
scores. It does not drop components that finish with zero fitness. Changing only
the component model therefore preserves catalog counts; report the number with
nonzero fitted fitness separately when relevant. No individual-detection or null
rerun is needed solely to refit components.

## Outputs and chromosome jobs

Results are written under `directories.results_dir/<name>/components/`:

- `components_all.tsv` and `components_filtered.tsv`: separately fitted fitness,
  A and A+B summaries, centers/intervals, peak/support counts and descriptive
  `stouffer_q_score`. Component `combined_p` and `q_value` are empty.
- `component_members.tsv` and `discarded_loci.tsv`: original seed/row identities.
- `component_model/`: newly prepared models, bootstrap draws and a manifest
  recording cohort/model identities, settings, event/source/file hashes.
- `fits/{all,filtered}/<chrom>/`: selection-point pickles and fitted signal arrays.
- `config.yaml` and `audit.json`: effective configuration, input hashes, model
  source, grouping, fit losses and within-CI fractions. CI uses strict bounds,
  non-centromere bins and includes zero-signal bins. `status: complete` marks success.

Use a new name/results directory for every run. `--chrom chr21` restricts grouping
and fitting to one chromosome while preserving original seed-row IDs. Separate
chromosome jobs need distinct output names. Their component IDs are local to each
job; merge with unique IDs or map by member identities before publication.
The external production pipeline/lock has not yet migrated to this native command.

## Saved models and the earlier shared-model experiment

For a controlled comparison or historical reproduction,
`input_files.component_model_dir` may point to a prepared directory containing
`data_per_length_scale/<chrom>.pickle`. This bypasses fresh model preparation;
existing bounds are consumed unchanged. Models are trusted SPICE pickle files,
and must match the configured cohort and assembly.

The earlier `input_files.cohort_model_dir` mode remains available for reproducing
[the shared-detection experiment](shared_cohort_pilot.md). That optional mode
requires matching model identities in seed tables. It is **not** the independent
workflow described here. Omit both model-directory inputs for the normal fresh
component-model workflow, and omit `cohort_model_dir` from independent detection
and null configs.
