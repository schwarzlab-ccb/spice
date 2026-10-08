# Components from multiple detection seeds

```bash
spice components --config configs/components_example.yaml
```

The command uses the normal SPICE YAML loader and the shared `--seed`, `--log`
and `--debug` options. It groups saved unfiltered, scored locus tables, selects
components and fits the all-component and filtered-component sets separately.
It does not schedule seed detections, generate a null, recompute member p/q values,
bootstrap signals or build a browser. Those remain separate workflow steps.

Start with [the example YAML](../configs/components_example.yaml). Its main settings are:

```yaml
params:
  seed: 9
components:
  n_seeds: 10
  reference_seed: 9
  selection:
    method: stouffer_q
    min_support: 10
    threshold: 0.05
  max_member_spans_mb: {small: 1, mid1: 2, mid2: 4, large: 8}
  scale_fitness_fraction: 0.25
  refit_iterations: 100000
```

`input_files.component_loci` maps integer seed IDs to indexed, unfiltered native
TSVs. Its entry count must equal `n_seeds`; seed IDs need not be consecutive.
Each table needs `chrom`, `type`, `pos`, `start`, `end`, `p_value`, `q_value`,
all eight `fitness_<scale>_<gain|loss>` columns, and `p_values_method` set to
`combined_fitness`. Supply repeated fits of the **same cohort using the same
null**. The command checks the table format/scoring tag and records input hashes;
the TSVs alone do not establish matching cohort/null provenance.

`input_files.component_model_dir` contains `data_per_length_scale/<chrom>.pickle`
from the declared `reference_seed`. Supply the matching assembly and cohort-specific
observed centromere/telomere tables through the usual SPICE configuration. Input
paths resolve under `directories.base_dir`, or can be absolute. Models are trusted
SPICE pickle files and must have all eight joint tracks. Their existing signal
bounds are used unchanged. To match production reporting, prepare the seed-9 model
with 1000-sample component signal bounds beforehand; changing a YAML bootstrap
count does not regenerate a saved model.

The command evaluates the chromosomes represented in the seed tables, including
chromosomes with no selected components. Completely empty seed tables cannot
establish a chromosome scope by themselves. A missing required model is an error.

## Grouping and selection

Grouping matches chromosome/direction and allows one member per seed. It assigns
each locus to its shortest positive same-direction scale reaching
`scale_fitness_fraction` times its strongest scale. Each group obeys its tightest
member's span limit; there is no extra centroid-radius constraint. Membership and
tie breaking do not use p/q values. Candidates with no positive directional
fitness are listed in `discarded_loci.tsv`.

All groups, including singletons, remain in the all-component catalog. The
filtered set requires at least `selection.min_support` distinct seeds and a score
strictly below `selection.threshold`. The currently supported selection method is
`stouffer_q`: equal-weight Stouffer combination of member q-values clipped to
`[1e-15, 1-1e-15]`. There is no further BH correction or missing-seed imputation.
This is a **descriptive selection score, not component p-value or FDR**.

Seed count and selection support are separate settings: for example, five input
seeds require `n_seeds: 5`, a reference seed present among those inputs, and
`min_support` between 2 and 5. Defaults require ten seeds and ten supports.

## Fitness fitting and outputs

Each chromosome/set gets one fixed-position joint optimization with
`refit_iterations` proposals. Initial fitness is the mean signed member fitness,
projected onto the component direction. All eight tracks can change, including
initial zeros. Positions, directions, memberships and member-interval envelopes
stay fixed. Only a finite, improved fit is kept. The RNG is reset to `params.seed`
(or `--seed`) for each chromosome/set, matching the pipeline's refit convention;
`reference_seed` identifies input data and is independent of this RNG override.

Results are written under `directories.results_dir/<name>/components/`:

- `components_all.tsv` and `components_filtered.tsv`: separately fitted fitness,
  A and A+B summaries, component centers/intervals, peak/support counts and
  `stouffer_q_score`. Component `combined_p` and `q_value` are empty.
- `component_members.tsv` and `discarded_loci.tsv`: original seed/row identities
  and membership or exclusion reasons.
- `fits/{all,filtered}/<chrom>/`: selection-point pickles and fitted signal arrays.
- `config.yaml` and `audit.json`: effective configuration, input hashes, grouping
  rules, fit losses and within-CI fractions (strict bounds, non-centromere bins,
  including zero-signal bins). `status: complete` marks a finished run.

Component `start`/`end`/`width` describe the member positional-interval envelope,
not physical signal extent. Fitting does not recalculate selection scores.
Use a new `name` or results directory for each run; existing component output
directories are never overwritten. The pipeline's browser format and production
lock have not yet been migrated to this native command.
