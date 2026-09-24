# Independent length-scale detection

Set this in the YAML used by both observed detection and `spice permute`:

```yaml
loci_detection:
  detection_scale_mode: independent
```

The default remains `joint`. Use a fresh results directory and generate a matching
permutation null when switching modes. The permutation placement rule is a separate
setting (`p_values_permute_mode`); this option does not change that rule.

Independent mode runs the existing full or fast detection cascade four times, once
for small, mid1, mid2 and large events. Gain and loss within each scale share locus
locations, but have separate fitness values with the usual OG/TSG sign constraints.
Locations, candidate budgets, residual searches, optimization loss, CI filtering,
merging, prominence filtering and width inference are independent across scales.
The common chromosome search boundaries remain unchanged.

`N_loci` and `N_loci_spacing` apply **per scale**, so the maximum total candidate
budget can be four times the joint-mode budget. This is not a promise about retained
peak counts or runtime. Bootstrap observations and convolution kernels are prepared
once per chromosome and shared across the four models. In this mode their random
streams are keyed by track; fitting and width-inference streams are keyed by scale.

Each final table row has `length_scale` and `detection_scale_mode`. Its tested
statistic is the positive directional fitness in its owning scale (OG: gain;
TSG: loss). It is calibrated against null loci detected in that same scale, using
the configured chromosome/direction pooling strategy. BH correction uses one family
containing all tested loci across chromosomes and scales. Per-scale p/q columns
contain values only for the owning scale, with other scales left missing.

Filtering triggers a fixed-location fitness refit only for a chromosome/scale
where loci were removed and at least one remains. A scale retaining all its loci
keeps its existing fit. Reported p/q values remain the pre-refit selection statistics;
fitness, final curves and CI-match scores use the final fit.

Stage caches are in `detection/<chrom>/<scale>/`. The chromosome directory also
contains an eight-track combined model and `final_locus_scales.pickle`, aligned to
its original locus ranks. In this representation other-scale fitness entries are
zero; the eight-track simulator and existing final plotting/CI routines can read it.
`scale_mode.json` prevents mixing independent and joint caches, including unmarked
legacy fits. A partial stage run invalidates the combined model until all four
scales complete their final widths.

This mode applies to de novo `loci_detection`. Reference-based `loci_assignment`
continues to require joint mode. Clustering and browser workflow changes are
separate from this detection option.
