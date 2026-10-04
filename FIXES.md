# Production fixes

This branch starts at upstream main `ef416ce`. The pipeline production selection
is `rotate` + directional mean fitness + `zpool_chrom`, with genome-wide BH.
Main already implements all three; the scoring formula is unchanged here.

Selected changes:

- `58406aa`, adapted as `cd69a60`: joint fixed-position refitting only on
  chromosomes where filtering removes peaks. One total iteration budget per
  chromosome, with locked zero fitness, preserved direction, and no loss increase.
  The original neighborhood optimizer remains an explicit compatibility option.
- `ecafae0` plus rollback `9ab547c`, combined as `f49a219`: correct the clipped
  chromosome-edge argmax offset; keep the fixed +/-10-bin prominence fallback.
- Cache/config/table checks adapted from `80cbcd5`, `d4dbbf4`, and `1585164`:
  reject independent-scale and bridge-conditioned artifacts and unsupported
  settings without carrying their experimental algorithms.
- Self-contained joint optimizer and cache-determinism regression fixtures.
- The within-CI export uses only chromosomes in the combined fit (selected from
  `6973995`), avoiding attempts to read chrX in autosome-only synthetic cohorts.

Independent detection, seed components, component maxT/spans, any-scale p-values,
unconditional final refits, and bridge-preserving permutations/background are not
included. Their implementation and feature inventory remain on `anyscale-pvalue`:
`git show anyscale-pvalue:BRANCH_NOTES.md` (notes commit `1736a3c`).

Native defaults remain main's unless listed above. The pipeline explicitly sets
mean fitness, zpool_chrom, rotate, seed42/null0, K16, bootstrap100 for both signal
and width, full pruning, 1 Mb candidate spacing, no absolute internal fitness floor,
and 100,000 joint refit iterations. A keep-all synthetic run retains its original
fitness after scoring; final refitting no longer runs merely because scoring ran.

The pipeline separately computes 1,000 final-report positional bootstrap samples
around the saved post-filter fit. Its top-level reporting setting does not enter
native detection or null generation. Existing calibrated 100-sample nulls remain
valid. Native YAML defaults remain main's signal100/width10; the pipeline keeps its
explicit detection100/width100 override. This supersedes the earlier shared1000
configuration change, which the user clarified should affect only final reporting.

Validation on 2026-10-04: 160 native tests pass, including real joint optimization,
permutation/scoring, cold/partial/warm caches, compatibility and edge prominence.
Saved rotation neutral (215 loci) and selection (242 loci) mean-fitness p/q values
reproduce at rtol 1e-12, with independent BH verification and unchanged input hashes.
No full cohort detection rerun was launched as part of switching branches.
