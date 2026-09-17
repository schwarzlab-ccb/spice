# Test fixtures

`centromeres_observed.tsv` / `telomeres_observed.tsv` — **fixtures only, not reference data.**

The `*_observed` tables record where a *cohort's* own inferred events stop either side of the
centromere and at the telomeres. They are cohort data, so since 2026-09-07 spice ships none and
`data_loaders._require_observed` raises unless `input_files.{centromeres,telomeres}_observed` is
configured — see the docstring there for why a silent per-assembly fallback invalidates a run
rather than merely degrading it.

That leaves the tests with a problem: `spice.tsg_og.simulation` loads both tables at MODULE IMPORT,
so every test that imports anything under `spice.tsg_og` failed at collection with
`FileNotFoundError`, whatever the test was actually about.

This pair is the hg19 table spice packaged before that change, kept here purely so the import
succeeds. `conftest.py` points `spice.config['input_files']` at it. Nothing about these numbers is
right for any real cohort — that is the whole reason the packaged copy was removed — so do not
copy them into a run, and do not treat a test that depends on their exact values as meaningful.
