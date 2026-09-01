"""Regression tests for chromosomes on which detection retains no loci."""

import pandas as pd

from spice.tsg_og.detection import filter_loci, infer_loci_widths, limiting_fitness
from spice.tsg_og.loci import calculate_events_per_loci_df, create_loci_df
from spice.tsg_og.simulation import SelectionPoints


def test_final_stages_preserve_empty_loci():
    empty = [[] for _ in range(8)]

    assert filter_loci(
        cur_chrom="chr21",
        final_selection_points=empty,
        loci_widths=None,
        data_per_length_scale=None,
        final_events_df=None,
    ) == empty
    assert limiting_fitness(
        cur_chrom="chr21",
        raw_selection_points=empty,
        data_per_length_scale=None,
    ) == empty
    assert infer_loci_widths(
        cur_chrom="chr21",
        final_selection_points=empty,
    ) == []


def test_create_loci_df_skips_empty_chromosomes():
    empty = [[] for _ in range(8)]
    one_locus = [
        [SelectionPoints(loci=[[1_000_000, 1.0 if i % 2 == 0 else 0.0]])]
        for i in range(8)
    ]

    loci = create_loci_df(
        all_selection_points={"chr1": one_locus, "chr21": empty},
        all_loci_widths={
            "chr1": [[900_000, 1_000_000, 1_100_000]],
            "chr21": [],
        },
    )

    assert len(loci) == 1
    assert loci.loc[0, "chrom"] == "chr1"

    scored = calculate_events_per_loci_df(
        loci,
        all_selection_points={"chr1": one_locus, "chr21": empty},
        final_events_df=pd.DataFrame(
            [{"chrom": "chr21", "pos": "internal"}]
        ),
    )
    assert scored.loc[0, "added_events"] == 0
