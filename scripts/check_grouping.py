from pathlib import Path

import polars as pl

from rankoon.data.grouping import build_query_groups

ROOT = Path(__file__).parent.parent
queries = pl.read_csv(
    ROOT / "data" / "raw" / "dataset" / "query.csv",
    separator="\t",
    schema_overrides={"query_id": pl.Int64},
)
annotations = pl.read_csv(ROOT / "data" / "annots" / "candidate_pairs.csv")
duplicate_pairs = (
    annotations.filter(pl.col("is_duplicate") == 1)
    .select("query_id_a", "query_id_b")
)
grouped = build_query_groups(queries, duplicate_pairs)

assert grouped["query_id"].to_list() == sorted(queries["query_id"]), "Incomplete or unsorted result"

group_by_query = dict(grouped.iter_rows())
for first, second in duplicate_pairs.iter_rows():
    assert group_by_query[first] == group_by_query[second], f"Duplicate pair split: {first}, {second}"

minimum_ids = grouped.group_by("group_id").agg(pl.col("query_id").min())
assert (minimum_ids["group_id"] == minimum_ids["query_id"]).all(), "Expected minimum query ID per group"

print("WANDS queries:", queries.height)
print("Annotated duplicate pairs:", duplicate_pairs.height)
print("Query groups:", grouped.get_column("group_id").n_unique())
