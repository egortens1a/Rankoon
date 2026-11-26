import polars as pl


def build_query_groups(
    queries: pl.DataFrame,
    duplicate_pairs: pl.DataFrame,
) -> pl.DataFrame:
    """Return query IDs sorted by ID with the minimum ID of each duplicate group."""
    if "query_id" not in queries.columns:
        raise ValueError("queries: missing required column query_id")
    
    pair_columns = ("query_id_a", "query_id_b")
    missing = [column for column in pair_columns if column not in duplicate_pairs.columns]
    if missing:
        raise ValueError(f"duplicate_pairs: missing required columns: {', '.join(missing)}")

    query_ids = queries.get_column("query_id")
    if query_ids.has_nulls():
        raise ValueError("queries: null values in query_id")
    all_queries = set(query_ids)
    if len(all_queries) != len(query_ids):
        raise ValueError("queries: duplicate query_id values")

    def find(parents: dict[int, int], query_id: int) -> int:
        while parents.get(query_id, query_id) != query_id:
            query_id = parents[query_id]
        return query_id
    
    def union(parents: dict[int, int], query_id_a: int, query_id_b: int) -> None:
        query_id_a, query_id_b = find(parents, query_id_a), find(parents, query_id_b)
        parents[max(query_id_a, query_id_b)] = min(query_id_a, query_id_b)

    parents: dict[int, int] = {}
    for query_id_a, query_id_b in duplicate_pairs.select(*pair_columns).iter_rows():
        if query_id_a is None or query_id_b is None:
            raise ValueError("query_id in duplicate_pairs is null")
        if query_id_a not in all_queries or query_id_b not in all_queries:
            raise ValueError("query_id not in queries dataset")
        union(parents, query_id_a, query_id_b)

    group_ids = pl.Series(
        "group_id",
        [find(parents, query_id) for query_id in query_ids],
        dtype=query_ids.dtype,
    )
    return queries.select("query_id").with_columns(group_ids).sort("query_id")
