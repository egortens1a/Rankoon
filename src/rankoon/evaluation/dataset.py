from pathlib import Path

import polars as pl


def load_evaluation_split(
    processed_dir: Path,
    splits_path: Path,
    split: str = "val",
) -> tuple[pl.DataFrame, dict[int, dict[int, int]]]:
    """Load queries and relevance judgments for a dataset split."""
    if split not in ('train', 'val', 'test'): raise ValueError("split schould be = 'train', 'val' or' 'test")
    
    queries = pl.read_parquet(
        processed_dir / "queries.parquet", columns=["query_id", "query"]
    )
    labels = pl.read_parquet(
        processed_dir / "labels.parquet", columns=["query_id", "product_id", "relevance"]
    )
    query_splits = pl.read_csv(
        splits_path, columns=["query_id", "group_id", "split"]
    )
    
    if queries.is_empty(): raise ValueError("queries.parquet schould not be empty")
    if any(queries.null_count().row(0)): raise ValueError("queries.parquet contains null values")
    
    if any(labels.null_count().row(0)): raise ValueError("labels.parquet contains null values")
    if not labels.get_column("relevance").is_in([0,1,2]).all(): raise ValueError("labels.parquet: relevance should be = 0, 1 or 2")
    
    if query_splits.is_empty(): raise ValueError("query_splits.csv schould not be empty")
    if query_splits.select("query_id").is_duplicated().any():
        raise ValueError("query_id from split manifest should be unique")
    if any(query_splits.null_count().row(0)): raise ValueError("query_splits.parquet contains null values")
    
    query_with_split = query_splits.filter(pl.col("split") == split).get_column("query_id").to_list()
    
    if len(query_with_split) == 0: raise ValueError(f"query_split by {split} should not be empty")
    unknown_ids = set(query_splits["query_id"]) - set(queries["query_id"])
    if unknown_ids:
        raise ValueError(f"Unknown query IDs in split manifest: {sorted(unknown_ids)}")
    
    queries = queries.filter(
        pl.col("query_id").is_in(
            query_with_split
            )
        ).sort(by="query_id")
    labels = labels.filter(
        pl.col("query_id").is_in(
                query_with_split
                )
        )
    
    if labels.select("query_id", "product_id").is_duplicated().any():
        raise ValueError("pairs query_id-product_id from labels should be unique")
    if queries["query_id"].is_duplicated().any():
        raise ValueError("queries.parquet: query_id should be unique")
    
    
    judgments = {query_id: {} for query_id in query_with_split}
    for query_id, product_id, relevance in labels.iter_rows():
        judgments[query_id][product_id] = relevance
        
    return queries, judgments
    
    
