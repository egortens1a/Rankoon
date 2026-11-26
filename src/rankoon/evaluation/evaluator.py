from collections.abc import Mapping, Sequence

import polars as pl

from rankoon.evaluation.metrics import (
    _validate_k,
    _validate_min_relevance,
    judged_at_k,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)


_PER_QUERY_SCHEMA = pl.Schema(
    {
        "query_id": pl.Int64,
        "num_retrieved": pl.Int64,
        "num_judgments": pl.Int64,
        "num_relevant": pl.Int64,
        "has_positive_judgments": pl.Boolean,
        "ndcg": pl.Float64,
        "recall": pl.Float64,
        "reciprocal_rank": pl.Float64,
        "judged": pl.Float64,
    }
)


def evaluate_queries(
    rankings: Mapping[int, Sequence[int]],
    judgments: Mapping[int, Mapping[int, int]],
    query_ids: Sequence[int],
    ndcg_k: int = 10,
    recall_k: int = 100,
    rr_k: int = 10,
    judged_k: int = 10,
    min_relevance: int = 1,
) -> pl.DataFrame:
    """Evaluate ranked retrieval results for individual queries."""
    for k in (ndcg_k, recall_k, rr_k, judged_k):
        _validate_k(k)
    _validate_min_relevance(min_relevance)
    if len(set(query_ids)) != len(query_ids):
        raise ValueError("query_id should be unique")

    rows = []
    for query_id in query_ids:
        ranked_ids = rankings.get(query_id, ())
        query_judgments = judgments.get(query_id, {})
        ndcg = ndcg_at_k(ranked_ids, query_judgments, ndcg_k)
        recall = recall_at_k(ranked_ids, query_judgments, recall_k, min_relevance)
        reciprocal_rank = reciprocal_rank_at_k(
            ranked_ids, query_judgments, rr_k, min_relevance
        )
        judged = judged_at_k(ranked_ids, query_judgments, judged_k)
        relevances = query_judgments.values()
        rows.append(
            {
                "query_id": query_id,
                "num_retrieved": len(ranked_ids),
                "num_judgments": len(query_judgments),
                "num_relevant": sum(rel >= min_relevance for rel in relevances),
                "has_positive_judgments": any(rel > 0 for rel in relevances),
                "ndcg": ndcg,
                "recall": recall,
                "reciprocal_rank": reciprocal_rank,
                "judged": judged,
            }
        )

    return pl.DataFrame(rows, schema=_PER_QUERY_SCHEMA).sort("query_id")


def aggregate_metrics(
    per_query: pl.DataFrame,
) -> dict[str, int | float | None]:
    """Aggregate per-query metrics using macro averaging."""
    has_positive_judgments = pl.col("has_positive_judgments")
    has_relevant = pl.col("num_relevant") > 0
    return per_query.select(
        pl.len().alias("num_queries"),
        (pl.col("num_retrieved") == 0).sum().alias("num_empty_rankings"),
        (pl.col("num_judgments") == 0).sum().alias("num_queries_without_judgments"),
        (~has_positive_judgments).sum().alias("num_queries_without_positive_judgments"),
        (~has_relevant).sum().alias("num_queries_without_relevant"),
        pl.col("ndcg").filter(has_positive_judgments).mean().alias("ndcg"),
        pl.col("recall").filter(has_relevant).mean().alias("recall"),
        pl.col("reciprocal_rank").filter(has_relevant).mean().alias("mrr"),
        pl.col("judged").mean().alias("judged"),
    ).row(0, named=True)
