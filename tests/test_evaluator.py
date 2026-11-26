from math import log2

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from rankoon.evaluation.evaluator import aggregate_metrics, evaluate_queries


PER_QUERY_SCHEMA = {
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


def test_evaluate_queries_matches_hand_calculation_and_sorts_results() -> None:
    rankings = {1: [102, 101, 103], 2: [201, 203, 202]}
    judgments = {1: {101: 2, 102: 1, 103: 0, 104: 2}, 2: {201: 0, 202: 1, 203: 2}}

    result = evaluate_queries(rankings, judgments, query_ids=[2, 1])

    ndcg_first = (1 + 3 / log2(3)) / (3 + 3 / log2(3) + 0.5)
    ndcg_second = (3 / log2(3) + 0.5) / (3 + 1 / log2(3))
    expected = pl.DataFrame(
        [
            (1, 3, 4, 3, True, ndcg_first, 2 / 3, 1.0, 0.3),
            (2, 3, 3, 2, True, ndcg_second, 1.0, 0.5, 0.3),
        ],
        schema=PER_QUERY_SCHEMA,
        orient="row",
    )
    assert_frame_equal(result, expected)


@pytest.mark.parametrize("rankings", [{}, {42: []}], ids=["missing", "empty"])
def test_evaluate_queries_keeps_queries_without_results(
    rankings: dict[int, list[int]],
) -> None:
    result = evaluate_queries(rankings, {42: {101: 2, 102: 0}}, query_ids=[42])

    expected = pl.DataFrame(
        [(42, 0, 2, 1, True, 0.0, 0.0, 0.0, 0.0)],
        schema=PER_QUERY_SCHEMA,
        orient="row",
    )
    assert_frame_equal(result, expected)


@pytest.mark.parametrize("judgments", [{}, {42: {}}], ids=["missing", "empty"])
def test_evaluate_queries_keeps_queries_without_judgments(
    judgments: dict[int, dict[int, int]],
) -> None:
    result = evaluate_queries({42: [101, 102]}, judgments, query_ids=[42])

    expected = pl.DataFrame(
        [(42, 2, 0, 0, False, 0.0, 0.0, 0.0, 0.0)],
        schema=PER_QUERY_SCHEMA,
        orient="row",
    )
    assert_frame_equal(result, expected)


def test_evaluate_queries_ignores_rankings_and_judgments_outside_query_ids() -> None:
    expected = evaluate_queries({1: [101]}, {1: {101: 2}}, query_ids=[1])

    result = evaluate_queries(
        {1: [101], 2: [201, 201]},
        {1: {101: 2}, 3: {301: 3}},
        query_ids=[1],
    )

    assert_frame_equal(result, expected)


def test_evaluate_queries_uses_independent_cutoffs_and_relevance_threshold() -> None:
    result = evaluate_queries(
        {7: [999, 10, 20, 30, 998, 31]},
        {7: {10: 1, 20: 2, 30: 2, 31: 2}},
        query_ids=[7],
        ndcg_k=2,
        recall_k=4,
        rr_k=1,
        judged_k=5,
        min_relevance=2,
    )

    ndcg = (1 / log2(3)) / (3 + 3 / log2(3))
    expected = pl.DataFrame(
        [(7, 6, 4, 3, True, ndcg, 2 / 3, 0.0, 3 / 5)],
        schema=PER_QUERY_SCHEMA,
        orient="row",
    )
    assert_frame_equal(result, expected)


def test_evaluate_queries_tracks_positive_judgments_separately_from_threshold() -> None:
    result = evaluate_queries({7: [10]}, {7: {10: 1}}, query_ids=[7], min_relevance=2)

    expected = pl.DataFrame(
        [(7, 1, 1, 0, True, 1.0, 0.0, 0.0, 0.1)],
        schema=PER_QUERY_SCHEMA,
        orient="row",
    )
    assert_frame_equal(result, expected)


def test_evaluate_queries_preserves_schema_for_empty_query_ids() -> None:
    result = evaluate_queries({1: [101]}, {1: {101: 2}}, query_ids=[])

    assert_frame_equal(result, pl.DataFrame(schema=PER_QUERY_SCHEMA))


@pytest.mark.parametrize("query_ids", [[], [1]], ids=["empty-queries", "missing-data"])
@pytest.mark.parametrize(
    "options",
    [
        {"ndcg_k": 0},
        {"recall_k": -1},
        {"rr_k": 1.5},
        {"judged_k": 0},
        {"min_relevance": 0},
        {"min_relevance": 3},
        {"min_relevance": 1.0},
    ],
    ids=[
        "ndcg-k",
        "recall-k",
        "rr-k",
        "judged-k",
        "zero-threshold",
        "high-threshold",
        "float-threshold",
    ],
)
def test_evaluate_queries_rejects_invalid_parameters_without_data(
    query_ids: list[int], options: dict[str, int | float]
) -> None:
    with pytest.raises(ValueError):
        evaluate_queries({}, {}, query_ids=query_ids, **options)


def test_evaluate_queries_rejects_duplicate_query_ids() -> None:
    with pytest.raises(ValueError):
        evaluate_queries({}, {}, query_ids=[1, 1])


@pytest.mark.parametrize("rankings", [{}, {1: [101]}], ids=["no-ranking", "with-ranking"])
@pytest.mark.parametrize("invalid_grade", [3, "invalid"], ids=["out-of-range", "non-integer"])
def test_evaluate_queries_rejects_invalid_judgments_for_evaluated_queries(
    rankings: dict[int, list[int]], invalid_grade: int | str
) -> None:
    with pytest.raises(ValueError):
        evaluate_queries(rankings, {1: {101: 2, 102: invalid_grade}}, query_ids=[1])


@pytest.mark.parametrize("judgments", [{}, {1: {101: 2}}], ids=["no-judgments", "with-judgments"])
def test_evaluate_queries_rejects_duplicate_products_even_beyond_cutoffs(
    judgments: dict[int, dict[int, int]],
) -> None:
    with pytest.raises(ValueError):
        evaluate_queries(
            {1: [101, 102, 101]},
            judgments,
            query_ids=[1],
            ndcg_k=1,
            recall_k=1,
            rr_k=1,
            judged_k=1,
        )


def test_aggregate_metrics_weights_each_query_equally() -> None:
    per_query = pl.DataFrame(
        [
            (10, 5, 5, 1, True, 1.0, 1.0, 1.0, 1.0),
            (20, 5, 7, 5, True, 0.4, 0.2, 0.5, 0.6),
        ],
        schema=PER_QUERY_SCHEMA,
        orient="row",
    )

    assert aggregate_metrics(per_query) == {
        "num_queries": 2,
        "num_empty_rankings": 0,
        "num_queries_without_judgments": 0,
        "num_queries_without_positive_judgments": 0,
        "num_queries_without_relevant": 0,
        "ndcg": pytest.approx((1.0 + 0.4) / 2),
        "recall": pytest.approx((1.0 + 0.2) / 2),
        "mrr": pytest.approx((1.0 + 0.5) / 2),
        "judged": pytest.approx((1.0 + 0.6) / 2),
    }


def test_aggregate_metrics_uses_eligible_queries_and_keeps_empty_rankings() -> None:
    per_query = pl.DataFrame(
        [
            (10, 4, 6, 5, True, 0.8, 0.6, 0.5, 0.75),
            (20, 0, 1, 1, True, 0.0, 0.0, 0.0, 0.0),
            (30, 2, 3, 0, True, 0.6, 0.0, 0.0, 0.5),
            (40, 2, 2, 0, False, 0.0, 0.0, 0.0, 0.25),
            (50, 1, 0, 0, False, 0.0, 0.0, 0.0, 0.0),
        ],
        schema=PER_QUERY_SCHEMA,
        orient="row",
    )

    assert aggregate_metrics(per_query) == {
        "num_queries": 5,
        "num_empty_rankings": 1,
        "num_queries_without_judgments": 1,
        "num_queries_without_positive_judgments": 2,
        "num_queries_without_relevant": 3,
        "ndcg": pytest.approx((0.8 + 0.0 + 0.6) / 3),
        "recall": pytest.approx((0.6 + 0.0) / 2),
        "mrr": pytest.approx((0.5 + 0.0) / 2),
        "judged": pytest.approx((0.75 + 0.0 + 0.5 + 0.25 + 0.0) / 5),
    }


def test_aggregate_metrics_returns_none_for_metrics_without_eligible_queries() -> None:
    per_query = pl.DataFrame(
        [(10, 2, 1, 0, False, 0.0, 0.0, 0.0, 0.25)],
        schema=PER_QUERY_SCHEMA,
        orient="row",
    )

    assert aggregate_metrics(per_query) == {
        "num_queries": 1,
        "num_empty_rankings": 0,
        "num_queries_without_judgments": 0,
        "num_queries_without_positive_judgments": 1,
        "num_queries_without_relevant": 1,
        "ndcg": None,
        "recall": None,
        "mrr": None,
        "judged": pytest.approx(0.25),
    }


def test_aggregate_metrics_returns_zero_counts_and_none_for_empty_input() -> None:
    assert aggregate_metrics(pl.DataFrame(schema=PER_QUERY_SCHEMA)) == {
        "num_queries": 0,
        "num_empty_rankings": 0,
        "num_queries_without_judgments": 0,
        "num_queries_without_positive_judgments": 0,
        "num_queries_without_relevant": 0,
        "ndcg": None,
        "recall": None,
        "mrr": None,
        "judged": None,
    }
