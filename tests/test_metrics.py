from collections.abc import Callable
from math import log2

import pytest

from rankoon.evaluation.metrics import (
    dcg_at_k,
    judged_at_k,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)

RANKING_METRICS = (recall_at_k, reciprocal_rank_at_k, judged_at_k)


def test_dcg_at_k_matches_hand_calculation() -> None:
    expected = 3 + 1 / log2(3) + 3 / log2(5)

    assert dcg_at_k([2, 1, 0, 2], k=4) == pytest.approx(expected)


def test_ndcg_at_k_is_one_for_ideal_ranking() -> None:
    judgments = {20: 1, 30: 0, 10: 2}

    assert ndcg_at_k([10, 20, 30], judgments, k=3) == pytest.approx(1.0)


def test_ndcg_at_k_decreases_when_products_are_out_of_order() -> None:
    judgments = {10: 2, 20: 1, 30: 0}
    expected = (1 + 3 / log2(3)) / (3 + 1 / log2(3))

    ideal = ndcg_at_k([10, 20, 30], judgments, k=3)
    reordered = ndcg_at_k([20, 10, 30], judgments, k=3)

    assert reordered == pytest.approx(expected)
    assert reordered < ideal


def test_ndcg_at_k_includes_unretrieved_relevant_products_in_idcg() -> None:
    judgments = {10: 2, 20: 1}
    expected = 1 / (3 + 1 / log2(3))

    assert ndcg_at_k([20], judgments, k=2) == pytest.approx(expected)


def test_ndcg_at_k_gives_unjudged_products_zero_gain() -> None:
    judgments = {10: 2}

    assert ndcg_at_k([999, 10], judgments, k=2) == pytest.approx(1 / log2(3))


def test_dcg_at_k_ignores_relevances_after_cutoff() -> None:
    assert dcg_at_k([1, 0, 2], k=2) == pytest.approx(1.0)


def test_ndcg_at_k_applies_cutoff_to_both_dcg_and_idcg() -> None:
    judgments = {10: 2, 20: 1}

    assert ndcg_at_k([20, 10], judgments, k=1) == pytest.approx(1 / 3)


def test_dcg_at_k_returns_zero_for_empty_input() -> None:
    assert dcg_at_k([], k=3) == 0.0


def test_ndcg_at_k_returns_zero_for_empty_ranking() -> None:
    assert ndcg_at_k([], {10: 2}, k=3) == 0.0


@pytest.mark.parametrize("judgments", [{}, {10: 0, 20: 0}], ids=["unjudged", "all-zero"])
def test_ndcg_at_k_returns_zero_without_positive_judgments(
    judgments: dict[int, int],
) -> None:
    assert ndcg_at_k([10, 20], judgments, k=2) == 0.0


@pytest.mark.parametrize("k", [0, -1, 1.5], ids=["zero", "negative", "fractional"])
def test_dcg_at_k_rejects_invalid_cutoff(k: int | float) -> None:
    with pytest.raises((ValueError, TypeError)):
        dcg_at_k([2, 1], k=k)


@pytest.mark.parametrize("k", [0, -1, 1.5], ids=["zero", "negative", "fractional"])
def test_ndcg_at_k_rejects_invalid_cutoff(k: int | float) -> None:
    with pytest.raises((ValueError, TypeError)):
        ndcg_at_k([10, 20], {10: 2, 20: 1}, k=k)


@pytest.mark.parametrize(
    "ranked_ids",
    [[10, 10], [10, 20, 10]],
    ids=["within-cutoff", "beyond-cutoff"],
)
def test_ndcg_at_k_rejects_duplicate_product_ids(ranked_ids: list[int]) -> None:
    with pytest.raises(ValueError):
        ndcg_at_k(ranked_ids, {10: 2, 20: 1}, k=2)


@pytest.mark.parametrize("relevance", [-1, 1.5], ids=["negative", "fractional"])
def test_dcg_at_k_rejects_invalid_relevance(relevance: int | float) -> None:
    with pytest.raises((ValueError, TypeError)):
        dcg_at_k([2, relevance], k=2)


@pytest.mark.parametrize("relevance", [-1, 1.5], ids=["negative", "fractional"])
def test_ndcg_at_k_rejects_invalid_judgments_for_unretrieved_products(
    relevance: int | float,
) -> None:
    with pytest.raises((ValueError, TypeError)):
        ndcg_at_k([10], {10: 2, 20: relevance}, k=2)


def test_recall_at_k_counts_unretrieved_relevant_products_and_respects_cutoff() -> None:
    judgments = {10: 2, 20: 1, 30: 2, 40: 0}

    assert recall_at_k([10, 40, 999, 20], judgments, k=3) == pytest.approx(1 / 3)


def test_recall_at_k_ignores_order_within_cutoff() -> None:
    judgments = {10: 2, 20: 1, 30: 2, 40: 0}

    first = recall_at_k([10, 40, 20, 30], judgments, k=3)
    reordered = recall_at_k([20, 10, 40, 30], judgments, k=3)

    assert first == reordered == pytest.approx(2 / 3)


@pytest.mark.parametrize(("min_relevance", "expected"), [(1, 0.75), (2, 0.5)])
def test_recall_at_k_respects_min_relevance(min_relevance: int, expected: float) -> None:
    judgments = {10: 1, 20: 2, 30: 1, 40: 0, 50: 2}

    result = recall_at_k([10, 20, 30], judgments, k=3, min_relevance=min_relevance)

    assert result == pytest.approx(expected)


def test_reciprocal_rank_at_k_uses_first_relevant_product_position() -> None:
    judgments = {10: 1, 20: 2, 40: 0}

    assert reciprocal_rank_at_k([999, 40, 10, 20], judgments, k=4) == pytest.approx(1 / 3)


def test_reciprocal_rank_at_k_ignores_products_after_first_match() -> None:
    judgments = {10: 1, 20: 2}

    first_match = reciprocal_rank_at_k([999, 20], judgments, k=3)
    with_later_match = reciprocal_rank_at_k([999, 20, 10], judgments, k=3)

    assert first_match == with_later_match == pytest.approx(0.5)


@pytest.mark.parametrize(("min_relevance", "expected"), [(1, 1.0), (2, 0.5)])
def test_reciprocal_rank_at_k_respects_min_relevance(
    min_relevance: int, expected: float
) -> None:
    result = reciprocal_rank_at_k([10, 20], {10: 1, 20: 2}, k=2, min_relevance=min_relevance)

    assert result == pytest.approx(expected)


def test_reciprocal_rank_at_k_excludes_first_match_after_cutoff() -> None:
    result = reciprocal_rank_at_k([999, 20], {20: 2}, k=1)

    assert type(result) is float
    assert result == 0.0


def test_judged_at_k_counts_irrelevant_products() -> None:
    assert judged_at_k([10], {10: 0}, k=1) == 1.0


def test_judged_at_k_excludes_unjudged_products_and_respects_cutoff() -> None:
    judgments = {10: 2, 20: 0, 30: 1}

    assert judged_at_k([10, 999, 20, 30], judgments, k=3) == pytest.approx(2 / 3)


def test_judged_at_k_uses_k_for_short_rankings() -> None:
    result = judged_at_k([10, 20], {10: 2, 20: 0}, k=4)

    assert type(result) is float
    assert result == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("metric", "expected"), [(recall_at_k, 0.5), (reciprocal_rank_at_k, 1.0)]
)
def test_relevance_metrics_use_available_results_for_short_rankings(
    metric: Callable[..., float], expected: float
) -> None:
    result = metric([10], {10: 2, 20: 2}, k=5)

    assert type(result) is float
    assert result == pytest.approx(expected)


@pytest.mark.parametrize("metric", RANKING_METRICS)
@pytest.mark.parametrize(
    ("ranked_ids", "judgments"),
    [([], {10: 2}), ([10], {}), ([], {})],
    ids=["empty-ranking", "empty-judgments", "both-empty"],
)
def test_ranking_metrics_return_float_zero_for_empty_inputs(
    metric: Callable[..., float], ranked_ids: list[int], judgments: dict[int, int]
) -> None:
    result = metric(ranked_ids, judgments, k=3)

    assert type(result) is float
    assert result == 0.0


@pytest.mark.parametrize("metric", [recall_at_k, reciprocal_rank_at_k])
def test_relevance_metrics_return_float_zero_without_relevant_judgments(
    metric: Callable[..., float],
) -> None:
    result = metric([10, 20], {10: 0, 20: 1}, k=2, min_relevance=2)

    assert type(result) is float
    assert result == 0.0


@pytest.mark.parametrize("metric", RANKING_METRICS)
@pytest.mark.parametrize("k", [0, -1, 1.5, 1.0], ids=["zero", "negative", "fractional", "float"])
def test_ranking_metrics_validate_cutoff_before_returning_for_empty_inputs(
    metric: Callable[..., float], k: int | float
) -> None:
    with pytest.raises(ValueError):
        metric([], {}, k=k)


@pytest.mark.parametrize("metric", RANKING_METRICS)
def test_ranking_metrics_validate_duplicates_beyond_cutoff_with_empty_judgments(
    metric: Callable[..., float],
) -> None:
    with pytest.raises(ValueError):
        metric([10, 20, 10], {}, k=1)


@pytest.mark.parametrize("metric", RANKING_METRICS)
@pytest.mark.parametrize("ranked_ids", [[], [10]], ids=["empty-ranking", "unretrieved-product"])
@pytest.mark.parametrize(
    "relevance", [-1, 3, 1.5, 1.0], ids=["negative", "above-two", "fractional", "float"]
)
def test_ranking_metrics_validate_all_relevance_grades(
    metric: Callable[..., float], ranked_ids: list[int], relevance: int | float
) -> None:
    with pytest.raises(ValueError):
        metric(ranked_ids, {10: 2, 20: relevance}, k=1)


@pytest.mark.parametrize("metric", [recall_at_k, reciprocal_rank_at_k])
@pytest.mark.parametrize(
    "min_relevance", [-1, 0, 3, 1.5, 1.0], ids=["negative", "zero", "three", "fractional", "float"]
)
def test_relevance_metrics_validate_threshold_before_returning_for_empty_inputs(
    metric: Callable[..., float], min_relevance: int | float
) -> None:
    with pytest.raises(ValueError):
        metric([], {}, k=1, min_relevance=min_relevance)
