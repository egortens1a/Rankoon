from math import log2

import pytest

from rankoon.evaluation.metrics import dcg_at_k, ndcg_at_k


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
