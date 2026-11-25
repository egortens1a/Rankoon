import polars as pl
import pytest
from polars.testing import assert_frame_equal

from rankoon.data.cleaning import clean_labels


def test_clean_labels_deduplicates_identical_judgments() -> None:
    labels = pl.DataFrame(
        {
            "id": [20, 10],
            "query_id": [1, 1],
            "product_id": [100, 100],
            "label": ["Exact", "Exact"],
        }
    )

    cleaned, conflicts = clean_labels(labels)

    assert_frame_equal(cleaned.select(labels.columns), labels.head(1))
    assert conflicts.is_empty()


def test_clean_labels_excludes_only_conflicting_pairs() -> None:
    labels = pl.DataFrame(
        {
            "query_id": [1, 1, 1, 2],
            "product_id": [100, 100, 101, 100],
            "label": ["Partial", "Exact", "Exact", "Partial"],
        }
    )

    cleaned, _ = clean_labels(labels)

    assert_frame_equal(
        cleaned.select(labels.columns).sort("query_id", "product_id"), labels.tail(2)
    )


def test_clean_labels_returns_original_conflicting_rows() -> None:
    labels = pl.DataFrame(
        {
            "id": [10, 11, 12, 13],
            "query_id": [1, 1, 1, 2],
            "product_id": [100, 100, 100, 100],
            "label": ["Partial", "Partial", "Exact", "Irrelevant"],
            "source": ["a", "b", "c", "d"],
        }
    )

    _, conflicts = clean_labels(labels)

    assert conflicts.schema == labels.schema
    assert_frame_equal(conflicts.sort("id"), labels.head(3))


def test_clean_labels_maps_relevance_and_preserves_text_labels() -> None:
    labels = pl.DataFrame(
        {
            "query_id": [1, 1, 1],
            "product_id": [100, 101, 102],
            "label": ["Irrelevant", "Partial", "Exact"],
        }
    )

    cleaned, conflicts = clean_labels(labels)

    expected = labels.with_columns(pl.Series("relevance", [0, 1, 2]))
    assert_frame_equal(cleaned.sort("product_id"), expected)
    assert conflicts.is_empty()


@pytest.mark.parametrize(
    "judgments",
    [["Unknown"], ["Exact", "Unknown"]],
    ids=["single-label", "conflicting-labels"],
)
def test_clean_labels_rejects_unknown_labels(judgments: list[str]) -> None:
    labels = pl.DataFrame(
        {
            "query_id": [1] * len(judgments),
            "product_id": [100] * len(judgments),
            "label": judgments,
        }
    )

    with pytest.raises(pl.exceptions.InvalidOperationError):
        clean_labels(labels)


def test_clean_labels_keeps_same_product_for_different_queries() -> None:
    labels = pl.DataFrame(
        {"query_id": [1, 2], "product_id": [100, 100], "label": ["Exact", "Exact"]}
    )

    cleaned, conflicts = clean_labels(labels)

    assert_frame_equal(cleaned.select(labels.columns).sort("query_id"), labels)
    assert conflicts.is_empty()
