import polars as pl
import pytest
from polars.testing import assert_frame_equal

from rankoon.data.grouping import build_query_groups


def _duplicate_pairs(rows: list[tuple[int | None, int | None]]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={"query_id_a": pl.Int64, "query_id_b": pl.Int64},
        orient="row",
    )


def test_build_query_groups_assigns_each_query_to_one_group() -> None:
    queries = pl.DataFrame(
        {"query_id": [40, 10, 30, 20], "query": ["d", "a", "c", "b"]}
    )

    result = build_query_groups(queries, _duplicate_pairs([(10, 20), (30, 40)]))

    expected = pl.DataFrame({"query_id": [10, 20, 30, 40], "group_id": [10, 10, 30, 30]})
    assert_frame_equal(result, expected)


def test_build_query_groups_merges_groups_through_transitive_pairs() -> None:
    queries = pl.DataFrame({"query_id": [10, 20, 30, 40]})
    pairs = _duplicate_pairs([(20, 40), (10, 30), (40, 30)])

    result = build_query_groups(queries, pairs)

    assert result["group_id"].to_list() == [10, 10, 10, 10]


def test_build_query_groups_keeps_isolated_queries_separate() -> None:
    queries = pl.DataFrame({"query_id": [10, 20, 30, 40]})

    result = build_query_groups(queries, _duplicate_pairs([(10, 20)]))

    group_10, group_20, group_30, group_40 = result["group_id"].to_list()
    assert group_10 == group_20
    assert len({group_10, group_30, group_40}) == 3
    assert result["group_id"].null_count() == 0


def test_build_query_groups_handles_empty_duplicate_pairs() -> None:
    queries = pl.DataFrame({"query_id": [30, 10, 20]})

    result = build_query_groups(queries, _duplicate_pairs([]))

    expected = pl.DataFrame({"query_id": [10, 20, 30], "group_id": [10, 20, 30]})
    assert_frame_equal(result, expected)


def test_build_query_groups_handles_empty_queries() -> None:
    queries = pl.DataFrame(schema={"query_id": pl.Int64})

    result = build_query_groups(queries, _duplicate_pairs([]))

    expected = pl.DataFrame(schema={"query_id": pl.Int64, "group_id": pl.Int64})
    assert_frame_equal(result, expected)


def test_build_query_groups_rejects_null_query_id() -> None:
    queries = pl.DataFrame({"query_id": [10, None]})

    with pytest.raises(ValueError, match="queries: null values in query_id"):
        build_query_groups(queries, _duplicate_pairs([]))


@pytest.mark.parametrize("pair", [(None, 10), (10, None)], ids=["first-id", "second-id"])
def test_build_query_groups_rejects_null_id_in_pair(
    pair: tuple[int | None, int | None],
) -> None:
    queries = pl.DataFrame({"query_id": [10, 20]})

    with pytest.raises(ValueError, match="query_id in duplicate_pairs is null"):
        build_query_groups(queries, _duplicate_pairs([pair]))


def test_build_query_groups_rejects_repeated_query_ids() -> None:
    queries = pl.DataFrame({"query_id": [10, 20, 10]})

    with pytest.raises(ValueError, match="queries: duplicate query_id values"):
        build_query_groups(queries, _duplicate_pairs([]))


def test_build_query_groups_accepts_repeated_pairs() -> None:
    queries = pl.DataFrame({"query_id": [10, 20]})
    pairs = _duplicate_pairs([(20, 10), (20, 10), (10, 20)])

    result = build_query_groups(queries, pairs)

    assert result["group_id"].to_list() == [10, 10]


def test_build_query_groups_requires_query_id_column() -> None:
    queries = pl.DataFrame({"query": ["example"]})

    with pytest.raises(ValueError, match="queries: missing required column query_id"):
        build_query_groups(queries, _duplicate_pairs([]))


@pytest.mark.parametrize("missing", [("query_id_a",), ("query_id_b",), ("query_id_a", "query_id_b")])
def test_build_query_groups_requires_pair_columns(missing: tuple[str, ...]) -> None:
    queries = pl.DataFrame({"query_id": [10, 20]})
    pairs = _duplicate_pairs([(10, 20)]).drop(*missing)

    with pytest.raises(ValueError) as error:
        build_query_groups(queries, pairs)

    assert str(error.value) == f"duplicate_pairs: missing required columns: {', '.join(missing)}"


@pytest.mark.parametrize("pair", [(999, 10), (10, 999)], ids=["first-id", "second-id"])
def test_build_query_groups_rejects_unknown_query_id(pair: tuple[int, int]) -> None:
    queries = pl.DataFrame({"query_id": [10, 20]})

    with pytest.raises(ValueError):
        build_query_groups(queries, _duplicate_pairs([pair]))


@pytest.mark.parametrize("rows", [[(10, 20), (30, 20)], [(20, 10), (20, 30)]])
def test_build_query_groups_result_is_independent_of_pair_order(
    rows: list[tuple[int | None, int | None]],
) -> None:
    queries = pl.DataFrame({"query_id": [10, 20, 30, 40]})
    pairs = _duplicate_pairs(rows)

    result = build_query_groups(queries, pairs)
    reversed_result = build_query_groups(queries, pairs.reverse())

    assert "group_id" in result.columns
    assert_frame_equal(result, reversed_result)
