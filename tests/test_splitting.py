import polars as pl
import pytest
from polars.testing import assert_frame_equal

from rankoon.data.splitting import split_queries


@pytest.fixture
def query_groups() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "query_id": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110],
            "group_id": [10, 10, 30, 40, 50, 60, 70, 80, 90, 100, 110],
        }
    )


def test_split_queries_returns_one_row_per_query_with_expected_schema(
    query_groups: pl.DataFrame,
) -> None:
    source = query_groups.reverse().with_columns(pl.lit("ignored").alias("extra"))

    result = split_queries(source)

    assert result.columns == ["query_id", "group_id", "split"]
    assert result.schema == {
        "query_id": pl.Int64,
        "group_id": pl.Int64,
        "split": pl.String,
    }
    assert_frame_equal(result.select("query_id", "group_id"), query_groups)
    assert set(result["split"]) == {"train", "val", "test"}


def test_split_queries_keeps_each_group_in_one_split(
    query_groups: pl.DataFrame,
) -> None:
    result = split_queries(query_groups)

    splits_per_group = result.group_by("group_id").agg(pl.col("split").n_unique())
    assert (splits_per_group["split"] == 1).all()


@pytest.mark.parametrize(
    ("excluded_group", "ratios", "expected"),
    [
        (110, {}, {"train": 5, "val": 1, "test": 3}),
        (None, {"train_ratio": 0.5, "val_ratio": 0.3}, {"train": 5, "val": 3, "test": 2}),
    ],
    ids=["default-ratios-with-rounding", "custom-ratios"],
)
def test_split_queries_allocates_groups_using_ratios(
    query_groups: pl.DataFrame,
    excluded_group: int | None,
    ratios: dict[str, float],
    expected: dict[str, int],
) -> None:
    if excluded_group is not None:
        query_groups = query_groups.filter(pl.col("group_id") != excluded_group)

    result = split_queries(query_groups, **ratios)

    counts = result.select("group_id", "split").unique().group_by("split").len()
    assert dict(counts.iter_rows()) == expected


def test_split_queries_is_reproducible_for_the_same_seed(
    query_groups: pl.DataFrame,
) -> None:
    first = split_queries(query_groups, seed=17)
    second = split_queries(query_groups, seed=17)

    assert_frame_equal(first, second)


def test_split_queries_is_independent_of_input_row_order(
    query_groups: pl.DataFrame,
) -> None:
    first = split_queries(query_groups, seed=17)
    second = split_queries(query_groups.reverse(), seed=17)

    assert_frame_equal(first, second)


@pytest.mark.parametrize("parameter", ["train_ratio", "val_ratio"])
@pytest.mark.parametrize("ratio", [-0.1, 0.0, 1.0, 1.1, float("nan"), float("inf"), -float("inf")])
def test_split_queries_rejects_invalid_individual_ratios(
    query_groups: pl.DataFrame, parameter: str, ratio: float
) -> None:
    with pytest.raises(ValueError):
        split_queries(query_groups, **{parameter: ratio})


@pytest.mark.parametrize(("train_ratio", "val_ratio"), [(0.8, 0.2), (0.8, 0.3)])
def test_split_queries_rejects_ratios_leaving_no_test_share(
    query_groups: pl.DataFrame, train_ratio: float, val_ratio: float
) -> None:
    with pytest.raises(ValueError):
        split_queries(query_groups, train_ratio=train_ratio, val_ratio=val_ratio)


@pytest.mark.parametrize("column", ["query_id", "group_id"])
def test_split_queries_requires_both_id_columns(
    query_groups: pl.DataFrame, column: str
) -> None:
    with pytest.raises(ValueError):
        split_queries(query_groups.drop(column))


@pytest.mark.parametrize("column", ["query_id", "group_id"])
def test_split_queries_rejects_null_ids(
    query_groups: pl.DataFrame, column: str
) -> None:
    source = query_groups.with_columns(
        pl.when(pl.col("query_id") == 10).then(None).otherwise(pl.col(column)).alias(column)
    )

    with pytest.raises(ValueError):
        split_queries(source)


def test_split_queries_rejects_repeated_query_ids(query_groups: pl.DataFrame) -> None:
    source = query_groups.vstack(query_groups.head(1))

    with pytest.raises(ValueError):
        split_queries(source)


def test_split_queries_rejects_empty_input(query_groups: pl.DataFrame) -> None:
    with pytest.raises(ValueError):
        split_queries(query_groups.head(0))


def test_split_queries_rejects_too_few_groups_at_default_ratios(
    query_groups: pl.DataFrame,
) -> None:
    four_groups = query_groups.head(5)

    with pytest.raises(ValueError):
        split_queries(four_groups)


@pytest.mark.parametrize("ratios", [{"train_ratio": 0.05}, {"val_ratio": 0.05}])
def test_split_queries_rejects_too_few_groups_at_custom_ratios(
    query_groups: pl.DataFrame, ratios: dict[str, float]
) -> None:
    with pytest.raises(ValueError):
        split_queries(query_groups, **ratios)
