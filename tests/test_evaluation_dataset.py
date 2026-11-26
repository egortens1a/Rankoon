from collections.abc import Iterator
from pathlib import Path
from tempfile import TemporaryDirectory

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from rankoon.evaluation.dataset import load_evaluation_split
from rankoon.evaluation.evaluator import evaluate_queries


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def evaluation_files() -> Iterator[tuple[Path, Path]]:
    with TemporaryDirectory(dir=ROOT, prefix=".pytest-evaluation-") as directory:
        root = Path(directory)
        processed_dir = root / "processed"
        processed_dir.mkdir()
        splits_path = root / "query_splits.csv"

        pl.DataFrame(
            {
                "query_id": [3, 2, 4, 1],
                "query": ["bedroom mirror", "wooden table", "stool", "leather chair"],
                "query_class": ["decor", "furniture", "furniture", "furniture"],
            }
        ).write_parquet(processed_dir / "queries.parquet")
        pl.DataFrame(
            {
                "query_id": [1, 2, 1, 4, 1],
                "product_id": [101, 201, 103, 401, 102],
                "relevance": [2, 2, 0, 1, 1],
                "id": [10, 11, 12, 13, 14],
            }
        ).write_parquet(processed_dir / "labels.parquet")
        pl.DataFrame(
            {
                "query_id": [3, 4, 1, 2],
                "group_id": [3, 4, 1, 2],
                "split": ["val", "test", "val", "train"],
                "description": [""] * 4,
            }
        ).write_csv(splits_path)

        yield processed_dir, splits_path


@pytest.mark.parametrize(
    ("options", "query_rows", "expected_judgments"),
    [
        (
            {},
            [(1, "leather chair"), (3, "bedroom mirror")],
            {1: {101: 2, 102: 1, 103: 0}, 3: {}},
        ),
        ({"split": "train"}, [(2, "wooden table")], {2: {201: 2}}),
        ({"split": "test"}, [(4, "stool")], {4: {401: 1}}),
    ],
    ids=["default-validation", "train", "test"],
)
def test_load_evaluation_split_returns_sorted_queries_and_nested_judgments(
    evaluation_files: tuple[Path, Path],
    options: dict[str, str],
    query_rows: list[tuple[int, str]],
    expected_judgments: dict[int, dict[int, int]],
) -> None:
    queries, judgments = load_evaluation_split(*evaluation_files, **options)

    expected_queries = pl.DataFrame(
        query_rows, schema={"query_id": pl.Int64, "query": pl.String}, orient="row"
    )
    assert_frame_equal(queries, expected_queries)
    assert judgments == expected_judgments


def test_load_evaluation_split_keeps_queries_when_labels_are_empty(
    evaluation_files: tuple[Path, Path],
) -> None:
    labels_path = evaluation_files[0] / "labels.parquet"
    pl.read_parquet(labels_path).head(0).write_parquet(labels_path)

    queries, judgments = load_evaluation_split(*evaluation_files)

    assert queries["query_id"].to_list() == [1, 3]
    assert judgments == {1: {}, 3: {}}


def test_load_evaluation_split_output_can_be_evaluated(
    evaluation_files: tuple[Path, Path],
) -> None:
    queries, judgments = load_evaluation_split(*evaluation_files)

    scores = evaluate_queries(
        rankings={1: [103, 101], 3: [301]},
        judgments=judgments,
        query_ids=queries["query_id"].to_list(),
        ndcg_k=2,
        recall_k=2,
        rr_k=2,
        judged_k=2,
    )

    assert scores.select(
        "query_id", "num_judgments", "num_relevant", "recall", "reciprocal_rank", "judged"
    ).rows() == [(1, 3, 2, 0.5, 0.5, 1.0), (3, 0, 0, 0.0, 0.0, 0.0)]


def test_load_evaluation_split_rejects_unknown_split(
    evaluation_files: tuple[Path, Path],
) -> None:
    with pytest.raises(ValueError):
        load_evaluation_split(*evaluation_files, split="validation")


def test_load_evaluation_split_rejects_empty_selected_split(
    evaluation_files: tuple[Path, Path],
) -> None:
    splits_path = evaluation_files[1]
    pl.read_csv(splits_path).filter(pl.col("split") != "val").write_csv(splits_path)

    with pytest.raises(ValueError):
        load_evaluation_split(*evaluation_files)


@pytest.mark.parametrize("duplicate_split", ["val", "train"], ids=["same-split", "different-split"])
def test_load_evaluation_split_rejects_duplicate_manifest_query_ids(
    evaluation_files: tuple[Path, Path], duplicate_split: str
) -> None:
    splits_path = evaluation_files[1]
    manifest = pl.read_csv(splits_path)
    duplicate = manifest.filter(pl.col("query_id") == 1).with_columns(
        pl.lit(duplicate_split).alias("split")
    )
    manifest.vstack(duplicate).write_csv(splits_path)

    with pytest.raises(ValueError):
        load_evaluation_split(*evaluation_files)


@pytest.mark.parametrize("relevance", [2, 1], ids=["same-grade", "different-grade"])
def test_load_evaluation_split_rejects_duplicate_selected_label_pairs(
    evaluation_files: tuple[Path, Path], relevance: int
) -> None:
    labels_path = evaluation_files[0] / "labels.parquet"
    labels = pl.read_parquet(labels_path)
    duplicate = labels.filter(pl.col("product_id") == 101).with_columns(
        pl.lit(relevance, dtype=pl.Int64).alias("relevance")
    )
    labels.vstack(duplicate).write_parquet(labels_path)

    with pytest.raises(ValueError):
        load_evaluation_split(*evaluation_files)


def test_load_evaluation_split_ignores_duplicate_label_pairs_in_other_splits(
    evaluation_files: tuple[Path, Path],
) -> None:
    labels_path = evaluation_files[0] / "labels.parquet"
    labels = pl.read_parquet(labels_path)
    labels.vstack(labels.filter(pl.col("query_id") == 2)).write_parquet(labels_path)

    _, judgments = load_evaluation_split(*evaluation_files)

    assert judgments == {1: {101: 2, 102: 1, 103: 0}, 3: {}}


def test_load_evaluation_split_rejects_manifest_with_unknown_query(
    evaluation_files: tuple[Path, Path],
) -> None:
    splits_path = evaluation_files[1]
    manifest = pl.read_csv(splits_path)
    unknown = manifest.filter(pl.col("query_id") == 1).with_columns(
        pl.lit(999, dtype=pl.Int64).alias("query_id"),
        pl.lit(999, dtype=pl.Int64).alias("group_id"),
    )
    manifest.vstack(unknown).write_csv(splits_path)

    with pytest.raises(ValueError):
        load_evaluation_split(*evaluation_files)


@pytest.mark.parametrize(
    ("filename", "column"),
    [
        ("queries.parquet", "query_id"),
        ("queries.parquet", "query"),
        ("labels.parquet", "query_id"),
        ("labels.parquet", "product_id"),
        ("labels.parquet", "relevance"),
        ("query_splits.csv", "query_id"),
        ("query_splits.csv", "group_id"),
        ("query_splits.csv", "split"),
    ],
)
@pytest.mark.parametrize(
    ("problem", "expected_error"),
    [("missing", pl.exceptions.ColumnNotFoundError), ("null", ValueError)],
    ids=["missing", "null"],
)
def test_load_evaluation_split_rejects_missing_or_null_required_fields(
    evaluation_files: tuple[Path, Path],
    filename: str,
    column: str,
    problem: str,
    expected_error: type[Exception],
) -> None:
    processed_dir, splits_path = evaluation_files
    is_csv = filename == splits_path.name
    path = splits_path if is_csv else processed_dir / filename
    frame = pl.read_csv(path) if is_csv else pl.read_parquet(path)
    if problem == "missing":
        frame = frame.drop(column)
    else:
        frame = frame.with_columns(
            pl.when(pl.int_range(pl.len()) == 0).then(None).otherwise(pl.col(column)).alias(column)
        )

    if is_csv:
        frame.write_csv(path)
    else:
        frame.write_parquet(path)

    with pytest.raises(expected_error):
        load_evaluation_split(*evaluation_files)


@pytest.mark.parametrize("relevance", [-1, 3], ids=["below-zero", "above-two"])
def test_load_evaluation_split_rejects_invalid_relevance(
    evaluation_files: tuple[Path, Path], relevance: int
) -> None:
    labels_path = evaluation_files[0] / "labels.parquet"
    labels = pl.read_parquet(labels_path).with_columns(
        pl.when(pl.col("product_id") == 101)
        .then(relevance)
        .otherwise(pl.col("relevance"))
        .alias("relevance")
    )
    labels.write_parquet(labels_path)

    with pytest.raises(ValueError):
        load_evaluation_split(*evaluation_files)


@pytest.mark.parametrize("filename", ["queries.parquet", "labels.parquet", "query_splits.csv"])
def test_load_evaluation_split_raises_file_not_found_for_missing_inputs(
    evaluation_files: tuple[Path, Path], filename: str
) -> None:
    processed_dir, splits_path = evaluation_files
    path = splits_path if filename == splits_path.name else processed_dir / filename
    path.rename(path.with_name(f"unused-{filename}"))

    with pytest.raises(FileNotFoundError):
        load_evaluation_split(*evaluation_files)
