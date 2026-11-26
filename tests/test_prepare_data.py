import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from runpy import run_path
from tempfile import TemporaryDirectory

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from rankoon.data.splitting import split_queries


ROOT = Path(__file__).resolve().parents[1]
prepare_data = run_path(str(ROOT / "scripts" / "prepare_data.py"))["prepare_data"]


@pytest.fixture
def work_dir() -> Iterator[Path]:
    with TemporaryDirectory(dir=ROOT, prefix=".pytest-prepare-") as directory:
        yield Path(directory)


@pytest.fixture
def wands_inputs(work_dir: Path) -> tuple[Path, Path]:
    raw_dir = work_dir / "raw"
    raw_dir.mkdir()
    (raw_dir / "product.csv").write_text(
        "product_id\tproduct_name\tproduct_description\n100\tChair\tWood\n101\tTable\tMetal\n",
        encoding="utf-8",
    )
    (raw_dir / "query.csv").write_text(
        "query_id\tquery\n3\twood chair\n1\tchair\n7\tarmchair\n2\tchairs\n"
        "6\tstool\n5\tdesk\n4\ttable\n",
        encoding="utf-8",
    )
    (raw_dir / "label.csv").write_text(
        "id\tquery_id\tproduct_id\tlabel\n"
        "10\t1\t100\tExact\n11\t1\t100\tExact\n"
        "12\t2\t100\tExact\n13\t2\t100\tPartial\n"
        "14\t3\t101\tPartial\n15\t7\t101\tIrrelevant\n",
        encoding="utf-8",
    )
    annotations = work_dir / "candidate_pairs.csv"
    annotations.write_text(
        "query_id_a,query_id_b,is_duplicate\n1,2,1\n2,3,1\n3,4,0\n",
        encoding="utf-8",
    )
    return raw_dir, annotations


def test_prepare_data_can_rerun_in_the_same_output_directories(
    work_dir: Path, wands_inputs: tuple[Path, Path]
) -> None:
    raw_dir, annotations = wands_inputs
    processed_dir = work_dir / "processed"
    splits_dir = work_dir / "splits"

    prepare_data(raw_dir, annotations, processed_dir, splits_dir)

    assert processed_dir.is_dir()
    assert splits_dir.is_dir()
    prepare_data(raw_dir, annotations, processed_dir, splits_dir)


def test_prepare_data_writes_wands_tables(
    work_dir: Path, wands_inputs: tuple[Path, Path]
) -> None:
    raw_dir, annotations = wands_inputs
    processed_dir = work_dir / "processed"
    splits_dir = work_dir / "splits"

    prepare_data(raw_dir, annotations, processed_dir, splits_dir)

    for table, raw_filename, key in (
        ("products", "product.csv", "product_id"),
        ("queries", "query.csv", "query_id"),
    ):
        result = pl.read_parquet(processed_dir / f"{table}.parquet")
        raw = pl.read_csv(raw_dir / raw_filename, separator="\t")
        assert_frame_equal(result.sort(key), raw.sort(key))

    labels = pl.read_parquet(processed_dir / "labels.parquet").sort("id")
    assert labels.columns == ["id", "query_id", "product_id", "label", "relevance"]
    assert labels.rows() == [
        (10, 1, 100, "Exact", 2),
        (14, 3, 101, "Partial", 1),
        (15, 7, 101, "Irrelevant", 0),
    ]
    conflicts = pl.read_parquet(processed_dir / "conflicts.parquet").sort("id")
    raw_labels = pl.read_csv(raw_dir / "label.csv", separator="\t")
    assert_frame_equal(conflicts, raw_labels.filter(pl.col("query_id") == 2))


def test_prepare_data_records_metadata_and_input_hashes(
    work_dir: Path, wands_inputs: tuple[Path, Path]
) -> None:
    raw_dir, annotations = wands_inputs
    processed_dir = work_dir / "processed"
    splits_dir = work_dir / "splits"

    prepare_data(raw_dir, annotations, processed_dir, splits_dir, train_ratio=0.4, seed=17)

    metadata = json.loads((processed_dir / "metadata.json").read_text(encoding="utf-8"))
    expected = {
        "dataset": "WANDS",
        "schema_version": 1,
        "seed": 17,
        "train_ratio": 0.4,
        "val_ratio": 0.2,
        "num_products": 2,
        "num_queries": 7,
        "num_raw_labels": 6,
        "num_cleaned_labels": 3,
        "num_conflicting_rows": 2,
        "num_conflicting_pairs": 1,
        "num_duplicate_pairs": 2,
        "num_query_groups": 5,
    }
    assert {key: metadata[key] for key in expected} == expected

    manifest = pl.read_csv(splits_dir / "query_splits.csv")
    for split, group_count in {"train": 2, "val": 1, "test": 2}.items():
        subset = manifest.filter(pl.col("split") == split)
        assert subset["group_id"].n_unique() == group_count
        assert metadata["split_counts"][split] == {
            "queries": subset.height,
            "groups": group_count,
        }

    expected_hashes = {
        filename: hashlib.sha256((raw_dir / filename).read_bytes()).hexdigest()
        for filename in ("product.csv", "query.csv", "label.csv")
    }
    expected_hashes["annotations"] = hashlib.sha256(annotations.read_bytes()).hexdigest()
    assert metadata["input_sha256"] == expected_hashes


def test_prepare_data_rejects_incomplete_split_manifest_before_writing(
    monkeypatch: pytest.MonkeyPatch, work_dir: Path, wands_inputs: tuple[Path, Path]
) -> None:
    raw_dir, annotations = wands_inputs
    processed_dir = work_dir / "processed"
    splits_dir = work_dir / "splits"
    monkeypatch.setitem(
        prepare_data.__globals__,
        "split_queries",
        lambda *args, **kwargs: split_queries(*args, **kwargs).slice(1),
    )

    with pytest.raises(ValueError, match="each source query exactly once"):
        prepare_data(raw_dir, annotations, processed_dir, splits_dir)

    assert not processed_dir.exists()
    assert not splits_dir.exists()


def test_prepare_data_writes_reproducible_csv_split_manifest(
    work_dir: Path, wands_inputs: tuple[Path, Path]
) -> None:
    raw_dir, annotations = wands_inputs
    first = work_dir / "first"
    second = work_dir / "second"

    for output_dir in (first, second):
        output_dir.mkdir()
        prepare_data(
            raw_dir, annotations, output_dir / "processed", output_dir / "splits", seed=17
        )

    manifest_path = Path("splits") / "query_splits.csv"
    manifest = pl.read_csv(first / manifest_path)
    assert manifest.columns == ["query_id", "group_id", "split"]
    assert manifest["query_id"].to_list() == [1, 2, 3, 4, 5, 6, 7]
    assert manifest["group_id"].to_list() == [1, 1, 1, 4, 5, 6, 7]
    assert set(manifest["split"]) == {"train", "val", "test"}
    assert manifest.select("group_id", "split").unique().height == 5
    assert_frame_equal(manifest, pl.read_csv(second / manifest_path))
