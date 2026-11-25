from io import StringIO
from pathlib import Path
from uuid import uuid4

import polars as pl
import pytest

from rankoon.data.loader import load_wands


WANDS_DIR = Path("wands")


@pytest.fixture
def wands_files(monkeypatch: pytest.MonkeyPatch) -> dict[Path, str]:
    files = {
        WANDS_DIR / "product.csv": "product_id\tproduct_name\tproduct_description\n100\tChair\t\n",
        WANDS_DIR / "label.csv": "id\tquery_id\tproduct_id\tlabel\n10\t1\t100\tExact\n11\t1\t100\tPartial\n",
        WANDS_DIR / "query.csv": "query_id\tquery\n1\tchair\n",
    }
    read_csv = pl.read_csv

    def read_fixture(path: Path, **kwargs) -> pl.DataFrame:
        return read_csv(StringIO(files[path]), **kwargs)

    monkeypatch.setattr(pl, "read_csv", read_fixture)
    return files


def test_load_wands_returns_raw_tables(wands_files: dict[Path, str]) -> None:
    product, label, query = load_wands(WANDS_DIR)

    assert product.rows() == [(100, "Chair", None)]
    assert label.rows() == [(10, 1, 100, "Exact"), (11, 1, 100, "Partial")]
    assert query.rows() == [(1, "chair")]


def test_load_wands_raises_when_a_file_is_missing() -> None:
    missing_dir = Path(__file__).parent / uuid4().hex

    with pytest.raises(FileNotFoundError, match=r"product\.csv"):
        load_wands(missing_dir)


def test_load_wands_accepts_empty_label_table(wands_files: dict[Path, str]) -> None:
    wands_files[WANDS_DIR / "label.csv"] = "query_id\tproduct_id\tlabel\n"

    _, label, _ = load_wands(WANDS_DIR)

    assert label.is_empty()
    assert label.schema["query_id"] == label.schema["product_id"] == pl.Int64


@pytest.mark.parametrize(
    ("filename", "content", "message"),
    [
        ("product.csv", "product_id\n100\n", "missing required columns: product_name"),
        ("query.csv", "query\nchair\n", "missing required columns: query_id"),
        ("label.csv", "query_id\tproduct_id\n1\t100\n", "missing required columns: label"),
        ("product.csv", "product_id\tproduct_name\n\tChair\n", "null values in required columns: product_id"),
        ("query.csv", "query_id\tquery\n\tchair\n", "null values in required columns: query_id"),
        ("label.csv", "query_id\tproduct_id\tlabel\n\t100\tExact\n", "null values in required columns: query_id"),
        ("label.csv", "query_id\tproduct_id\tlabel\n1\t\tExact\n", "null values in required columns: product_id"),
        ("product.csv", "product_id\tproduct_name\n100\t\n", "null values in required columns: product_name"),
        ("query.csv", "query_id\tquery\n1\t\n", "null values in required columns: query"),
        ("label.csv", "query_id\tproduct_id\tlabel\n1\t100\t\n", "null values in required columns: label"),
        ("product.csv", "product_id\tproduct_name\n100\tChair\n100\tTable\n", "duplicate product_id"),
        ("query.csv", "query_id\tquery\n1\tchair\n1\ttable\n", "duplicate query_id"),
        ("label.csv", "query_id\tproduct_id\tlabel\n1\t999\tExact\n", "unknown product_id"),
        ("label.csv", "query_id\tproduct_id\tlabel\n999\t100\tExact\n", "unknown query_id"),
    ],
    ids=[
        "missing-product-name",
        "missing-query-id",
        "missing-label",
        "null-product-key",
        "null-query-key",
        "null-label-query-key",
        "null-label-product-key",
        "null-product-name",
        "null-query-text",
        "null-label-text",
        "duplicate-product-key",
        "duplicate-query-key",
        "unknown-product-reference",
        "unknown-query-reference",
    ],
)
def test_load_wands_rejects_invalid_tables(
    wands_files: dict[Path, str], filename: str, content: str, message: str
) -> None:
    wands_files[WANDS_DIR / filename] = content

    with pytest.raises(ValueError) as error:
        load_wands(WANDS_DIR)

    assert f"{filename}: {message}" in str(error.value)
