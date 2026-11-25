from pathlib import Path

import polars as pl


def load_wands(
    data_dir: Path,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Load raw WANDS tables and validate their keys and references.

    Args:
        data_dir: Directory containing product.csv, label.csv, and query.csv.
    Returns:
        (product, label, query) as Polars DataFrames.
    Raises:
        ValueError: Missing required columns, nulls in required columns,
            duplicate keys, or dangling label references.
    """
    product = pl.read_csv(
        data_dir / "product.csv", separator="\t", schema_overrides={"product_id": pl.Int64}
    )
    label = pl.read_csv(
        data_dir / "label.csv",
        separator="\t",
        schema_overrides={"query_id": pl.Int64, "product_id": pl.Int64},
    )
    query = pl.read_csv(
        data_dir / "query.csv", separator="\t", schema_overrides={"query_id": pl.Int64}
    )

    for filename, table, keys, value_column in (
        ("product.csv", product, ("product_id",), "product_name"),
        ("label.csv", label, ("query_id", "product_id"), "label"),
        ("query.csv", query, ("query_id",), "query"),
    ):
        missing = [column for column in (*keys, value_column) if column not in table.columns]
        if missing:
            raise ValueError(f"{filename}: missing required columns: {', '.join(missing)}")
        
        null_columns = [
            column
            for column in (*keys, value_column)
            if table.get_column(column).has_nulls()
        ]

        if null_columns:
            raise ValueError(
                f"{filename}: null values in required columns: "
                f"{', '.join(null_columns)}"
            )
        
        
    for filename, table, key in (
        ("product.csv", product, "product_id"),
        ("query.csv", query, "query_id"),
    ):
        if table.get_column(key).is_duplicated().any():
            raise ValueError(f"{filename}: duplicate {key} values")

        unknown = label.select(key).join(table.select(key), on=key, how="anti")
        if not unknown.is_empty():
            examples = unknown.get_column(key).unique().sort().head(5).to_list()
            raise ValueError(f"label.csv: unknown {key} values in {filename}: {examples}")

    return product, label, query
