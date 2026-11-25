import polars as pl


def clean_labels(
    labels: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Clean relevance judgments by (query_id, product_id).

    Args:
        labels: DataFrame with non-null [query_id, product_id, label] columns.

    Returns:
        (cleaned, conflicts). Cleaned keeps the first row per single-label pair
        and adds relevance (Exact=2, Partial=1, Irrelevant=0). Conflicts keeps
        all original rows of pairs with multiple labels, including duplicates.

    Raises:
        ValueError: Null in a required column.
        pl.exceptions.InvalidOperationError: Unknown label.
    """
    required_columns = ["query_id", "product_id", "label"]

    if labels.select(pl.any_horizontal(pl.col(required_columns).is_null()).any()).item():
        raise ValueError("Required columns must not contain null values")
    
    pair_keys = ["query_id", "product_id"]
    label_mapping = {
        "Exact": 2,
        "Partial": 1,
        "Irrelevant": 0,
    }
    unique_labels = labels.unique(subset=[*pair_keys, "label"], 
                                  keep="first", 
                                  maintain_order=True).with_columns(
                                     pl.col("label")
                                     .replace_strict(label_mapping)
                                     .alias("relevance")
                                  )

    conflicting_pairs = (
        unique_labels.group_by(pair_keys)
        .len()
        .filter(pl.col("len") > 1)
        .select(pair_keys)
    )
    labels_without_conflicts = unique_labels.join(
        conflicting_pairs, on=pair_keys, how="anti"
    )
    labels_with_conflicts = labels.join(
        conflicting_pairs, on=pair_keys, how="inner"
    )

    return labels_without_conflicts, labels_with_conflicts
