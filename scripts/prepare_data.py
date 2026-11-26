import hashlib
import json
from pathlib import Path

import polars as pl

from rankoon.core.config import get_settings
from rankoon.data.cleaning import clean_labels
from rankoon.data.grouping import build_query_groups
from rankoon.data.loader import load_annotated_duplications, load_wands
from rankoon.data.splitting import split_queries


def prepare_data(
    raw_dir: Path,
    annotations_path: Path,
    processed_dir: Path,
    splits_dir: Path,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> None:
    """Prepare and persist the WANDS dataset."""
    products, raw_labels, queries = load_wands(raw_dir)
    labels, conflicts = clean_labels(raw_labels)
    duplicate_pairs = load_annotated_duplications(annotations_path)
    query_groups = build_query_groups(queries, duplicate_pairs)
    splits = split_queries(query_groups, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed)

    if not splits["query_id"].sort().equals(queries["query_id"].sort()):
        raise ValueError("Split manifest must contain each source query exactly once")
    split_names = ("train", "val", "test")
    if set(splits["split"]) != set(split_names):
        raise ValueError("Split manifest must contain all three splits: train, val, test")

    splits_per_group = splits.group_by("group_id").agg(pl.col("split").n_unique())
    if (splits_per_group["split"] != 1).any():
        raise ValueError("A query group appears in multiple splits")

    split_by_query = dict(splits.select("query_id", "split").iter_rows())
    if any(split_by_query[first] != split_by_query[second]
           for first, second in duplicate_pairs.iter_rows()):
        raise ValueError("A confirmed duplicate pair appears in different splits")
    
    if not set(labels["query_id"].unique()).issubset(split_by_query):
        raise ValueError("Cleaned labels reference queries missing from the split manifest")
    
    if not set(labels["product_id"].unique()).issubset(products["product_id"]):
        raise ValueError("Cleaned labels reference unknown products")

    label_pair = ("query_id", "product_id")
    if labels.n_unique(subset=label_pair) != labels.height:
        raise ValueError("Cleaned labels contain duplicate query-product pairs")
    if not set(labels["relevance"].unique()).issubset({0, 1, 2}):
        raise ValueError("Cleaned labels contain invalid relevance values")

    counts = splits.group_by("split", maintain_order=True).agg(
        pl.len().alias("queries"),
        pl.col("group_id").n_unique().alias("groups"),
    )
    split_counts = {
        split: {"queries": queries_count, "groups": groups_count}
        for split, queries_count, groups_count in counts.iter_rows()
    }
    input_paths = {
        "product.csv": raw_dir / "product.csv",
        "query.csv": raw_dir / "query.csv",
        "label.csv": raw_dir / "label.csv",
        "annotations": annotations_path,
    }
    input_sha256 = {}
    for name, path in input_paths.items():
        with path.open("rb") as source:
            input_sha256[name] = hashlib.file_digest(source, "sha256").hexdigest()

    metadata = {
        "dataset": "WANDS",
        "schema_version": 1,
        "seed": seed,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "num_products": products.height,
        "num_queries": queries.height,
        "num_raw_labels": raw_labels.height,
        "num_cleaned_labels": labels.height,
        "num_conflicting_rows": conflicts.height,
        "num_conflicting_pairs": conflicts.n_unique(subset=label_pair),
        "num_duplicate_pairs": duplicate_pairs.height,
        "num_query_groups": query_groups.n_unique(subset="group_id"),
        "split_counts": {split: split_counts[split] for split in split_names},
        "input_sha256": input_sha256,
    }

    processed_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    products.write_parquet(processed_dir / "products.parquet")
    queries.write_parquet(processed_dir / "queries.parquet")
    labels.write_parquet(processed_dir / "labels.parquet")
    conflicts.write_parquet(processed_dir / "conflicts.parquet")
    
    manifest_path = splits_dir / "query_splits.csv"
    if manifest_path.exists():
        existing = pl.read_csv(manifest_path)
        if not existing.equals(splits):
            raise ValueError(
                "Generated split differs from the existing split manifest"
            )
    splits.write_csv(manifest_path)

    with (processed_dir / "metadata.json").open("w", encoding="utf-8") as output:
        json.dump(metadata, output, indent=2)
        output.write("\n")

    print(
        f"WANDS: {products.height} products, {queries.height} queries, "
        f"{labels.height} cleaned labels"
    )
    print(
        "Queries per split:",
        ", ".join(f"{split}={split_counts[split]['queries']}" for split in split_names),
    )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    settings = get_settings(root)
    prepare_data(
        raw_dir=settings.wands_data_dir,
        annotations_path=settings.wands_annots_path,
        processed_dir=settings.processed_dir,
        splits_dir=settings.splits_dir,
    )


if __name__ == "__main__":
    main()
