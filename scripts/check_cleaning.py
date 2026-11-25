from pathlib import Path

from rankoon.data import cleaning
import polars as pl

ROOT = Path(__file__).parent.parent
label = pl.read_csv(ROOT / r"data\raw\dataset\label.csv", separator='\t')

cleaned, conflicts = cleaning.clean_labels(label)

print("Raw rows:", label.height)
print("Cleaned rows:", cleaned.height)
print("Conflicting rows:", conflicts.height)

print(
    "Conflicting pairs:",
    conflicts.select("query_id", "product_id").unique().height,
)

print(cleaned.group_by("relevance").len())