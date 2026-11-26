import polars as pl
import numpy as np


def split_queries(
    query_groups: pl.DataFrame,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> pl.DataFrame:
    """Split queries into disjoint group-aware subsets.
    query_groups = pl.DataFrame({
        "query_id": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "group_id": [10, 10, 30, 40, 50, 60, 70, 80, 90, 100],
    })
    
    OUT:
    {
        "query_id": pl.Int64,
        "group_id": pl.Int64,
        "split": pl.String, train/val/test
    }
    """
    required_columns = ("query_id", "group_id")
    missing = [column for column in required_columns
               if column not in query_groups.columns]
    if missing:
        raise ValueError(f"query_groups: missing required columns: {', '.join(missing)}")
    
    if not (0 < train_ratio < 1 and 0 < val_ratio < 1):
        raise ValueError("Ratios must be between 0 and 1")
    
    if train_ratio + val_ratio >= 1:
        raise ValueError("Train and validation ratios must sum to less than 1")
    
    if query_groups.get_column("query_id").n_unique() != query_groups.height:
        raise ValueError("query_id must be primary key")
    
    has_null_query = query_groups["query_id"].is_null().any()
    has_null_group = query_groups["group_id"].is_null().any()
    if has_null_query or has_null_group:
        raise ValueError("null in query_id or group_id")

    group_ids = query_groups.get_column("group_id")
    unique_group_ids = group_ids.unique()
    
    n_groups = group_ids.n_unique()
    n_train = int(n_groups * train_ratio)
    n_val = int(n_groups * val_ratio)
    
    if n_train == 0: raise ValueError("train groups = 0")
    if n_val == 0: raise ValueError("val groups = 0")
    if n_train + n_val == n_groups: raise ValueError("test groups = 0")
    
    perm_group_ids = np.random.default_rng(seed).permutation(sorted(unique_group_ids))
    train_ids = set(perm_group_ids[ : n_train])
    val_ids   = set(perm_group_ids[n_train : n_train + n_val])

    def get_split(id):
        if id in train_ids: return 'train'
        elif id in val_ids: return 'val'
        else: return 'test'
        
    split_col = pl.Series('split', 
                          [get_split(id) for id in group_ids],
                          dtype=pl.String)
    
    return (query_groups
            .select("query_id", "group_id")
            .with_columns(split_col)
            .sort("query_id"))
    
    
    
    
    