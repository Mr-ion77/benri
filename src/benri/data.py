import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def split_df(df, split_by):
    
    df_list = []
    labels = []

    for element in df[split_by].unique():
        df_list.append( df[ df[split_by] == element ] )
        labels.append(str(element))
    
    return df_list, labels

def aggregate_and_save_top_configs(df, group_cols, value_column, table_dir, n=3):
    """Aggregate results by hyperparameter columns and save aggregated + top-n CSVs.

    Args:
        df: DataFrame or convertible sequence of dicts/rows.
        group_cols: list of columns to group by.
        value_column: the column to compute median and std for.
        table_dir: Path where CSVs will be saved.
        n: number of top configurations to save (based on median descending).

    Returns:
        (agg, top_n) DataFrames for aggregated and top-n results.
    """
    # Prepare table dir
    if isinstance(table_dir, str):
        table_dir = Path(table_dir)
        
    table_dir.mkdir(parents=True, exist_ok=True)

    if df is None or len(df) == 0:
        print("df is empty — nothing to aggregate or plot.")
        return None, None

    # Ensure DataFrame
    if not isinstance(df, pd.DataFrame):
        try:
            df = pd.DataFrame(df)
        except Exception:
            print("Could not convert df to DataFrame.")
            return None, None

    # Compute median and std for each grouping tuple
    agg = df.groupby(group_cols)[value_column].agg(['median', 'std']).reset_index()
    agg['median_std'] = agg.apply(lambda r: f"{r['median']:.4f} ± {r['std']:.4f}", axis=1)

    # Save aggregated table
    csv_path = table_dir / f"aggregated_{value_column}.csv"
    agg.to_csv(csv_path, index=False)
    print(f"Saved aggregated results to {csv_path}")

    # Label for display
    agg['label'] = agg[group_cols].astype(str).agg(lambda x: ', '.join(x), axis=1)

    # Select top-n
    agg_sorted = agg.sort_values(by='median', ascending=False).reset_index(drop=True)
    top_n = agg_sorted.head(n)
    top_csv = table_dir / f"top_{n}_{value_column}.csv"
    top_n.to_csv(top_csv, index=False)
    print(f"Saved top {n} configurations to {top_csv}")

    # Print concise view
    try:
        print(top_n[group_cols + ['median', 'std']].to_string(index=False))
    except Exception:
        print(top_n.to_string(index=False))

    return agg, top_n