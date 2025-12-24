import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def split_df(df, split_by):
    
    df_list = []
    labels = []
    print(df[split_by].unique())
    for element in df[split_by].unique():
        df_list.append( df[ df[split_by] == element ] )
        labels.append(str(element))
    
    return df_list, labels

def aggregate_and_save_top_configs(df_results, graph_columns, table_dir, n=3):
    """Aggregate results by hyperparameter columns and save aggregated + top-n CSVs.

    Args:
        df_results: DataFrame or convertible sequence of dicts/rows.
        graph_columns: list where last element is the value column, others are group cols.
        table_dir: Path where CSVs will be saved.
        n: number of top configurations to save (based on median descending).

    Returns:
        (agg, top_n) DataFrames for aggregated and top-n results.
    """
    # Prepare table dir
    table_dir.mkdir(parents=True, exist_ok=True)

    # Determine columns
    value_column = graph_columns[-1]
    group_cols = graph_columns[:-1]

    if df_results is None or len(df_results) == 0:
        print("df_results is empty — nothing to aggregate or plot.")
        return None, None

    # Ensure DataFrame
    if not isinstance(df_results, pd.DataFrame):
        try:
            df_results = pd.DataFrame(df_results)
        except Exception:
            print("Could not convert df_results to DataFrame.")
            return None, None

    # Compute median and std for each grouping tuple
    agg = df_results.groupby(group_cols)[value_column].agg(['median', 'std']).reset_index()
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