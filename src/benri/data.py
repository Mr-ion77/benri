import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy import stats
import itertools

def split_df(df, split_by):
    
    df_list = []
    labels = []

    for element in df[split_by].unique():
        df_list.append( df[ df[split_by] == element ] )
        labels.append(str(element))
    
    return df_list, labels

def aggregate_and_save_top_configs(df, group_cols, value_column, table_dir, n=10):
    """Aggregate results by hyperparameter columns and save aggregated + top-n CSVs.

    Args:
        df: DataFrame or convertible sequence of dicts/rows.
        group_cols: list of columns to group by.
        value_column: the column to compute mean, median and std for.
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

    # Compute mean, median and std for each grouping tuple
    agg = df.groupby(group_cols)[value_column].agg(['mean', 'median', 'std']).reset_index()
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
        print(top_n[group_cols + ['mean', 'median', 'std']].to_string(index=False))
    except Exception:
        print(top_n.to_string(index=False))

    return agg, top_n


def compare_group_distributions(df, group_cols, value_col='test_auc', alpha=0.05):
    """
    Groups a DataFrame, tests for normality (Shapiro-Wilk), and applies 
    conditional one-tailed pairwise tests (Welch's t-test or Mann-Whitney U)
    with significance stars.
    """
    def get_stars(p):
        if p < 0.001: return "***"
        if p < 0.01: return "**"
        if p < 0.05: return "*"
        return ""

    print(f"--- Grouping by: {group_cols} | Evaluating: '{value_col}' ---")
    groups = df.groupby(group_cols)
    
    # 1. Normality Testing
    normality_results = {}
    print("\n--- Shapiro-Wilk Normality Test ---")
    
    for name, group in groups:
        data = group[value_col].dropna()
        
        if len(data) < 3:
            print(f"Group {name}: Not enough data (n={len(data)}). Defaulting to non-normal.")
            normality_results[name] = False
            continue
            
        stat, p_value = stats.shapiro(data)
        is_normal = p_value > alpha
        normality_results[name] = is_normal
        
        print(f"Group {name}: p-value = {p_value:.4f} -> Normal: {is_normal}")

    # 2. Pairwise Hypothesis Testing
    print("\n--- Pairwise Hypothesis Testing (One-Tailed) ---")
    group_keys = list(groups.groups.keys())
    pairs = list(itertools.combinations(group_keys, 2))

    if not pairs:
        print("Error: Not enough groups to perform pairwise testing.")
        return normality_results

    for g1_key, g2_key in pairs:
        data1 = groups.get_group(g1_key)[value_col].dropna()
        data2 = groups.get_group(g2_key)[value_col].dropna()
        
        # Conditional selection
        if normality_results.get(g1_key, False) and normality_results.get(g2_key, False):
            test_name = "Welch's t-test"
            _, p_val_g1_greater = stats.ttest_ind(data1, data2, equal_var=False, alternative='greater')
            _, p_val_g2_greater = stats.ttest_ind(data2, data1, equal_var=False, alternative='greater')
        else:
            test_name = "Mann-Whitney U test"
            _, p_val_g1_greater = stats.mannwhitneyu(data1, data2, alternative='greater')
            _, p_val_g2_greater = stats.mannwhitneyu(data2, data1, alternative='greater')

        stars1 = get_stars(p_val_g1_greater)
        stars2 = get_stars(p_val_g2_greater)

        print(f"\nComparing {g1_key} vs {g2_key} using {test_name}:")
        print(f"  H1: {g1_key} > {g2_key} -> p-value = {p_val_g1_greater:.4e} {stars1}")
        print(f"  H1: {g2_key} > {g1_key} -> p-value = {p_val_g2_greater:.4e} {stars2}")

    return normality_results