import pandas as pd
import numpy as np
import os
import re
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
from typing import List

def save_to_csv(dataframe, save_path=None):
    """Save the comparison dataframe to CSV into given dir"""
    prefix = "anime_score_comparison"
    filename = f"{prefix}.csv"

    if save_path:
        save_path = os.path.expanduser(save_path)
        os.makedirs(save_path, exist_ok=True)
        filepath = os.path.join(save_path, filename)
    else:
        filepath = filename

    dataframe.to_csv(filepath, index=False, encoding="utf-8")
    print(f"CSV saved to: {filepath}\n")
    return filepath


def print_section_header(title, width=80):
    """Print a formatted section header"""
    print("=" * width)
    print(title)
    print("=" * width)
    print()


def print_metadata(comparison, valid_stats, valid_ratings):
    """Print metadata section"""
    print_section_header("METADATA")
    print(f"Total anime analyzed:        {len(comparison):,}")
    print(f"Anime with stats:            {len(valid_stats):,}")
    print(f"Anime with ratings:          {len(valid_ratings):,}")
    print(f"Coverage stats (%):          {len(valid_stats)/len(comparison)*100:.2f}%")
    print(f"Coverage ratings (%):        {len(valid_ratings)/len(comparison)*100:.2f}%")
    print()


def print_key_results(mean_distance_stats, mean_distance_ratings):
    """Print key results section"""
    print_section_header("KEY RESULTS")
    print(f"Mean distance (stats_df):    {mean_distance_stats:.6f}")
    print(f"Mean distance (ratings_df):  {mean_distance_ratings:.6f}")
    print(f"Absolute difference:         {abs(mean_distance_stats - mean_distance_ratings):.6f}")

    winner = "stats_df" if mean_distance_stats < mean_distance_ratings else "ratings_df"
    improvement = abs(mean_distance_stats - mean_distance_ratings) / max(mean_distance_stats, mean_distance_ratings) * 100

    print(f"Winner:                      {winner}")
    print(f"Improvement:                 {improvement:.2f}%")
    print()


def print_distribution_for_dataset(name, data, distance_col, labels, bins):
    """Print distribution for a single dataset"""
    print(f"{name}:")
    dist = pd.cut(data[distance_col], bins=bins, labels=labels).value_counts().sort_index()

    for label in labels:
        count = dist.get(label, 0)
        pct = (count / len(data) * 100) if len(data) > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"  {label: >10}: {count:>6,} ({pct:>5.1f}%) {bar}")
    print()


def print_distance_distribution(valid_stats, valid_ratings):
    """Print distance distribution section"""
    bins = [0, 0.01, 0.05, 0.1, 0.5, 1.0, float("inf")]
    labels = ["0-0.01", "0.01-0.05", "0.05-0.1", "0.1-0.5", "0.5-1.0", ">1.0"]

    print_section_header("DISTANCE DISTRIBUTION")
    print_distribution_for_dataset("STATS_DF", valid_stats, "distance_stats", labels, bins)
    print_distribution_for_dataset("RATINGS_DF", valid_ratings, "distance_ratings", labels, bins)


def print_detailed_report(comparison, valid_stats, valid_ratings, mean_distance_stats, mean_distance_ratings):
    """Print comprehensive analysis report to console"""
    print_metadata(comparison, valid_stats, valid_ratings)
    print_key_results(mean_distance_stats, mean_distance_ratings)
    print_distance_distribution(valid_stats, valid_ratings)


def quick_profile(df: pd.DataFrame, name: str) -> None:
    """
    Generate and print a quick diagnostic profile about
    duplicated and missing values of a given pandas DataFrame.

    This function provides:
      - Dataset dimensions (rows and columns)
      - Estimated memory usage
      - Count of missing values per column
      - Number of duplicated rows

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to analyze.
    name : str
        A descriptive name for the dataset, used in the printed header.

    Returns
    -------
    None
        Prints a formatted summary to the console.
    """

    print(f"\n{'='*70}")
    print(f"{name.upper():^70}")
    print(f"{'='*70}")

    # Shape + memory
    n_rows, n_cols = df.shape
    memory_mb = df.memory_usage(deep=True).sum() / 1024**2
    print(f"Shape: {n_rows:,} rows × {n_cols} columns")
    print(f"Memory usage: {memory_mb:.2f} MB")

    # Missing values
    missing = df.isnull().sum()
    total_missing = missing.sum()

    if total_missing > 0:
        print("\nMissing Values:")
        print(missing[missing > 0].sort_values(ascending=False))
    else:
        print("\nNo missing values!")

    # Duplicates
    dupes = df.duplicated().sum()
    if dupes > 0:
        print(f"\nDuplicated Rows: {dupes}")
    else:
        print("\nNo duplicated rows!")

    print(f"\n%_missing = {(total_missing / (n_rows * n_cols) * 100):.3f}")
    print(f"n_dupes   = {dupes}")


def downcast_df(df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
    """
    Optimize a DataFrame by downcasting numeric types and converting columns.

    This function performs the following optimizations:
      - Downcasts integers and floats to smallest possible types
      - Converts object columns to datetime when >80% contain valid dates
      - Converts low-cardinality object columns (<50% unique) to category

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to optimize.
    inplace : bool, optional
        If True, modify df in place. If False, return a copy, by default False.

    Returns
    -------
    pd.DataFrame
        Optimized DataFrame. If inplace=True, returns the modified input DataFrame.
    """
    if not inplace:
        df = df.copy()

    for col in df.columns:
        col_type = df[col].dtype

        # Integer downcast
        if pd.api.types.is_integer_dtype(col_type):
            df[col] = pd.to_numeric(df[col], downcast="integer")

        # Float downcast
        elif pd.api.types.is_float_dtype(col_type):
            df[col] = pd.to_numeric(df[col], downcast="float")

        # Handle object columns
        elif col_type == "object":
            # Skip if column is empty or all null
            col_series = df[col]
            if col_series.isna().sum() == len(df):
                continue

            # Attempt datetime conversion (only if column name suggests dates)
            if any(
                date_hint in col.lower()
                for date_hint in ["date", "time", "timestamp", "dt"]
            ):
                converted = pd.to_datetime(col_series, errors="coerce")
                valid_count = converted.notna().sum()
                if valid_count / len(df) > 0.8:
                    df[col] = converted
                    continue

            # Don't convert in other cases

    return df


def parse_list(s: str) -> List[str]:
    """
    Extract items from a string representation of a list using regex.

    Handles strings like: "['item1', 'item2']" or '["item1", "item2"]'

    Parameters
    ----------
    s : str
        String representation of a list with quoted items.

    Returns
    -------
    List[str]
        List of extracted items. Returns empty list if input is invalid,
        empty, or NaN.
    """
    if pd.isna(s) or s == "[]" or not s:
        return []

    # Find all quoted strings
    matches = re.findall(r'["\']([^"\']+)["\']', s)
    return matches


def add_count_on_vertical_bars(ax, fmt='{:,.0f}', padding=5, fontsize=9) -> None:
    """
    Add labels on top of VERTICAL bar chart bars using direct vectorization.
    Assumes a single data series (ax.containers[0]).
    """
    if not ax.containers:
        return

    # Direct access to the first container
    container = ax.containers[0]
    
    # Vectorized labeling
    ax.bar_label(
        container, 
        fmt=fmt, 
        padding=padding, 
        fontsize=fontsize, 
        fontweight='bold'
    )
    
    # Calculate Y-axis limit using NumPy
    if len(container.datavalues) > 0:
        max_height = np.max(container.datavalues)
        ax.set_ylim(0, max_height * 1.20)


def add_count_on_horizontal_bars(ax, fmt='{:,.0f}', padding=5, fontsize=10) -> None:
    """
    Add labels to the right of HORIZONTAL bar chart bars using direct vectorization.
    Assumes a single data series (ax.containers[0]).
    """
    if not ax.containers:
        return

    # 1. Accesso diretto
    container = ax.containers[0]

    # 2. Etichettatura vettorizzata
    ax.bar_label(
        container, 
        fmt=fmt, 
        padding=padding, 
        fontsize=fontsize, 
        fontweight='bold'
    )
        
    # 3. Calcolo del limite X usando NumPy
    if len(container.datavalues) > 0:
        max_width = np.max(container.datavalues)
        ax.set_xlim(0, max_width * 1.15)