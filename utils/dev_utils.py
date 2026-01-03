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
    print(
        f"Absolute difference:         {abs(mean_distance_stats - mean_distance_ratings):.6f}"
    )

    winner = "stats_df" if mean_distance_stats < mean_distance_ratings else "ratings_df"
    improvement = (
        abs(mean_distance_stats - mean_distance_ratings)
        / max(mean_distance_stats, mean_distance_ratings)
        * 100
    )

    print(f"Winner:                      {winner}")
    print(f"Improvement:                 {improvement:.2f}%")
    print()


def print_distribution_for_dataset(name, data, distance_col, labels, bins):
    """Print distribution for a single dataset."""
    print(f"{name}:")
    
    s_labels = pd.Series(labels)
    
    dist = (
        pd.cut(data[distance_col], bins=bins, labels=s_labels)
        .value_counts()
        .sort_index()
        .reindex(s_labels, fill_value=0)
    )

    total = len(data)
    pcts = (dist / total * 100) if total > 0 else pd.Series(0.0, index=s_labels)

    bar_counts = (pcts / 2).astype(int)
    bars = pd.Series("█", index=s_labels.index).str.repeat(bar_counts)

    fmt_labels = s_labels.str.rjust(10)
    fmt_counts = dist.map('{:>6,}'.format)
    fmt_pcts = pcts.map('({:>5.1f}%)'.format)

    final_lines = (
        "  " + fmt_labels + ": " + fmt_counts + " " + fmt_pcts + " " + bars
    )

    print(final_lines.str.cat(sep="\n"))
    print()


def print_distance_distribution(valid_stats, valid_ratings):
    """Print distance distribution section"""
    bins = [0, 0.01, 0.05, 0.1, 0.5, 1.0, float("inf")]
    labels = ["0-0.01", "0.01-0.05", "0.05-0.1", "0.1-0.5", "0.5-1.0", ">1.0"]

    print_section_header("DISTANCE DISTRIBUTION")
    print_distribution_for_dataset(
        "STATS_DF", valid_stats, "distance_stats", labels, bins
    )
    print_distribution_for_dataset(
        "RATINGS_DF", valid_ratings, "distance_ratings", labels, bins
    )


def print_detailed_report(
    comparison, valid_stats, valid_ratings, mean_distance_stats, mean_distance_ratings
):
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

    n_rows, n_cols = df.shape
    memory_mb = df.memory_usage(deep=True).sum() / 1024**2
    print(f"Shape: {n_rows:,} rows × {n_cols} columns")
    print(f"Memory usage: {memory_mb:.2f} MB")

    missing = df.isnull().sum()
    total_missing = missing.sum()

    if total_missing > 0:
        print("\nMissing Values:")
        print(missing[missing > 0].sort_values(ascending=False))
    else:
        print("\nNo missing values!")

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
    
    int_cols = df.select_dtypes(include=['integer']).columns
    df[int_cols] = df[int_cols].apply(pd.to_numeric, downcast="integer")

    float_cols = df.select_dtypes(include=['float']).columns
    df[float_cols] = df[float_cols].apply(pd.to_numeric, downcast="float")
    
    object_cols = df.select_dtypes(include=['object']).columns
    
    non_empty_obj_cols = object_cols[df[object_cols].notna().any()]
    
    date_hints = ["date", "time", "timestamp", "dt"]
    col_names_lower = pd.Series(non_empty_obj_cols).str.lower()
    
    pattern = '|'.join(map(re.escape, date_hints))
    has_date_hint = col_names_lower.str.contains(pattern, case=False, na=False)
    date_candidate_cols = non_empty_obj_cols[has_date_hint]
    
    if len(date_candidate_cols) > 0:
        converted = df[date_candidate_cols].apply(pd.to_datetime, errors="coerce")
        
        valid_ratios = converted.notna().sum() / len(df)
        cols_to_convert = valid_ratios[valid_ratios > 0.8].index
        
        df[cols_to_convert] = converted[cols_to_convert]
    
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

    matches = re.findall(r'["\']([^"\']+)["\']', s)
    return matches


def add_count_on_vertical_bars(ax, fmt="{:,.0f}", padding=5, fontsize=9) -> None:
    """
    Add labels on top of VERTICAL bar chart bars using direct vectorization.
    Assumes a single data series (ax.containers[0]).
    """
    if not ax.containers:
        return

    container = ax.containers[0]

    ax.bar_label(
        container, fmt=fmt, padding=padding, fontsize=fontsize, fontweight="bold"
    )

    if len(container.datavalues) > 0:
        max_height = np.max(container.datavalues)
        ax.set_ylim(0, max_height * 1.20)


def add_count_on_horizontal_bars(ax, custom_labels=None, fmt="{:,.0f}"):
    """
    Adds labels to the end of horizontal bars in a Matplotlib axes.

    If 'custom_labels' is provided, it uses those strings. Otherwise, it formats
    the numerical values of the bars using the provided 'fmt' string.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object containing the horizontal bar plot.
    custom_labels : array-like, optional
        A list or array of strings to use as labels. If None, values are inferred 
        from the bar lengths.
    fmt : str, default="{:,.0f}"
        The format string to apply to the values if custom_labels is None.
    """
    if custom_labels is None:
        custom_labels = (
            pd.Series(ax.containers[0].datavalues)
            .map(fmt.format)
            .values
        )

    ax.bar_label(
        ax.containers[0],
        labels=custom_labels,
        padding=3,
        fontsize=10,
        fontweight="bold"
    )

