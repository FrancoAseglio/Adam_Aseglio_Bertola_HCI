import pandas as pd
import re
import warnings
from typing import List


def add_count_on_bars(ax, fontsize: int = 9, offset: int = 5) -> None:
    """Adds count labels on top of bar chart bars.

    Args:
        ax: Matplotlib axes object
        fontsize: Font size for labels (default: 9)
        offset: Vertical offset in points (default: 5)

    Labels are centered above each bar showing the bar's height as an integer.
    """
    for patch in ax.patches:
        height = patch.get_height()
        if height != 0:  # Skip zero-height bars
            ax.annotate(
                f"{int(height)}",
                (patch.get_x() + patch.get_width() / 2.0, height),
                ha="center",
                va="bottom",
                fontsize=fontsize,
                color="black",
                xytext=(0, offset),
                textcoords="offset points",
            )


def downcast_df(df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
    """
    Optimizes a DataFrame by downcasting and converting types.

    Optimizations:
    - Downcasts integers and floats to smallest possible types
    - Converts object columns to datetime when >80% valid dates
    - Converts low-cardinality object columns (<50%) to category

    Args:
        df: Input DataFrame
        inplace: If True, modify df in place (default: False)

    Returns:
        Optimized DataFrame (or modified in place if inplace=True)
    """
    if not inplace:
        df = df.copy()

    for col in df.columns:
        col_type = df[col].dtype

        # 1. Integer downcast
        if pd.api.types.is_integer_dtype(col_type):
            df[col] = pd.to_numeric(df[col], downcast="integer")

        # 2. Float downcast
        elif pd.api.types.is_float_dtype(col_type):
            df[col] = pd.to_numeric(df[col], downcast="float")

        # 3. Handle object columns
        elif col_type == "object":
            # Skip if column is empty or all null
            col_series = df[col]
            if col_series.isna().sum() == len(df):
                continue

            # Attempt datetime conversion (only if column name suggests dates)
            # This is much faster than trying on every object column
            if any(
                date_hint in col.lower()
                for date_hint in ["date", "time", "timestamp", "dt"]
            ):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    converted = pd.to_datetime(col_series, errors="coerce")
                    valid_count = converted.notna().sum()
                    if valid_count / len(df) > 0.8:
                        df[col] = converted
                        continue

            # Convert to category if low cardinality
            n_unique = col_series.nunique(dropna=True)
            if n_unique > 0:  # Avoid division by zero
                cardinality = n_unique / len(df)
                if cardinality < 0.5:
                    df[col] = col_series.astype("category")

    return df


def parse_list(s: str) -> List[str]:
    """Extract items from list string using regex.

    Handles strings like: "['item1', 'item2']" or '["item1", "item2"]'

    Args:
        s: String representation of a list

    Returns:
        List of extracted items (empty list if invalid/empty)
    """
    if pd.isna(s) or s == "[]" or not s:
        return []

    # Find all quoted strings
    matches = re.findall(r'["\']([^"\']+)["\']', s)
    return matches


def parse_list_series(series: pd.Series) -> pd.Series:
    """Vectorized version for parsing list strings in a Series.

    Args:
        series: Pandas Series containing list strings

    Returns:
        Series with parsed lists (each element is a list)
    """
    # Pre-compile regex
    pattern = re.compile(r'["\']([^"\']+)["\']')

    def parse_item(s) -> List[str]:
        if pd.isna(s) or s == "[]" or not s:
            return []
        return pattern.findall(s)

    return pd.Series(series.apply(parse_item), index=series.index)


def memory_usage_check(df: pd.DataFrame, detailed: bool = False) -> None:
    """Print memory usage of DataFrame, optionally by column.

    Args:
        df: DataFrame to analyze
        detailed: If True, show per-column breakdown
    """
    memory_mb = df.memory_usage(deep=True).sum() / 1024**2
    print(f"Total memory usage: {memory_mb:.2f} MB")

    if detailed:
        print("\nPer-column breakdown:")
        memory_by_col = df.memory_usage(deep=True) / 1024**2
        for col, mem in memory_by_col.items():
            col_name = str(col)
            if col_name == "Index":
                dtype_str = "Index"
            else:
                dtype_str = str(df[col].dtype)
            print(f"  {col_name:30s} {dtype_str:15s} {mem:8.2f} MB")


def optimize_df(df: pd.DataFrame, inplace: bool = False, show_memory: bool = True) -> pd.DataFrame:
    """Complete optimization pipeline for DataFrames.

    Args:
        df: Input DataFrame
        inplace: If True, modify in place
        show_memory: If True, print before/after memory usage

    Returns:
        Optimized DataFrame (modified df if inplace=True, otherwise new df)
    """
    if show_memory:
        print("Before optimization:")
        memory_usage_check(df)

    result = downcast_df(df, inplace=inplace)

    if show_memory:
        print("\nAfter optimization:")
        memory_usage_check(result)

    return result
