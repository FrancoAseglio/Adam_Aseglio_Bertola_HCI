import pandas as pd
import re
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
from typing import List


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
            if any(
                date_hint in col.lower()
                for date_hint in ["date", "time", "timestamp", "dt"]
            ):
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


def add_count_on_bars(ax, fontsize: int = 9, offset: int = 5) -> None:
    """
    Add count labels on top of bar chart bars.
    Labels are centered above each bar showing the bar's height as an integer.
    Zero-height bars are skipped.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Matplotlib axes object containing the bar chart.
    fontsize : int, optional
        Font size for the count labels, by default 9.
    offset : int, optional
        Vertical offset in points for label placement, by default 5.

    Returns
    -------
    None
        Modifies the axes object in place by adding annotations.
    """
    max_height = 0

    for patch in ax.patches:
        height = patch.get_height()
        if height != 0:
            max_height = max(max_height, height)
            ax.annotate(
                f"{int(height):,}",
                (patch.get_x() + patch.get_width() / 2.0, height),
                ha="center",
                va="bottom",
                fontsize=fontsize,
                fontweight="bold",
                color="black",
                xytext=(0, offset),
                textcoords="offset points",
            )
    ax.set_ylim(0, max_height * 1.20)
