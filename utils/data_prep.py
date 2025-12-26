"""
Data Preparation Utilities for Anime Analysis
==============================================

Reusable functions for preparing and processing anime-related datasets.
"""

import pandas as pd
import numpy as np

from .dev_utils import parse_list


def prepare_character_favorites(characters_df, favs_df):
    """
    Calculate total character favorites (platform + user favorites).

    Parameters
    ----------
    characters_df : pd.DataFrame
        Character details dataset
    favs_df : pd.DataFrame
        User favorites dataset

    Returns
    -------
    pd.DataFrame
        Character data with total_favs column
    """
    char_favs = favs_df[favs_df["fav_type"] == "character"].copy()
    char_fav_counts = char_favs.groupby("id").size().reset_index(name="user_favs")

    char_popularity = characters_df.merge(
        char_fav_counts, left_on="character_mal_id", right_on="id", how="left"
    )
    char_popularity["user_favs"] = char_popularity["user_favs"].fillna(0)
    char_popularity["total_favs"] = (
        char_popularity["favorites"].fillna(0) + char_popularity["user_favs"]
    )

    return char_popularity[["character_mal_id", "name", "total_favs"]]


def explode_genres(df, genre_column="genres"):
    """
    Parse and explode genre column into separate rows.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with genre column
    genre_column : str, default="genres"
        Name of column containing comma-separated genres

    Returns
    -------
    pd.DataFrame
        DataFrame with genres_list column exploded
    """
    df = df.copy()
    df["genres_list"] = df[genre_column].apply(parse_list)
    exploded = df.explode("genres_list")

    return exploded[exploded["genres_list"] != ""]


def prepare_engagement_data(
    details_df, stats_df, min_users=0, include_rates=True, include_episodes=False
):
    """
    Prepare base anime engagement data with started users and rates.

    Parameters
    ----------
    details_df : pd.DataFrame
        Anime details dataset
    stats_df : pd.DataFrame
        Anime statistics dataset
    min_users : int, default=0
        Minimum started users threshold
    include_rates : bool, default=True
        Whether to calculate completion/dropout rates
    include_episodes : bool, default=False
        Whether to include episode count column

    Returns
    -------
    pd.DataFrame
        Merged dataset with engagement metrics
    """
    cols = ["mal_id", "title", "type", "genres", "score"]
    if include_episodes:
        cols.append("episodes")

    anime_engagement = details_df[cols].merge(
        stats_df[["mal_id", "watching", "completed", "on_hold", "dropped"]],
        on="mal_id",
        how="inner",
    )

    anime_engagement["started_users"] = (
        anime_engagement["watching"]
        + anime_engagement["completed"]
        + anime_engagement["on_hold"]
        + anime_engagement["dropped"]
    )

    if min_users > 0:
        anime_engagement = anime_engagement[
            anime_engagement["started_users"] >= min_users
        ].copy()

    if include_rates:
        anime_engagement["completion_rate"] = (
            anime_engagement["completed"] / anime_engagement["started_users"] * 100
        )
        anime_engagement["dropout_rate"] = (
            anime_engagement["dropped"] / anime_engagement["started_users"] * 100
        )
        anime_engagement["watching_rate"] = (
            anime_engagement["watching"] / anime_engagement["started_users"] * 100
        )
        anime_engagement["on_hold_rate"] = (
            anime_engagement["on_hold"] / anime_engagement["started_users"] * 100
        )

    return anime_engagement


def aggregate_by_dimension(data, groupby_col, agg_cols):
    """
    Generic aggregation function for completion/dropout analysis.

    Parameters
    ----------
    data : pd.DataFrame
        Data with engagement metrics
    groupby_col : str
        Column to group by
    agg_cols : dict
        Aggregation dictionary

    Returns
    -------
    pd.DataFrame
        Aggregated statistics
    """
    stats = data.groupby(groupby_col, observed=True).agg(agg_cols).reset_index()

    # Flatten multi-level columns if present
    if isinstance(stats.columns, pd.MultiIndex):
        stats.columns = ["_".join(col).strip("_") for col in stats.columns.values]

    return stats


def bin_episodes(df, bins=None, labels=None):
    """
    Create episode count categories.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'episodes' column
    bins : list, optional
        Custom bin edges
    labels : list, optional
        Custom labels for bins

    Returns
    -------
    pd.DataFrame
        DataFrame with 'episode_category' column added
    """
    if bins is None:
        bins = [1, 2, 13, 25, 51, 101, 501, 3000]

    if labels is None:
        labels = [
            "1 ep\n(Movie/Special)",
            "2-12 eps\n(1 cour)",
            "13-24 eps\n(2 cour)",
            "25-50 eps\n(Long)",
            "51-100 eps\n(Very Long)",
            "101-500 eps\n(Ongoing)",
            "501+ eps\n(Mega)",
        ][: len(bins) - 1]

    df = df.copy()
    df = df[(df["episodes"].notna()) & (df["episodes"] > 0)]
    df["episode_category"] = pd.cut(
        df["episodes"], bins=bins, labels=labels, include_lowest=True
    )

    return df


def prepare_episode_score_data(
    details_df, stats_df, max_episodes=500, bins=None, labels=None
):
    """
    Prepare anime data with episode categorization for score analysis.

    Parameters
    ----------
    details_df : pd.DataFrame
        Anime details with 'mal_id', 'episodes', 'score', 'title' columns
    stats_df : pd.DataFrame
        Anime statistics with 'mal_id' column
    max_episodes : int, default=500
        Maximum episode count to include (filters outliers)
    bins : list, optional
        Custom bin edges for episode categories.
        Default:  [0, 1, 12, 24, 50, 100, 500]
    labels : list, optional
        Custom labels for episode categories.
        Default: ["Single", "2-12", "13-24", "25-50", "51-100", "100+"]

    Returns
    -------
    pd.DataFrame
        Merged and categorized data with 'episode_category' column
    """
    # Set defaults
    if bins is None:
        bins = [0, 1, 12, 24, 50, 100, 500]

    if labels is None:
        labels = ["Single", "2-12", "13-24", "25-50", "51-100", "100+"]

    # Merge datasets
    df = details_df.merge(stats_df, on="mal_id", how="inner")

    # Filter valid data
    df = df[
        (df["episodes"].notna())
        & (df["episodes"] > 0)
        & (df["episodes"] < max_episodes)
        & (df["score"].notna())
    ].copy()

    # Categorize by episode count
    df["episode_category"] = pd.cut(
        df["episodes"], bins=bins, labels=labels, include_lowest=True
    )

    return df


def prepare_completion_drop_data(stats_df, min_members=100):
    """
    Prepare completion and drop rate data for distribution analysis.

    Parameters
    ----------
    stats_df :  pd.DataFrame
        Anime statistics with watching, completed, dropped, on_hold columns
    min_members : int, default=100
        Minimum total members required for inclusion

    Returns
    -------
    pd.DataFrame
        DataFrame with completion_rate and drop_rate columns
    """
    stats_filtered = stats_df.copy()

    # Calculate total members
    stats_filtered["total_members"] = (
        stats_filtered["watching"]
        + stats_filtered["completed"]
        + stats_filtered["dropped"]
        + stats_filtered["on_hold"]
    )

    # Filter by minimum members
    stats_filtered = stats_filtered[
        stats_filtered["total_members"] >= min_members
    ].copy()

    # Calculate rates
    stats_filtered["completion_rate"] = (
        stats_filtered["completed"] / stats_filtered["total_members"]
    ) * 100
    stats_filtered["drop_rate"] = (
        stats_filtered["dropped"] / stats_filtered["total_members"]
    ) * 100

    return stats_filtered
