"""
Data Preparation Utilities for Anime Analysis
==============================================

Reusable functions for preparing and processing anime-related datasets.
"""

import pandas as pd
import numpy as np

from utils.geo_utils import get_country_code
from .dev_utils import parse_list


# ------------------------------
# GEOGRAPHIC DATA PREPARATION
# ------------------------------


def prepare_country_data(profiles_df):
    """
    Prepare geographic distribution data with country codes.
    Uses pandas apply and list expansion for cleaner syntax.
    """
    profiles_df = profiles_df.copy()
    profiles_df["year_label"] = pd.to_datetime(
        profiles_df["joined"], errors="coerce"
    ).dt.year

    mapped_locations = profiles_df["location"].map(get_country_code)

    loc_df = pd.DataFrame(
        mapped_locations.tolist(),
        columns=["country", "country_code"],
        index=profiles_df.index,
    )

    profiles_df["country"] = loc_df["country"]
    profiles_df["country_code"] = loc_df["country_code"]

    profiles_with_country = profiles_df[profiles_df["country_code"].notna()].copy()

    country_year_counts = (
        profiles_with_country.groupby(["country", "country_code", "year_label"])
        .size()
        .reset_index(name="user_count")
    )

    country_year_counts = country_year_counts.sort_values(["country", "year_label"])
    country_year_counts["cumulative_users"] = country_year_counts.groupby("country")[
        "user_count"
    ].cumsum()
    country_year_counts["log_cumulative"] = np.log10(
        country_year_counts["cumulative_users"] + 1
    )

    return country_year_counts


# ------------------------------
# STUDIO DATA PREPARATION
# ------------------------------


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

    if isinstance(stats.columns, pd.MultiIndex):
        stats.columns = stats.columns.map("_".join).str.strip("_")

    return stats


def prepare_studio_data(details_df, stats_df, min_anime_threshold=5):
    """
    Prepare studio performance data with productivity and quality metrics.

    Parameters
    ----------
    details_df : pd.DataFrame
        Anime details with 'mal_id', 'studios', 'score' columns
    stats_df : pd.DataFrame
        Anime statistics with 'mal_id' column
    min_anime_threshold :  int, default=5
        Minimum number of anime for a studio to be included

    Returns
    -------
    pd.DataFrame
        Studio analysis data with columns:  Studio, Avg Score, Count
        Sorted by Count descending
    """
    df = details_df.merge(stats_df, on="mal_id", how="inner")
    df["studios_list"] = df["studios"].astype(str).apply(parse_list)

    studio_df = df.explode("studios_list")
    studio_df = studio_df[
        (studio_df["studios_list"] != "") & (studio_df["studios_list"].notna())
    ].copy()

    studio_analysis = (
        studio_df.groupby("studios_list")
        .agg({"score": "mean", "mal_id": "count"})
        .round(2)
        .reset_index()
    )

    studio_analysis.columns = ["Studio", "Avg Score", "Count"]
    studio_analysis = studio_analysis[
        studio_analysis["Count"] >= min_anime_threshold
    ].copy()
    studio_analysis = studio_analysis.sort_values("Count", ascending=False)

    return studio_analysis


# ------------------------------
# EPISODE DATA PREPARATION
# ------------------------------


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
            "2-12 eps\n(1 season)",
            "13-24 eps\n(2 seasons)",
            "25-50 eps\n(3-4 seasons)",
            "51-100 eps\n(Long)",
            "101-500 eps\n(Very Long)",
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
    if bins is None:
        bins = [0, 1, 12, 24, 50, 100, 500]

    if labels is None:
        labels = ["Single", "2-12", "13-24", "25-50", "51-100", "100+"]

    df = details_df.merge(stats_df, on="mal_id", how="inner")

    df = df[
        (df["episodes"].notna())
        & (df["episodes"] > 0)
        & (df["episodes"] < max_episodes)
        & (df["score"].notna())
    ].copy()

    df["episode_category"] = pd.cut(
        df["episodes"], bins=bins, labels=labels, include_lowest=True
    )

    return df


# ------------------------------
# GENRE DATA PREPARATION
# ------------------------------


def fill_missing_years(df):
    """
    Fill missing years for each genre to avoid line gaps using vectorized operations.
    Creates a full Cartesian product (All Years x All Genres) and merges data.
    """

    all_years = pd.DataFrame(
        {"year": range(int(df["year"].min()), int(df["year"].max()) + 1)}
    )
    all_genres = pd.DataFrame({"genres_list": df["genres_list"].unique()})

    full_grid = all_years.merge(all_genres, how="cross")

    filled_df = full_grid.merge(df, on=["year", "genres_list"], how="left")

    filled_df["popularity"] = filled_df["popularity"].fillna(0)
    filled_df["anime_count"] = filled_df["anime_count"].fillna(0).astype(int)

    return filled_df


# ------------------------------
# CHARACTER DATA PREPARATION
# ------------------------------


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

    stats_filtered["total_members"] = (
        stats_filtered["watching"]
        + stats_filtered["completed"]
        + stats_filtered["dropped"]
        + stats_filtered["on_hold"]
    )

    stats_filtered = stats_filtered[
        stats_filtered["total_members"] >= min_members
    ].copy()
    stats_filtered["completion_rate"] = (
        stats_filtered["completed"] / stats_filtered["total_members"]
    ) * 100
    stats_filtered["drop_rate"] = (
        stats_filtered["dropped"] / stats_filtered["total_members"]
    ) * 100

    return stats_filtered


def prepare_viewing_status_data(stats_df):
    """
    Aggregate viewing status counts across all anime.

    Parameters
    ----------
    stats_df : pd.DataFrame
        Anime statistics with status columns

    Returns
    -------
    pd.DataFrame
        DataFrame with status and count columns, ordered for visualization
    """
    status_counts = {
        "on_hold": stats_df["on_hold"].sum(),
        "dropped": stats_df["dropped"].sum(),
        "watching": stats_df["watching"].sum(),
        "plan_to_watch": stats_df["plan_to_watch"].sum(),
        "completed": stats_df["completed"].sum(),
    }

    status_df = pd.DataFrame(
        [
            {"status": "on_hold", "count": status_counts["on_hold"]},
            {"status": "dropped", "count": status_counts["dropped"]},
            {"status": "watching", "count": status_counts["watching"]},
            {"status": "plan_to_watch", "count": status_counts["plan_to_watch"]},
            {"status": "completed", "count": status_counts["completed"]},
        ]
    )

    total = status_df["count"].sum()

    status_df["percentage"] = (status_df["count"] / total) * 100

    return status_df


def prepare_active_users_with_personas(profiles_df, min_anime=10):
    """
    Prepare active users dataset with persona assignments based on viewing behavior.

    Parameters
    ----------
    profiles_df : pd.DataFrame
        User profiles with viewing statistics
    min_anime : int, default=10
        Minimum total anime watched to be considered active

    Returns
    -------
    pd.DataFrame
        Active users with persona column assigned
    """
    profiles_analysis = profiles_df.copy()

    cols_to_convert = ["watching", "completed", "on_hold", "dropped", "plan_to_watch"]
    profiles_analysis[cols_to_convert] = (
        profiles_analysis[cols_to_convert]
        .astype(str)
        .replace(r",", "", regex=True)
        .astype(float)
    )

    profiles_analysis["total_anime"] = (
        profiles_analysis["completed"]
        + profiles_analysis["dropped"]
        + profiles_analysis["on_hold"]
        + profiles_analysis["watching"]
    )

    profiles_analysis["completion_ratio"] = profiles_analysis["completed"] / (
        profiles_analysis["total_anime"] + 1
    )
    profiles_analysis["drop_ratio"] = profiles_analysis["dropped"] / (
        profiles_analysis["total_anime"] + 1
    )

    active_users = profiles_analysis[
        profiles_analysis["total_anime"] >= min_anime
    ].copy()

    conditions = [
        (active_users["completion_ratio"] > 0.90) & (active_users["total_anime"] > 300),
        active_users["drop_ratio"] > 0.10,
        active_users["total_anime"] < 100,
        active_users["total_anime"] > 500,
    ]

    choices = ["Completionist", "Selective Critic", "Casual Viewer", "Hardcore Fan"]

    active_users["persona"] = np.select(
        conditions, choices, default="Balanced Enthusiast"
    )

    return active_users


def prepare_persona_genre_data(active_users, favs_df, details_df, top_n=15):
    """
    Prepare genre preference data by fan persona.

    Parameters
    ----------
    active_users : pd.DataFrame
        Users with persona assignments
    favs_df : pd.DataFrame
        User favorites dataset
    details_df : pd.DataFrame
        Anime details with genres
    top_n : int, default=15
        Number of top genres to include

    Returns
    -------
    pd.DataFrame
        Pivot table with genres as rows, personas as columns, percentages as values
    """
    user_favs = active_users[["username", "persona"]].merge(
        favs_df[favs_df["fav_type"] == "anime"], on="username", how="inner"
    )
    user_favs_genres = user_favs.merge(
        details_df[["mal_id", "genres"]], left_on="id", right_on="mal_id", how="left"
    )
    user_favs_genres["genres_list"] = user_favs_genres["genres"].apply(parse_list)

    persona_genre_df = user_favs_genres[["persona", "genres_list"]].explode(
        "genres_list"
    )
    persona_genre_df = persona_genre_df[
        persona_genre_df["genres_list"].notna()
        & (persona_genre_df["genres_list"] != "")
    ]
    persona_genre_df = persona_genre_df.rename(columns={"genres_list": "genre"})

    persona_genre_counts = (
        persona_genre_df.groupby(["persona", "genre"], observed=True)
        .size()
        .reset_index(name="count")
    )

    persona_totals = (
        persona_genre_counts.groupby("persona", observed=True)["count"]
        .sum()
        .reset_index(name="total")
    )

    persona_genre_counts = persona_genre_counts.merge(persona_totals, on="persona")
    persona_genre_counts["percentage"] = (
        persona_genre_counts["count"] / persona_genre_counts["total"]
    ) * 100

    top_genres = (
        persona_genre_counts.groupby("genre", observed=True)["count"]
        .sum()
        .nlargest(top_n)
        .index.tolist()
    )

    persona_genre_filtered = persona_genre_counts[
        persona_genre_counts["genre"].isin(top_genres)
    ].copy()

    heatmap_data = persona_genre_filtered.pivot(
        index="genre", columns="persona", values="percentage"
    ).fillna(0)

    column_order = [
        "Completionist",
        "Hardcore Fan",
        "Balanced Enthusiast",
        "Casual Viewer",
        "Selective Critic",
    ]

    existing_columns = heatmap_data.columns.intersection(column_order)
    ordered_columns = pd.Index(column_order).intersection(existing_columns)

    heatmap_data = heatmap_data[ordered_columns]
    heatmap_data["total"] = heatmap_data.sum(axis=1)
    heatmap_data = heatmap_data.sort_values("total", ascending=False).drop(
        "total", axis=1
    )

    return heatmap_data
