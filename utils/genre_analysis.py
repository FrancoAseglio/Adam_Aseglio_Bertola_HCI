"""
Genre Analysis Functions
========================

Most watched and most liked genres analysis.
"""

import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

from .data_prep import explode_genres, fill_missing_years
from .viz_components import create_horizontal_bar_chart


def genre_comparison_watched_vs_liked(
    details_df, favs_df, stats_df, top_n=10, figsize=(24, 10)
):
    """
    Compare most liked vs most watched genres side-by-side.

    Creates two horizontal bar charts:
    - Left: Top N most liked (favorited) genres
    - Right:  Top N most watched (engaged) genres

    Parameters
    ----------
    details_df :  pd.DataFrame
        Anime details with genres
    favs_df : pd.DataFrame
        User favorites dataset
    stats_df : pd. DataFrame
        Anime statistics with engagement data
    top_n : int, default=10
        Number of top genres to display
    figsize :  tuple, default=(24, 10)
        Figure size (width, height)

    Returns
    -------
    fig :  matplotlib. figure.Figure
    axes : np.ndarray of matplotlib.axes.Axes
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    merged_watched = details_df.merge(
        stats_df[["mal_id", "watching", "completed", "on_hold", "dropped"]],
        on="mal_id",
        how="inner",
    )

    merged_watched["started_users"] = (
        merged_watched["watching"]
        + merged_watched["completed"]
        + merged_watched["on_hold"]
        + merged_watched["dropped"]
    )

    merged_watched = merged_watched[merged_watched["genres"].notna()].copy()
    merged_watched = explode_genres(merged_watched)

    genre_stats = (
        merged_watched.groupby("genres_list")["started_users"].sum().reset_index()
    )
    genre_stats.columns = ["Genre", "Total Engaged Users"]
    top_genres_watched = genre_stats.nlargest(top_n, "Total Engaged Users")

    create_horizontal_bar_chart(
        data=top_genres_watched,
        y_col="Genre",
        x_col="Total Engaged Users",
        title=f"Top {top_n} Most Watched Genres",
        x_title="Total Engaged Users",
        y_title="",
        colormap="viridis_r",
        ax=axes[0],
    )

    anime_favs = favs_df[favs_df["fav_type"] == "anime"].copy()
    merged_liked = anime_favs.merge(
        details_df, left_on="id", right_on="mal_id", how="inner"
    )
    merged_liked = merged_liked[merged_liked["genres"].notna()].copy()
    merged_liked = explode_genres(merged_liked)

    genre_counts = merged_liked["genres_list"].value_counts().reset_index()
    genre_counts.columns = ["Genre", "Favorite Count"]
    top_genres_liked = genre_counts.head(top_n)

    create_horizontal_bar_chart(
        data=top_genres_liked,
        y_col="Genre",
        x_col="Favorite Count",
        title=f"Top {top_n} Most Liked Genres",
        x_title="Favorite Count",
        y_title="",
        colormap="plasma_r",
        ax=axes[1],
    )

    plt.tight_layout()

    return fig, axes


def genre_evolution_over_time(
    details_df,
    stats_df,
    top_n=10,
    year_range=(1965, 2025),
    figsize=(1400, 700),
    smooth=False,
):
    """
    Interactive line chart showing how genre popularity evolves over time.
    Click legend items to toggle genres on/off.

    Parameters
    ----------
    details_df : pd.DataFrame
        Anime details with genres and year/aired dates
    stats_df : pd.DataFrame
        User statistics with watching, completed, dropped, on_hold data
    top_n : int, default=10
        Number of top genres to display
    year_range : tuple, optional
        (start_year, end_year) to filter years, e.g., (2010, 2023)
    figsize : tuple, default=(1400, 700)
        Figure size in pixels (width, height)
    smooth : bool, default=False
        Apply 3-year moving average to reduce noise

    Returns
    -------
    plotly.graph_objects.Figure
    """
    data = details_df.merge(
        stats_df[["mal_id", "watching", "completed", "on_hold", "dropped"]],
        on="mal_id",
        how="inner",
    )

    data["started_users"] = (
        data["watching"] + data["completed"] + data["on_hold"] + data["dropped"]
    )

    if "year" not in data.columns:
        if "aired" not in data.columns:
            raise ValueError("Dataset must have 'year' or 'aired' column")
        data["year"] = pd.to_datetime(
            data["aired"].str.split(" to ").str[0], errors="coerce"
        ).dt.year

    data = data[data["year"].notna() & data["genres"].notna()].copy()

    if year_range:
        data = data[(data["year"] >= year_range[0]) & (data["year"] <= year_range[1])]

    data_exploded = explode_genres(data)
    genre_year_stats = (
        data_exploded.groupby(["genres_list", "year"])
        .agg({"started_users": "sum", "mal_id": "count"})
        .reset_index()
        .rename(columns={"started_users": "popularity", "mal_id": "anime_count"})
    )

    genre_year_stats = fill_missing_years(genre_year_stats)

    genre_year_stats = genre_year_stats.sort_values(["genres_list", "year"])

    if smooth:
        genre_year_stats["popularity"] = (
            genre_year_stats.groupby("genres_list")["popularity"]
            .rolling(window=3, center=True, min_periods=1)
            .mean()
            .droplevel(0)
        )

    top_genres = (
        genre_year_stats.groupby("genres_list")["popularity"]
        .sum()
        .nlargest(top_n)
        .index.tolist()
    )

    genre_year_stats = genre_year_stats[
        genre_year_stats["genres_list"].isin(top_genres)
    ]

    colors = px.colors.qualitative.Plotly + px.colors.qualitative.Set2
    top_genres_series = pd.Series(top_genres)
    color_indices = pd.RangeIndex(len(top_genres)) % len(colors)
    color_map = dict(
        zip(top_genres_series, pd.Series(colors).iloc[color_indices].values)
    )

    fig = go.Figure()

    traces = genre_year_stats.groupby("genres_list").apply(
        lambda genre_data: go.Scatter(
            x=genre_data["year"],
            y=genre_data["popularity"],
            mode="lines+markers",
            name=genre_data.name,
            line=dict(color=color_map[genre_data.name], width=2.5),
            marker=dict(size=5, line=dict(width=0.5, color="white")),
            customdata=genre_data[["anime_count"]].values,
            hovertemplate=(
                f"<b>{genre_data.name}</b><br>"
                "Year: %{x}<br>"
                "Total Engaged Users: %{y:,.0f}<br>"
                "Anime Count: %{customdata[0]:,}<br>"
                "<extra></extra>"
            ),
        )
    )

    fig.add_traces(traces.tolist())

    fig.update_layout(
        title=f"Genre Popularity Evolution Over Time<br>"
        f"<sub>Top {top_n} genres by total engaged users",
        xaxis_title="Year",
        yaxis_title="Total Engaged Users",
        width=figsize[0],
        height=figsize[1],
        hovermode="x unified",
        legend=dict(
            title="<b>Genre</b>",
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.01,
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="#CCCCCC",
            borderwidth=1,
        ),
        font=dict(size=12),
        plot_bgcolor="white",
        xaxis=dict(gridcolor="#E5E5E5", showgrid=True, dtick=5),
        yaxis=dict(gridcolor="#E5E5E5", showgrid=True),
    )
    return fig
