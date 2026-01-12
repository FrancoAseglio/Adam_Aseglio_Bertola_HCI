"""
Engagement Analysis Functions
==============================

Completion, dropout, and user engagement analysis by various dimensions.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import (
    create_grouped_bar_chart,
    create_dual_line_distribution,
    prepare_engagement_data,
    explode_genres,
    aggregate_by_dimension,
    bin_episodes,
    prepare_completion_drop_data,
)


def completion_dropout_by_episode_count(
    details_df, stats_df, min_users=100, figsize=(1400, 700)
):
    """
    Analyze completion and dropout rates by episode count categories.

    Parameters
    ----------
    details_df : pd.DataFrame
    stats_df : pd.DataFrame
    min_users : int, default=100
    figsize : tuple, default=(1400, 700)

    Returns
    -------
    plotly.graph_objects.Figure
    """
    data = prepare_engagement_data(
        details_df, stats_df, min_users=min_users, include_episodes=True
    )
    data = bin_episodes(data)
    stats = aggregate_by_dimension(
        data,
        groupby_col="episode_category",
        agg_cols={
            "completion_rate": "mean",
            "dropout_rate": "mean",
            "mal_id": "count",
            "started_users": "sum",
        },
    )
    stats.columns = [
        "Category",
        "Avg Completion Rate",
        "Avg Dropout Rate",
        "Anime Count",
        "Total Started Users",
    ]
    data_agg = (
        data.groupby("episode_category", observed=True)
        .agg({"watching": "sum", "on_hold": "sum", "started_users": "sum"})
        .reset_index()
    )
    data_agg["Avg In-Progress Rate"] = (
        (data_agg["watching"] + data_agg["on_hold"]) / data_agg["started_users"] * 100
    )
    stats = stats.merge(
        data_agg[["episode_category", "Avg In-Progress Rate"]],
        left_on="Category",
        right_on="episode_category",
        how="left",
    )
    stats = stats.drop(columns=["episode_category"])

    fig = go.Figure()
    colors = {
        "Completion Rate": "#4ECDC4",
        "Dropout Rate": "#FF6B6B",
        "In-Progress/On-Hold": "#FFA500",
    }

    trace_configs = pd.DataFrame(
        {
            "col": ["Avg Completion Rate", "Avg Dropout Rate", "Avg In-Progress Rate"],
            "label": ["Completion Rate", "Dropout Rate", "In-Progress/On-Hold"],
        }
    )

    traces = trace_configs.apply(
        lambda row: go.Bar(
            x=stats["Category"],
            y=stats[row["col"]],
            name=row["label"],
            marker=dict(color=colors[row["label"]]),
            text=stats[row["col"]],
            texttemplate="%{text:.1f}%",
            textposition="outside",
            hovertemplate=(
                "<b>%{x}</b><br>"
                + f"{row['label']}: %{{y:.1f}}%<br>"
                + "Anime Count: %{customdata[0]:,}<br>"
                + "Total Started: %{customdata[1]:,}<br>"
                + "<extra></extra>"
            ),
            customdata=stats[["Anime Count", "Total Started Users"]].values,
        ),
        axis=1,
    )

    fig.add_traces(traces.tolist())

    max_val = (
        stats[["Avg Completion Rate", "Avg Dropout Rate", "Avg In-Progress Rate"]]
        .max()
        .max()
    )
    fig.update_layout(
        title="Completion vs Dropout vs In-Progress Rate by Episode Count<br>"
        "<sub>Does anime length affect user commitment?</sub>",
        xaxis_title="Episode Count Category",
        yaxis_title="Rate (%)",
        barmode="group",
        width=figsize[0],
        height=figsize[1],
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        font=dict(size=12),
        yaxis=dict(range=[0, max_val * 1.15]),
    )
    return fig


def completion_dropout_by_type(
    details_df, stats_df, min_users=100, figsize=(1400, 600)
):
    """
    Analyze completion and dropout rates by anime type.

    Parameters
    ----------
    details_df : pd.DataFrame
    stats_df : pd.DataFrame
    min_users : int, default=100
    figsize : tuple, default=(1400, 600)

    Returns
    -------
    plotly.graph_objects.Figure
    """
    data = prepare_engagement_data(details_df, stats_df, min_users=min_users)

    stats = aggregate_by_dimension(
        data,
        groupby_col="type",
        agg_cols={
            "completion_rate": "mean",
            "dropout_rate": "mean",
            "mal_id": "count",
            "started_users": "sum",
        },
    )

    stats.columns = [
        "Type",
        "Avg Completion Rate",
        "Avg Dropout Rate",
        "Anime Count",
        "Total Started Users",
    ]

    return create_grouped_bar_chart(
        data=stats,
        x_col="Type",
        y_cols=["Avg Completion Rate", "Avg Dropout Rate"],
        y_labels=["Completion Rate", "Dropout Rate"],
        colors=["#4ECDC4", "#FF6B6B"],
        title="Completion vs Dropout Rate by Anime Type<br><sub>How does format affect user engagement?</sub>",
        x_title="Anime Type",
        y_title="Rate (%)",
        figsize=figsize,
        sort_by="Avg Completion Rate",
        ascending=False,
        customdata_cols=["Anime Count", "Total Started Users"],
    )


def completion_dropout_by_genre(
    details_df, stats_df, top_n=10, min_users=100, figsize=(1400, 700)
):
    """
    Analyze completion and dropout rates by top anime genres.

    Parameters
    ----------
    details_df : pd.DataFrame
    stats_df : pd.DataFrame
    top_n : int, default=10
    min_users : int, default=100
    figsize : tuple, default=(1400, 700)

    Returns
    -------
    plotly.graph_objects.Figure
    """
    data = prepare_engagement_data(details_df, stats_df, min_users=min_users)
    data = explode_genres(data)

    stats = aggregate_by_dimension(
        data,
        groupby_col="genres_list",
        agg_cols={
            "completion_rate": "mean",
            "dropout_rate": "mean",
            "mal_id": "count",
            "started_users": "sum",
        },
    )

    stats.columns = [
        "Genre",
        "Avg Completion Rate",
        "Avg Dropout Rate",
        "Anime Count",
        "Total Started Users",
    ]
    stats = stats.nlargest(top_n, "Anime Count")

    return create_grouped_bar_chart(
        data=stats,
        x_col="Genre",
        y_cols=["Avg Completion Rate", "Avg Dropout Rate"],
        y_labels=["Completion Rate", "Dropout Rate"],
        colors=["#4ECDC4", "#FF6B6B"],
        title=f"Completion vs Dropout Rate by Top {top_n} Genres<br><sub>How does genre affect user engagement?</sub>",
        x_title="Genre",
        y_title="Rate (%)",
        figsize=figsize,
        sort_by="Avg Completion Rate",
        ascending=False,
        customdata_cols=["Anime Count", "Total Started Users"],
        tickangle=-45,
    )


def completion_dropout_distribution(
    stats_df, min_members=100, bin_size=5, figsize=(1400, 600)
):
    """
    Analyze and visualize distribution of completion and drop rates across anime.

    Creates an interactive line plot showing how many anime fall into each
    rate percentage bracket.

    Parameters
    ----------
    stats_df : pd. DataFrame
        Anime statistics with watching, completed, dropped, on_hold columns
    min_members : int, default=100
        Minimum total members required for inclusion
    bin_size : int, default=5
        Bin size for histogram (percentage points)
    figsize : tuple, default=(1400, 600)
        Figure size (width, height) in pixels

    Returns
    -------
    fig : plotly.graph_objects. Figure
        Interactive distribution plot
    """
    stats_filtered = prepare_completion_drop_data(stats_df, min_members=min_members)

    bins = np.arange(0, 101, bin_size)

    fig = create_dual_line_distribution(
        data1=stats_filtered["completion_rate"],
        data2=stats_filtered["drop_rate"],
        bins=bins,
        labels=["Completion Rate", "Drop Rate"],
        colors=["#2ecc71", "#e74c3c"],
        title="Distribution of Completion and Drop Rates Across Anime<br>"
        f"<sub>Anime with at least {min_members} members</sub>",
        x_title="Rate (%)",
        y_title="Number of Anime",
        figsize=figsize,
    )

    return fig
