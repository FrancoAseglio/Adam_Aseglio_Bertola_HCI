"""
Recommendation Analysis Functions
==================================

Analyze recommendation patterns by genre and type.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go

from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

from .data_prep import explode_genres


def recommendation_genre_cooccurrence(
    details_df,
    recommendations_df,
    min_recommendations=250,
    top_n=10,
    figsize=(1000, 800),
):
    """
    Interactive heatmap showing which anime genres get recommended together.

    Parameters
    ----------
    details_df : pd.DataFrame
        Anime details with genres information
    recommendations_df : pd.DataFrame
        Recommendation pairs between anime
    min_recommendations : int, default=250
        Minimum number of recommendations to include a genre pair
    top_n : int, default=10
        Number of top genres to display
    figsize : tuple, default=(1000, 800)
        Figure size in pixels (width, height)

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive heatmap showing genre co-occurrence patterns
    """
    anime_genres = explode_genres(details_df[["mal_id", "title", "genres"]])

    rec_with_genres = recommendations_df.merge(
        anime_genres, left_on="mal_id", right_on="mal_id", how="inner"
    ).merge(
        anime_genres,
        left_on="recommendation_mal_id",
        right_on="mal_id",
        how="inner",
        suffixes=("_source", "_target"),
    )

    rec_exploded = (
        rec_with_genres.explode("genres_list_source")
        .explode("genres_list_target")
        .rename(
            columns={
                "genres_list_source": "source_genre",
                "genres_list_target": "target_genre",
            }
        )
    )

    rec_exploded = rec_exploded[
        (rec_exploded["source_genre"] != "") & (rec_exploded["target_genre"] != "")
    ]

    genre_pairs = (
        rec_exploded.groupby(["source_genre", "target_genre"])
        .size()
        .reset_index(name="recommendation_count")
    )

    top_genres = (
        pd.concat(
            [
                genre_pairs.groupby("source_genre")["recommendation_count"].sum(),
                genre_pairs.groupby("target_genre")["recommendation_count"].sum(),
            ]
        )
        .groupby(level=0)
        .sum()
        .nlargest(top_n)
        .index.tolist()
    )

    matrix_data = genre_pairs[
        genre_pairs["source_genre"].isin(top_genres)
        & genre_pairs["target_genre"].isin(top_genres)
        & (genre_pairs["recommendation_count"] >= min_recommendations)
    ]

    heatmap_matrix = matrix_data.pivot_table(
        index="source_genre",
        columns="target_genre",
        values="recommendation_count",
        fill_value=0,
    )

    if len(heatmap_matrix) > 1:
        dist_matrix = 1 / (heatmap_matrix + 1)
        dist_matrix = (dist_matrix + dist_matrix.T) / 2
        np.fill_diagonal(dist_matrix.values, 0)

        linkage_matrix = linkage(squareform(dist_matrix), method="average")
        dendro = dendrogram(linkage_matrix, no_plot=True)
        order = dendro["leaves"]

        ordered_genres = heatmap_matrix.index[order].tolist()
        heatmap_matrix = heatmap_matrix.loc[ordered_genres, ordered_genres]

    fig = go.Figure(
        data=go.Heatmap(
            z=heatmap_matrix.values,
            x=heatmap_matrix.columns,
            y=heatmap_matrix.index,
            colorscale="YlOrRd",
            text=heatmap_matrix.values,
            texttemplate="%{text:,.0f}",
            textfont={"size": 10},
            hovertemplate=(
                "<b>%{y}</b> → <b>%{x}</b><br>"
                "Recommendations: %{z:,.0f}<br>"
                "<extra></extra>"
            ),
            colorbar=dict(title="Recommendation<br>Count"),
        )
    )

    fig.update_layout(
        title=f"Genre Co-occurrence in Recommendations (Top {top_n} Genres)",
        xaxis_title="Recommended Genre",
        yaxis_title="Source Genre",
        width=figsize[0],
        height=figsize[1],
        font=dict(size=11),
        xaxis=dict(tickangle=-45),
    )

    return fig


def recommendation_type_patterns(
    details_df, recommendations_df, min_count=200, figsize=(1200, 600)
):
    """
    Interactive bar chart showing what anime types get recommended together.

    Parameters
    ----------
    details_df : pd.DataFrame
    recommendations_df : pd.DataFrame
    min_count : int, default=200
    figsize : tuple, default=(1200, 600)

    Returns
    -------
    plotly.graph_objects.Figure
    """
    rec_types = recommendations_df.merge(
        details_df[["mal_id", "type", "title"]],
        left_on="mal_id",
        right_on="mal_id",
        how="inner",
    ).merge(
        details_df[["mal_id", "type", "title"]],
        left_on="recommendation_mal_id",
        right_on="mal_id",
        how="inner",
        suffixes=("_source", "_target"),
    )

    type_pairs = (
        rec_types.groupby(["type_source", "type_target"])
        .size()
        .reset_index(name="count")
    )

    type_pairs = type_pairs[type_pairs["count"] >= min_count]
    type_pairs = type_pairs.sort_values("count", ascending=False)

    fig = go.Figure()

    source_types = type_pairs["type_source"].unique()

    for source_type in source_types:
        data = type_pairs[type_pairs["type_source"] == source_type]

        fig.add_trace(
            go.Bar(
                name=f"From {source_type}",
                x=data["type_target"],
                y=data["count"],
                texttemplate="%{y}",
                textposition="auto",
                textfont=dict(color="white"),
                hovertemplate=(
                    f"<b>{source_type} → %{{x}}</b><br>"
                    "Recommendations: %{y:,}<br>"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title="Anime Type Recommendation Patterns",
        xaxis_title="Recommended Type",
        yaxis_title="Number of Recommendations (log scale)",
        barmode="group",
        width=figsize[0],
        height=figsize[1],
        legend_title="Source Type",
        hovermode="x unified",
    )

    fig.update_yaxes(type="log")

    return fig
