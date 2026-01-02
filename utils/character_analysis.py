"""
Character Analysis Functions
=============================

Character popularity and voice actor analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import plotly.graph_objects as go

from .data_prep import prepare_character_favorites, explode_genres
from .dev_utils import add_count_on_vertical_bars, add_count_on_horizontal_bars
from .viz_components import create_dual_horizontal_bar_chart


def voice_actor_character_popularity(
    characters_df,
    favs_df,
    person_voice_works_df,
    person_details_df,
    top_n=10,
    character_threshold=0,
    figsize=(16, 8),
):
    """
    Analyze the correlation between voice actors and character popularity using shared viz components.
    
    Delegates plotting to:
    - viz_components.create_dual_horizontal_bar_chart (layout & styling)
    - dev_utils.add_count_on_horizontal_bars (vectorized labeling)
    """
    # Data Preparation
    char_data = prepare_character_favorites(characters_df, favs_df)

    voice_char = person_voice_works_df.merge(
        char_data, on="character_mal_id", how="inner"
    )

    va_stats = (
        voice_char.groupby("person_mal_id")
        .agg({"total_favs": ["sum", "mean", "count"]})
        .reset_index()
    )
    va_stats.columns = [
        "person_mal_id",
        "total_character_favs",
        "avg_character_favs",
        "num_characters",
    ]
    va_stats = va_stats.merge(
        person_details_df[["person_mal_id", "name"]], on="person_mal_id", how="left"
    )
    
    # Filter for minimum relevance
    va_stats = va_stats[va_stats["num_characters"] >= character_threshold]

    # Create ranked subsets
    top_by_total = va_stats.nlargest(top_n, "total_character_favs")
    top_by_avg = va_stats.nlargest(top_n, "avg_character_favs")

    # Visualization
    fig, axes = create_dual_horizontal_bar_chart(
        data_left=top_by_total,
        data_right=top_by_avg,
        y_col="name",
        x_cols=["total_character_favs", "avg_character_favs"],
        titles=[
            f"Top {top_n} Most Impactful VAs\n(Total favorites)",
            f"Top {top_n} Most Consistent VAs\n(Highest average)",
        ],
        x_titles=["Total Favorites", "Average Favorites"],
        colors=None,
        figsize=figsize,
        colormap_left="plasma",
        colormap_right="viridis" 
    )

    add_count_on_horizontal_bars(axes[0], fmt='{:,.0f}')
    add_count_on_horizontal_bars(axes[1], fmt='{:.0f}') 

    return fig, axes


def character_popularity_by_role(
    characters_df, favs_df, character_anime_works_df, figsize=(14, 5)
):
    """
    Analyze and visualize character popularity based on their role in anime.

    Parameters
    ----------
    characters_df : pd.DataFrame
    favs_df : pd.DataFrame
    character_anime_works_df : pd.DataFrame
    figsize : tuple, default=(14, 5)

    Returns
    -------
    matplotlib.figure.Figure, matplotlib.axes.Axes
    """
    char_data = prepare_character_favorites(characters_df, favs_df)

    char_roles = character_anime_works_df.merge(
        char_data, on="character_mal_id", how="inner"
    )

    role_analysis = (
        char_roles.groupby("role")
        .agg({"total_favs": "mean", "character_mal_id": "count"})
        .round(2)
    )
    role_analysis.columns = ["Avg Favorites", "Count"]
    role_analysis = role_analysis.sort_values("Avg Favorites", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Count per role
    role_analysis["Count"].plot(kind="bar", ax=axes[0], color=["#FF6B6B", "#4ECDC4"])
    axes[0].set_title("Characters Count by Role", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Character Count")
    axes[0].set_xlabel("Role")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].grid(axis="y", alpha=0.5, linestyle="--")
    add_count_on_vertical_bars(axes[0])

    # Average favorites
    role_analysis["Avg Favorites"].plot(kind="bar", ax=axes[1], color=["#FF6B6B", "#4ECDC4"])
    axes[1].set_title("Average Favorites by Character Role", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Average Favorites")
    axes[1].set_xlabel("Role")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].grid(axis="y", alpha=0.5, linestyle="--")
    add_count_on_vertical_bars(axes[1])

    plt.tight_layout()
    return fig, axes


def character_popularity_by_genre_bars(
    characters_df,
    favs_df,
    character_anime_works_df,
    details_df,
    min_char_count=10,
    top_n=10,
    figsize=(1200, 700),
):
    """
    Interactive horizontal bar chart showing top genres by character popularity.

    Parameters
    ----------
    characters_df : pd.DataFrame
    favs_df : pd.DataFrame
    character_anime_works_df : pd.DataFrame
    details_df : pd.DataFrame
    min_char_count : int, default=10
    top_n : int, default=10
    figsize : tuple, default=(1200, 700)

    Returns
    -------
    plotly.graph_objects.Figure
    """
    char_data = prepare_character_favorites(characters_df, favs_df)

    genre_df = explode_genres(details_df[["mal_id", "title", "genres"]])
    
    # Ensure consistent types for merge
    character_anime_works_df = character_anime_works_df.copy()
    genre_df = genre_df.copy()
    character_anime_works_df["anime_mal_id"] = character_anime_works_df["anime_mal_id"].astype("int64")
    genre_df["mal_id"] = genre_df["mal_id"].astype("int64")

    char_genre = character_anime_works_df.merge(
        genre_df, left_on="anime_mal_id", right_on="mal_id", how="inner"
    ).merge(char_data, on="character_mal_id", how="inner")

    genre_analysis = (
        char_genre.groupby("genres_list")
        .agg({"total_favs": ["mean", "sum"], "character_mal_id": "count"})
        .round(2)
    )
    genre_analysis.columns = ["Avg Favorites", "Total Favorites", "Character Count"]
    genre_analysis = genre_analysis[genre_analysis["Character Count"] >= min_char_count]

    plot_df = genre_analysis.reset_index().rename(columns={"genres_list": "Genre"})
    plot_df = plot_df.nlargest(top_n, "Avg Favorites")

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            y=plot_df["Genre"],
            x=plot_df["Avg Favorites"],
            name="Avg Favorites",
            orientation="h",
            marker=dict(
                color=plot_df["Character Count"],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Character<br>Count"),
            ),
            text=plot_df["Avg Favorites"].apply(lambda x: f"{x:,.0f}"),
            textposition="outside",
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Avg Favorites: %{x:,.2f}<br>"
                "Character Count: %{customdata[0]:,}<br>"
                "Total Favorites: %{customdata[1]:,}<br>"
                "<extra></extra>"
            ),
            customdata=plot_df[["Character Count", "Total Favorites"]].values,
        )
    )

    fig.update_layout(
        title=f"Top {top_n} Genres by Average Character Popularity",
        xaxis_title="Average Favorites (Efficiency)",
        yaxis_title="Genre",
        yaxis={"categoryorder": "total ascending"},
        height=figsize[1],
        width=figsize[0],
        showlegend=False,
        font=dict(size=12),
        hovermode="closest",
    )

    return fig
