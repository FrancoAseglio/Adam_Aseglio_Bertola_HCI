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
from .dev_utils import add_count_on_bars


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

    # Average favorites
    role_analysis["Avg Favorites"].plot(
        kind="bar", ax=axes[0], color=["#FF6B6B", "#4ECDC4"]
    )
    axes[0].set_title(
        "Average Favorites by Character Role", fontsize=12, fontweight="bold"
    )
    axes[0].set_ylabel("Average Favorites")
    axes[0].set_xlabel("Role")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].grid(axis="y", alpha=0.5, linestyle="--")
    add_count_on_bars(axes[0])

    # Count per role
    role_analysis["Count"].plot(kind="bar", ax=axes[1], color=["#FF6B6B", "#4ECDC4"])
    axes[1].set_title("Number of Characters by Role", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Character Count")
    axes[1].set_xlabel("Role")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].grid(axis="y", alpha=0.5, linestyle="--")
    add_count_on_bars(axes[1])

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


def voice_actor_character_popularity(
    characters_df,
    favs_df,
    person_voice_works_df,
    person_details_df,
    top_n=10,
    figsize=(16, 8),
):
    """
    Analyze the correlation between voice actors and character popularity.

    Parameters
    ----------
    characters_df : pd.DataFrame
    favs_df : pd.DataFrame
    person_voice_works_df : pd.DataFrame
    person_details_df : pd.DataFrame
    top_n : int, default=10
    figsize : tuple, default=(16, 8)

    Returns
    -------
    matplotlib.figure.Figure, matplotlib.axes.Axes
    """
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
    va_stats = va_stats[va_stats["num_characters"] >= 3]

    top_by_total = va_stats.sort_values("total_character_favs", ascending=False).head(
        top_n
    )
    top_by_avg = va_stats.sort_values("avg_character_favs", ascending=False).head(top_n)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Left: Total Favorites
    ax1 = axes[0]
    plot_data_total = top_by_total.sort_values("total_character_favs", ascending=True)

    bars1 = ax1.barh(
        range(len(plot_data_total)),
        plot_data_total["total_character_favs"],
        color=cm.plasma(np.linspace(0.3, 0.9, len(plot_data_total))),
    )

    ax1.set_yticks(range(len(plot_data_total)))
    ax1.set_yticklabels(plot_data_total["name"].values)
    ax1.set_xlabel("Total Character Favorites", fontsize=12)
    ax1.set_title(
        f"Top {top_n} Most Impactful VAs\n(Total favorites across all characters)",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    ax1.grid(axis="x", alpha=0.3)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    for i, (bar, value, num_chars) in enumerate(
        zip(
            bars1,
            plot_data_total["total_character_favs"],
            plot_data_total["num_characters"],
        )
    ):
        ax1.text(
            value,
            i,
            f" {int(value):,} ({int(num_chars)} chars)",
            va="center",
            ha="left",
            fontsize=9,
            fontweight="bold",
        )

    # Right: Average Favorites
    ax2 = axes[1]
    plot_data_avg = top_by_avg.sort_values("avg_character_favs", ascending=True)

    bars2 = ax2.barh(
        range(len(plot_data_avg)),
        plot_data_avg["avg_character_favs"],
        color=cm.viridis(np.linspace(0.3, 0.9, len(plot_data_avg))),
    )

    ax2.set_yticks(range(len(plot_data_avg)))
    ax2.set_yticklabels(plot_data_avg["name"].values)
    ax2.set_xlabel("Average Favorites per Character", fontsize=12)
    ax2.set_title(
        f"Top {top_n} Most Consistent VAs\n(Highest average per character)",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    ax2.grid(axis="x", alpha=0.3)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    for i, (bar, value, num_chars) in enumerate(
        zip(bars2, plot_data_avg["avg_character_favs"], plot_data_avg["num_characters"])
    ):
        ax2.text(
            value,
            i,
            f" {int(value):,} ({int(num_chars)} chars)",
            va="center",
            ha="left",
            fontsize=9,
            fontweight="bold",
        )

    plt.tight_layout()
    return fig, axes
