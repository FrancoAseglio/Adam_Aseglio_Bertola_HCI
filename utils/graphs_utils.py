import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pylab as plt
import plotly.graph_objects as go

from utils.dev_utils import add_count_on_bars, parse_list

sys.path.append(str(Path.cwd().parent))

## Comment or uncomment the appropriate data path
# INPUT_DIR_PATH = "C:\\Users\\Lorenzo\\OneDrive\\Desktop\\datasets\\cleaned\\"
INPUT_DIR_PATH = "~/datasets/cleaned/"

## files to read format: adapt to pd.read_format
FILE_EXTENSION = ".csv"

character_anime_works_df = pd.read_csv(INPUT_DIR_PATH + "character_anime_works_clean" + FILE_EXTENSION)
# character_nicknames_df = pd.read_csv(INPUT_DIR_PATH + "character_nicknames_clean" + FILE_EXTENSION)
characters_df = pd.read_csv(INPUT_DIR_PATH + "characters_clean" + FILE_EXTENSION)
details_df = pd.read_csv(INPUT_DIR_PATH + "details_clean" + FILE_EXTENSION)
favs_df = pd.read_csv(INPUT_DIR_PATH + "favs_clean" + FILE_EXTENSION)
# person_alternate_names_df = pd.read_csv(INPUT_DIR_PATH + "person_alternate_names_clean" + FILE_EXTENSION)
# person_anime_works_df = pd.read_csv(INPUT_DIR_PATH + "person_anime_works_clean" + FILE_EXTENSION)
# person_details_df = pd.read_csv(INPUT_DIR_PATH + "person_details_clean" + FILE_EXTENSION)
# person_voice_works_df = pd.read_csv(INPUT_DIR_PATH + "person_voice_works_clean" + FILE_EXTENSION)
# profiles_df = pd.read_csv(INPUT_DIR_PATH + "profiles_clean" + FILE_EXTENSION)
# ratings_df = pd.read_csv(INPUT_DIR_PATH + "ratings_clean" + FILE_EXTENSION)
# recommendations_df = pd.read_csv(INPUT_DIR_PATH + "recommendations_clean" + FILE_EXTENSION)
# stats_df = pd.read_csv(INPUT_DIR_PATH + "stats_clean" + FILE_EXTENSION)


def top_n_most_favorited_characters(top_n=10, figsize=(12, 8)):
    """
    Plot a bar chart of the most favorited anime characters.

    Parameters
    ----------
    top_n : int, default=10
        Number of characters to display.
    figsize : tuple, default=(12, 8)
        Matplotlib figure size.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
        Visualization of the top-N most favorited characters.
    """

    # Compute user favorites
    char_favs = favs_df[favs_df["fav_type"] == "character"].copy()
    char_fav_counts = char_favs.groupby("id").size().reset_index(name="user_favs")

    # Merge with character dataset
    char_popularity = characters_df.merge(char_fav_counts, left_on="character_mal_id", right_on="id", how="left")
    char_popularity["user_favs"] = char_popularity["user_favs"].fillna(0)

    # Total popularity score
    char_popularity["total_favs"] = (char_popularity["favorites"].fillna(0) + char_popularity["user_favs"])

    # Extract top entries
    top_n_chars = char_popularity.nlargest(top_n, "total_favs").sort_values("total_favs", ascending=False)

    # --- Visualization ---
    fig, ax = plt.subplots(figsize=figsize)
    plot_data = top_n_chars.sort_values("total_favs", ascending=True)

    bars = ax.barh(
        range(len(plot_data)),
        plot_data["total_favs"],
        color=cm.plasma(np.linspace(0.3, 0.9, len(plot_data))),
    )

    ax.set_yticks(range(len(plot_data)))
    ax.set_yticklabels(plot_data["name"].values)
    ax.set_xlabel("Total Favorites", fontsize=12)
    ax.set_title(f"Top {top_n} Most Favorited Characters", fontsize=14, fontweight="bold", pad=20)

    # Value labels
    for i, (bar, value) in enumerate(zip(bars, plot_data["total_favs"])):
        ax.text(
            value,
            i,
            f" {int(value):,}",
            va="center",
            ha="left",
            fontsize=10,
            fontweight="bold",
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.6, linestyle="--")

    plt.tight_layout()
    return fig, ax


def character_popularity_by_role(figsize=(14, 5)):
    """
    Analyze and visualize character popularity based on their role in anime.

    Parameters
    ----------
    figsize : tuple, default=(14, 5)
        Matplotlib figure size for the 1x2 layout.

    Returns
    -------
    fig, axes : matplotlib Figure and Axes array
        Side-by-side bar charts of average favorites and character counts by role.
    """

    # Compute user favorites
    char_favs = favs_df[favs_df["fav_type"] == "character"].copy()
    char_fav_counts = char_favs.groupby("id").size().reset_index(name="user_favs")

    # Merge with character details
    char_popularity = characters_df.merge(char_fav_counts, left_on="character_mal_id", right_on="id", how="left")
    char_popularity["user_favs"] = char_popularity["user_favs"].fillna(0)
    char_popularity["total_favs"] = (char_popularity["favorites"].fillna(0) + char_popularity["user_favs"])

    # Merge with role dataset
    char_roles = character_anime_works_df.merge(
        char_popularity[["character_mal_id", "name", "total_favs"]],
        on="character_mal_id",
        how="inner",
    )

    # Aggregate by role
    role_analysis = (
        char_roles.groupby("role")
        .agg({"total_favs": "mean", "character_mal_id": "count"})
        .round(2)
    )
    role_analysis.columns = ["Avg Favorites", "Count"]
    role_analysis = role_analysis.sort_values("Avg Favorites", ascending=False)

    # --- Visualization ---
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Average favorites
    role_analysis["Avg Favorites"].plot(kind="bar", ax=axes[0], color=["#FF6B6B", "#4ECDC4"])
    axes[0].set_title(
        "Average Favorites by Character Role",
        fontsize=12,
        fontweight="bold",
    )
    axes[0].set_ylabel("Average Favorites")
    axes[0].set_xlabel("Role")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].grid(axis="y", alpha=0.5, linestyle="--")
    add_count_on_bars(axes[0])

    # Count per role
    role_analysis["Count"].plot(kind="bar", ax=axes[1], color=["#FF6B6B", "#4ECDC4"])
    axes[1].set_title(
        "Number of Characters by Role",
        fontsize=12,
        fontweight="bold",
    )
    axes[1].set_ylabel("Character Count")
    axes[1].set_xlabel("Role")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].grid(axis="y", alpha=0.5, linestyle="--")
    add_count_on_bars(axes[1])

    plt.tight_layout()
    return fig, axes


def character_popularity_by_genre_bars(min_char_count=10, top_n=10, figsize=(1200, 700)):
    """
    Interactive horizontal bar chart showing top genres.
    Bar length shows avg favorites, color shows character count (volume).

    Parameters
    ----------
    min_char_count : int, default=10
        Minimum number of characters required for a genre to be included.
    top_n : int, default=10
        Number of top genres to display.
    figsize : tuple, default=(1200, 700)
        Figure size in pixels (width, height).

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Interactive bar chart.
    """

    # Parse and explode genres
    details_df["genres_list"] = details_df["genres"].astype(str).apply(parse_list)
    genre_df = (
        details_df[["mal_id", "title", "genres_list"]]
        .explode("genres_list")
        .query("genres_list != ''")
    )

    # Compute user favorites
    char_favs = favs_df[favs_df["fav_type"] == "character"].copy()
    char_fav_counts = char_favs.groupby("id").size().reset_index(name="user_favs")

    # Merge with character details
    char_popularity = characters_df.merge(char_fav_counts, left_on="character_mal_id", right_on="id", how="left")
    char_popularity["user_favs"] = char_popularity["user_favs"].fillna(0)
    char_popularity["total_favs"] = (char_popularity["favorites"].fillna(0) + char_popularity["user_favs"])

    # Merge character-anime works with genres
    char_genre = character_anime_works_df.merge(
        genre_df, left_on="anime_mal_id", right_on="mal_id", how="inner"
    ).merge(
        char_popularity[["character_mal_id", "name", "total_favs"]],
        on="character_mal_id",
        how="inner",
    )

    # Aggregate by genre
    genre_analysis = (
        char_genre.groupby("genres_list")
        .agg({"total_favs": ["mean", "sum"], "character_mal_id": "count"})
        .round(2)
    )
    genre_analysis.columns = ["Avg Favorites", "Total Favorites", "Character Count"]
    genre_analysis = genre_analysis[genre_analysis["Character Count"] >= min_char_count]

    # Get top N genres
    plot_df = genre_analysis.reset_index().rename(columns={"genres_list": "Genre"})
    plot_df = plot_df.nlargest(top_n, "Avg Favorites")

    # --- Visualization ---
    fig = go.Figure()

    # Add horizontal bar chart colored by character count
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
            hovertemplate="<b>%{y}</b><br>"
            + "Avg Favorites: %{x:,.2f}<br>"
            + "Character Count: %{customdata[0]:,}<br>"
            + "Total Favorites: %{customdata[1]:,}<br>"
            + "<extra></extra>",
            customdata=plot_df[["Character Count", "Total Favorites"]].values,
        )
    )

    # Update layout
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


