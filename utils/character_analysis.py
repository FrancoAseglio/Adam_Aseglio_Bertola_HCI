"""
Character Analysis Functions
=============================

Character popularity and voice actor analysis.
"""

import matplotlib.pyplot as plt
import plotly.graph_objects as go

from .data_prep import prepare_character_favorites, explode_genres
from .dev_utils import add_count_on_vertical_bars, add_count_on_horizontal_bars
from .viz_components import create_dual_horizontal_bar_chart


def expand_axis_limit(ax):
        left, right = ax.get_xlim()
        ax.set_xlim(left, right * 1.30)


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
    Analyze and plot the correlation between voice actors and character popularity.

    Generates a dual horizontal bar chart showing the top VAs by total favorites
    and by average favorites per character.

    Parameters
    ----------
    characters_df : pd.DataFrame
        DataFrame containing character metadata (must include 'character_mal_id').
    favs_df : pd.DataFrame
        DataFrame containing favorites/popularity counts for characters.
    person_voice_works_df : pd.DataFrame
        Mapping of voice actors to the character IDs they have voiced.
    person_details_df : pd.DataFrame
        Metadata for the voice actors, including 'name' and 'person_mal_id'.
    top_n : int, default=10
        The number of top performers to display in each subplot.
    character_threshold : int, default=50
        Minimum number of characters a VA must have voiced to be included in the stats.
    figsize : tuple, default=(16, 8)
        The dimensions of the resulting matplotlib figure.

    Returns
    -------
    matplotlib.figure.Figure, np.ndarray
        The figure object and an array containing the two subplots (axes).
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

    va_stats = va_stats[va_stats["num_characters"] >= character_threshold]

    top_by_total = va_stats.nlargest(top_n, "total_character_favs")
    top_by_avg = va_stats.nlargest(top_n, "avg_character_favs")

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
        colormap_right="viridis",
    )

    labels_left = (
        top_by_total["total_character_favs"].map('{:,.0f}'.format)
        + " (" 
        + top_by_total["num_characters"].astype(str) 
        + " roles)"
    ).values

    labels_right = (
        top_by_avg["avg_character_favs"].map('{:,.0f}'.format)
        + " (" 
        + top_by_avg["num_characters"].astype(str) 
        + " roles)"
    ).values

    add_count_on_horizontal_bars(axes[0], custom_labels=labels_left)
    add_count_on_horizontal_bars(axes[1], custom_labels=labels_right)

    list(map(expand_axis_limit, axes))

    fig.subplots_adjust(wspace=0.3)

    return fig, axes


## What makes a character resonate the most with audience?

We can simply begin by taking a look at the influence of their role. If any...



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
            text = plot_df["Avg Favorites"],
            texttemplate = '%{text:,.0f}',
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
