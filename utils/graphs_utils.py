"""
Anime Analysis - Main API
==========================

High-level wrapper functions that load data and call analysis functions.
"""

import pandas as pd
from pathlib import Path

from .studio_analysis import (
    studio_performance_analysis as _studio_performance,
)

from .episode_analysis import (
    anime_distribution_by_episode_range as _episode_distribution,
    episode_score_correlation_analysis as _episode_correlation,
)

from .genre_analysis import (
    genre_comparison_watched_vs_liked as _genre_comparison,
    genre_evolution_over_time as _genre_evolution,
)

from .character_analysis import (
    character_popularity_by_role as _char_by_role,
    character_popularity_by_genre_bars as _char_by_genre,
    voice_actor_character_popularity as _va_popularity,
)

from .engagement_analysis import (
    completion_dropout_by_type as _completion_by_type,
    completion_dropout_by_genre as _completion_by_genre,
    completion_dropout_by_episode_count as _completion_by_episodes,
    completion_dropout_distribution as _completion_drop_dist,
)

from.geo_analysis import (
    create_animated_map as _create_animated_map,    
    language_vs_geographic_distribution as _language_vs_geographic_distribution,
)


# ========================================
# DATA LOADING (Module-level cache)
# ========================================

_DATA_CACHE = {}


def _load_data():
    """Load all datasets once and cache them."""
    if _DATA_CACHE:
        return _DATA_CACHE

    INPUT_DIR_PATH = Path("~/datasets/cleaned/").expanduser()
    if not INPUT_DIR_PATH.exists():
        INPUT_DIR_PATH = Path("C:\\Users\\Lorenzo\\OneDrive\\Desktop\\datasets\\cleaned\\")

    FILE_EXTENSION = ".csv"

    _DATA_CACHE["character_anime_works_df"] = pd.read_csv(INPUT_DIR_PATH / f"character_anime_works_clean{FILE_EXTENSION}")
    _DATA_CACHE["character_nicknames_df"] = pd.read_csv(INPUT_DIR_PATH / f"character_nicknames_clean{FILE_EXTENSION}")
    _DATA_CACHE["characters_df"] = pd.read_csv(INPUT_DIR_PATH / f"characters_clean{FILE_EXTENSION}")
    _DATA_CACHE["details_df"] = pd.read_csv(INPUT_DIR_PATH / f"details_clean{FILE_EXTENSION}")
    _DATA_CACHE["favs_df"] = pd.read_csv(INPUT_DIR_PATH / f"favs_clean{FILE_EXTENSION}")
    _DATA_CACHE["person_alternate_names_df"] = pd.read_csv(INPUT_DIR_PATH / f"person_alternate_names_clean{FILE_EXTENSION}")
    _DATA_CACHE["person_anime_works_df"] = pd.read_csv(INPUT_DIR_PATH / f"person_anime_works_clean{FILE_EXTENSION}")
    _DATA_CACHE["person_details_df"] = pd.read_csv(INPUT_DIR_PATH / f"person_details_clean{FILE_EXTENSION}")
    _DATA_CACHE["person_voice_works_df"] = pd.read_csv(INPUT_DIR_PATH / f"person_voice_works_clean{FILE_EXTENSION}")
    _DATA_CACHE["profiles_df"] = pd.read_csv(INPUT_DIR_PATH / f"profiles_clean{FILE_EXTENSION}")
    _DATA_CACHE["recommendations_df"] = pd.read_csv(INPUT_DIR_PATH / f"recommendations_clean{FILE_EXTENSION}")
    _DATA_CACHE["stats_df"] = pd.read_csv(INPUT_DIR_PATH / f"stats_clean{FILE_EXTENSION}")

    return _DATA_CACHE


# ========================================
# DATA CLEANING: Clear data cache
# ========================================


def clear_cache():
    """Clear the data cache to free memory or force reload."""

    global _DATA_CACHE
    _DATA_CACHE = {}
    print("✓ Data cache cleared")


# ========================================
# STUDIOS ANALYSIS FUNCTIONS
# ========================================


def studio_performance_analysis(min_anime_threshold=5, top_n=10, figsize=(16, 8)):
    """
    Analyze anime studios by productivity and quality.
    
    Shows:
    - Left: Top studios by number of anime produced (prolific)
    - Right: Top studios by average score (quality)
    
    Parameters
    ----------
    min_anime_threshold :  int, default=5
        Minimum number of anime for a studio to be included
    top_n :  int, default=10
        Number of top studios to display in each chart
    figsize : tuple, default=(16, 8)
        Figure size (width, height)
    
    Returns
    -------
    fig, axes : matplotlib figure and axes objects
    studio_analysis : DataFrame with studio metrics
    """
    data = _load_data()
    
    return _studio_performance(
        data["details_df"],
        data["stats_df"],
        min_anime_threshold=min_anime_threshold,
        top_n=top_n,
        figsize=figsize,
    )


# ========================================
# EPISODE ANALYSIS FUNCTIONS
# ========================================


def anime_distribution_by_episode_range(figsize=(14, 5)):
    """
    Analyze and visualize anime distribution and average scores by episode ranges.

    Creates two side-by-side bar charts:
    - Left: Number of anime in each episode range
    - Right: Average score for each episode range

    Parameters
    ----------
    figsize : tuple, default=(14, 5)
        Figure size (width, height)

    Returns
    -------
    fig, axes : matplotlib Figure and Axes array
    """
    data = _load_data()

    return _episode_distribution(data["details_df"], data["stats_df"], figsize=figsize)


def episode_score_correlation_analysis(max_episodes=500, figsize=(15, 6)):
    """
    Analyze correlation between episode count and anime scores.

    Creates two visualizations:
    - Left: Hexbin density plot showing episode count vs score
    - Right: Violin plot showing score distribution by episode category

    Parameters
    ----------
    max_episodes :  int, default=500
        Maximum episode count to include (filters outliers)
    figsize : tuple, default=(15, 6)
        Figure size (width, height)

    Returns
    -------
    fig, axes : matplotlib Figure and Axes array
    correlation_stats : dict
        Dictionary with 'pearson_r', 'interpretation', 'n_samples'
    """
    data = _load_data()

    return _episode_correlation(
        data["details_df"], data["stats_df"], max_episodes=max_episodes, figsize=figsize
    )


# ========================================
# GENRE ANALYSIS FUNCTIONS
# ========================================


def genre_comparison_watched_vs_liked(top_n=10, figsize=(24, 10)):
    """
    Compare most liked vs most watched genres side-by-side.

    Creates two horizontal bar charts:
    - Left: Top N most liked (favorited) genres
    - Right:  Top N most watched (engaged) genres

    Parameters
    ----------
    top_n : int, default=10
        Number of top genres to display
    figsize :  tuple, default=(24, 6)
        Figure size (width, height)

    Returns
    -------
    fig, axes : matplotlib Figure and Axes array
    """
    data = _load_data()

    return _genre_comparison(
        data["details_df"],
        data["favs_df"],
        data["stats_df"],
        top_n=top_n,
        figsize=figsize,
    )


def genre_evolution_over_time(top_n=10, year_range=None, smooth=False, figsize=(1400, 700)):
    """
    Interactive line chart showing how genre popularity evolves over time.
    Click on legend items to toggle genres on/off.

    Parameters
    ----------
    top_n : int, default=10
        Number of top genres to display
    year_range :  tuple, optional
        (start_year, end_year) to limit the time range, e.g., (2010, 2023)
    smooth : bool, default=False
        Apply 3-year moving average to reduce noise
    figsize : tuple, default=(1400, 700)
        Figure size in pixels (width, height)

    Returns
    -------
    fig : plotly.graph_objects. Figure
        Interactive line chart
    """
    data = _load_data()

    return _genre_evolution(
        data["details_df"],
        data["stats_df"],
        top_n=top_n,
        year_range=year_range,
        figsize=figsize,
        smooth=smooth,
    )


# ========================================
# CHARACTER ANALYSIS FUNCTIONS
# ========================================


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
    """
    data = _load_data()

    return _char_by_role(
        data["characters_df"],
        data["favs_df"],
        data["character_anime_works_df"],
        figsize=figsize,
    )


def character_popularity_by_genre_bars(min_char_count=10, top_n=10, figsize=(1200, 700)):
    """
    Interactive horizontal bar chart showing top genres by character popularity.

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
    """
    data = _load_data()

    return _char_by_genre(
        data["characters_df"],
        data["favs_df"],
        data["character_anime_works_df"],
        data["details_df"],
        min_char_count=min_char_count,
        top_n=top_n,
        figsize=figsize,
    )


def voice_actor_character_popularity(top_n=10, figsize=(16, 8)):
    """
    Analyze the correlation between voice actors and character popularity.

    Parameters
    ----------
    top_n : int, default=10
        Number of top voice actors to display
    figsize : tuple, default=(16, 8)
        Matplotlib figure size

    Returns
    -------
    fig, axes : matplotlib Figure and Axes
    """
    data = _load_data()

    return _va_popularity(
        data["characters_df"],
        data["favs_df"],
        data["person_voice_works_df"],
        data["person_details_df"],
        top_n=top_n,
        figsize=figsize,
    )


# ========================================
# ENGAGEMENT ANALYSIS FUNCTIONS
# ========================================


def completion_dropout_by_type(figsize=(1400, 600)):
    """
    Analyze completion and dropout rates by anime type.

    Parameters
    ----------
    figsize : tuple, default=(1400, 600)
        Figure size in pixels (width, height).

    Returns
    -------
    fig : plotly.graph_objects.Figure
    """
    data = _load_data()

    return _completion_by_type(data["details_df"], data["stats_df"], figsize=figsize)


def completion_dropout_by_genre(top_n_genres=10, figsize=(1400, 700)):
    """
    Analyze completion and dropout rates by top anime genres.

    Parameters
    ----------
    top_n_genres : int, default=10
        Number of top genres to display.
    figsize : tuple, default=(1400, 700)
        Figure size in pixels (width, height).

    Returns
    -------
    fig : plotly.graph_objects.Figure
    """
    data = _load_data()

    return _completion_by_genre(data["details_df"], data["stats_df"], top_n=top_n_genres, figsize=figsize)


def completion_dropout_by_episode_count(figsize=(1400, 700)):
    """
    Analyze completion and dropout rates by anime episode count.

    Parameters
    ----------
    figsize : tuple, default=(1400, 700)
        Figure size in pixels (width, height).

    Returns
    -------
    fig : plotly.graph_objects.Figure
    """
    data = _load_data()

    return _completion_by_episodes(data["details_df"], data["stats_df"], figsize=figsize)


def completion_dropout_distribution(min_members=100, figsize=(1400, 600)):
    """
    Analyze and visualize distribution of completion and drop rates across anime.

    Creates an interactive line plot showing how many anime fall into each rate percentage bracket.

    Parameters
    ----------
    min_members : int, default=100
        Minimum total members required for inclusion
    figsize : tuple, default=(1400, 600)
        Figure size (width, height) in pixels

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Interactive distribution plot
    """
    data = _load_data()

    return _completion_drop_dist(data["stats_df"], min_members=min_members, figsize=figsize)

# ========================================
# GEOGRAPHIC ANALYSIS FUNCTIONS
# ========================================

def create_animated_map(profiles_df):
    """
    Create animated choropleth map showing anime user growth over time.

    Parameters
    ----------
    profiles_df : pd.DataFrame
        User profiles with location and joined date

    Returns
    -------
    plotly.graph_objects.Figure
    """

    data = _load_data()
    return _create_animated_map(data["profiles_df"])


def language_vs_geographic_distribution(profiles_df, figsize=(10, 6)):
    """
    Create a bar chart showing user distribution by primary language.

    Parameters
    ----------
    profiles_df : pd.DataFrame
        User profiles with location and primary language
    figsize : tuple, default=(10, 6)
        Figure size (width, height)

    Returns
    -------
    fig : matplotlib Figure
    ax : matplotlib Axes
    """

    data = _load_data()
    return _language_vs_geographic_distribution(data["profiles_df"], figsize=figsize)