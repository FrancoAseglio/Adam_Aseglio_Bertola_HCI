"""
Anime Analysis - Main API
==========================

High-level wrapper functions that load data and call analysis functions.
"""

import pandas as pd
from pathlib import Path

from .character_analysis import (
    character_popularity_by_role as _char_by_role,
    character_popularity_by_genre_bars as _char_by_genre,
    voice_actor_character_popularity as _va_popularity,
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
        INPUT_DIR_PATH = Path(
            "C:\\Users\\Lorenzo\\OneDrive\\Desktop\\datasets\\cleaned\\"
        )

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
