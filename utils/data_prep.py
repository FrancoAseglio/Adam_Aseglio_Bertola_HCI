"""
Data Preparation Utilities for Anime Analysis
==============================================

Reusable functions for preparing and processing anime-related datasets.
"""

import pandas as pd
import numpy as np

from .dev_utils import parse_list


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

    char_popularity = characters_df.merge(char_fav_counts, left_on="character_mal_id", right_on="id", how="left")
    char_popularity["user_favs"] = char_popularity["user_favs"].fillna(0)
    char_popularity["total_favs"] = (char_popularity["favorites"].fillna(0) + char_popularity["user_favs"])

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
