import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pylab as plt

sys.path.append(str(Path.cwd().parent))

## Comment or uncomment the appropriate data path
# INPUT_DIR_PATH = "C:\\Users\\Lorenzo\\OneDrive\\Desktop\\datasets\\cleaned\\"
INPUT_DIR_PATH = "~/datasets/cleaned/"

## files to read format: adapt to pd.read_format
FILE_EXTENSION = ".csv"

# character_anime_works_df = pd.read_csv(INPUT_DIR_PATH + "character_anime_works_clean" + FILE_EXTENSION)
# character_nicknames_df = pd.read_csv(INPUT_DIR_PATH + "character_nicknames_clean" + FILE_EXTENSION)
characters_df = pd.read_csv(INPUT_DIR_PATH + "characters_clean" + FILE_EXTENSION)
# details_df = pd.read_csv(INPUT_DIR_PATH + "details_clean" + FILE_EXTENSION)
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
