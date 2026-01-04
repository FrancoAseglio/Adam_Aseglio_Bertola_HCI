"""
Studio Analysis Functions
=========================

Studio productivity and quality analysis.
"""

from .data_prep import prepare_studio_data
from .dev_utils import add_count_on_horizontal_bars
from .viz_components import create_dual_horizontal_bar_chart


def studio_performance_analysis(
    details_df, stats_df, min_anime_threshold=0, top_n=10, figsize=(16, 8)
):
    """
    Analyze anime studios by productivity and quality.

    Shows:
    - Left: Top studios by number of anime produced (prolific)
    - Right: Top studios by average score (quality)
    """
    studio_analysis = prepare_studio_data(
        details_df, stats_df, min_anime_threshold=min_anime_threshold
    )

    top_studios_count = studio_analysis.nlargest(top_n, "Count")
    top_studios_score = studio_analysis.nlargest(top_n, "Avg Score")

    fig, axes = create_dual_horizontal_bar_chart(
        data_left=top_studios_count,
        data_right=top_studios_score,
        y_col="Studio",
        x_cols=["Count", "Avg Score"],
        titles=[
            f"Top {top_n} Most Prolific Studios",
            f"Top {top_n} Studios by Average Score",
        ],
        x_titles=["Number of Anime", "Average Score"],
        colors=["#1f77b4", "#2ca02c"],
        figsize=figsize,
        x_limits=[None, (0, 10)],
        colormap_left="magma",
        colormap_right="viridis",
    )

    add_count_on_horizontal_bars(axes[0])
    add_count_on_horizontal_bars(axes[1], fmt="{:.2f}")

    axes[1].set_xlim(0, 10.5)

    return fig, axes
