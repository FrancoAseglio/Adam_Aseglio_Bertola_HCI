"""
Studio Analysis Functions
=========================

Studio productivity and quality analysis. 
"""

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from . data_prep import prepare_studio_data
from .viz_components import create_dual_horizontal_bar_chart


def studio_performance_analysis(
    details_df, 
    stats_df, 
    min_anime_threshold=5, 
    top_n=10, 
    figsize=(16, 8)
):
    """
    Analyze anime studios by productivity and quality.
    
    Shows:
    - Left: Top studios by number of anime produced (prolific)
    - Right: Top studios by average score (quality)
    
    Parameters
    ----------
    details_df : pd.DataFrame
        Anime details with studios column
    stats_df : pd. DataFrame
        Anime statistics dataset
    min_anime_threshold :  int, default=5
        Minimum number of anime for a studio to be included
    top_n :  int, default=10
        Number of top studios to display in each chart
    figsize : tuple, default=(16, 8)
        Figure size (width, height)
    
    Returns
    -------
    fig :  matplotlib. figure.Figure
    axes : np.ndarray of matplotlib.axes.Axes
    studio_analysis : pd.DataFrame
        DataFrame with studio metrics (Avg Score, Count)
    """
    # Prepare data
    studio_analysis = prepare_studio_data(
        details_df, 
        stats_df, 
        min_anime_threshold=min_anime_threshold
    )
    
    # Get top studios for each metric
    top_studios_count = studio_analysis.nlargest(top_n, "Count")
    top_studios_score = studio_analysis.nlargest(top_n, "Avg Score")
    
    # Create visualization
    fig, axes = create_dual_horizontal_bar_chart(
        data_left=top_studios_count,
        data_right=top_studios_score,
        y_col="Studio",
        x_cols=["Count", "Avg Score"],
        titles=[
            f"Top {top_n} Most Prolific Studios",
            f"Top {top_n} Studios by Average Score"
        ],
        x_titles=["Number of Anime", "Average Score"],
        colors=["#1f77b4", "#2ca02c"],
        figsize=figsize,
        x_limits=[None, (0, 10)],
        colormap_left="magma",
        colormap_right="viridis"
    )
    
    return fig, axes
