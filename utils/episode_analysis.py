"""
Episode Analysis Functions
==========================

Analysis of anime distribution and scores by episode count.
"""

from .data_prep import prepare_episode_score_data, aggregate_by_dimension
from .viz_components import create_dual_bar_chart, create_hexbin_violin_plot


def anime_episode_range_analysis(
    details_df, stats_df, bins=None, labels=None, figsize=(14, 5)
):
    """
    Analyze and visualize anime distribution and average scores by episode ranges.

    Creates two side-by-side bar charts:
    - Left: Number of anime in each episode range
    - Right:  Average score for each episode range

    Parameters
    ----------
    details_df : pd.DataFrame
        Anime details with 'mal_id', 'episodes', 'score' columns
    stats_df : pd.DataFrame
        Anime statistics with 'mal_id' column
    bins : list, optional
        Custom bin edges.  Default: [0, 1, 12, 24, 50, 100, 5000]
    labels : list, optional
        Custom category labels. Default: ["Single", "2-12 eps", ...]
    figsize : tuple, default=(14, 5)
        Figure size (width, height)

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : np.ndarray of matplotlib.axes. Axes
    """
    if bins is None:
        bins = [0, 1, 12, 24, 50, 100, 5000]

    if labels is None:
        labels = [
            "Single",
            "2-12 eps",
            "13-24 eps",
            "25-50 eps",
            "51-100 eps",
            "100+ eps",
        ]

    df = prepare_episode_score_data(
        details_df,
        stats_df,
        max_episodes=5000,
        bins=bins,
        labels=labels,
    )

    stats = aggregate_by_dimension(
        df,
        groupby_col="episode_category",
        agg_cols={"score": "mean", "mal_id": "count"},
    )

    stats.columns = ["episode_category", "avg_score", "anime_count"]

    fig, axes = create_dual_bar_chart(
        data=stats,
        x_col="episode_category",
        y_cols=["anime_count", "avg_score"],
        titles=[
            "Anime Count by Episode Range",
            "Average Score by Episode Range",
        ],
        y_titles=["Number of Anime", "Average Score"],
        colors=["#FF6B6B", "#4ECDC4"],
        figsize=figsize,
        bar_label_formats=["{:,.0f}", "{:.2f}"],
        y_limits=[None, (0, 10)],
    )

    return fig, axes


def episode_score_correlation_analysis(
    details_df, stats_df, max_episodes=500, bins=None, labels=None, figsize=(15, 6)
):
    """
    Analyze correlation between episode count and anime scores.

    Creates two visualizations:
    - Left: Hexbin density plot (episode count vs score)
    - Right: Violin plot (score distribution by episode category)

    Parameters
    ----------
    details_df : pd.DataFrame
        Anime details with 'mal_id', 'episodes', 'score' columns
    stats_df : pd.DataFrame
        Anime statistics with 'mal_id' column
    max_episodes : int, default=500
        Maximum episode count (filters outliers)
    bins : list, optional
        Custom bin edges. Default: [0, 1, 12, 24, 50, 100, 500]
    labels : list, optional
        Custom category labels. Default: ["Single", "2-12", ...]
    figsize : tuple, default=(15, 6)
        Figure size (width, height)

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : np. ndarray of matplotlib.axes. Axes
    correlation_stats : dict
        Contains 'pearson_r', 'interpretation', 'n_samples'
    """
    if bins is None:
        bins = [0, 1, 12, 24, 50, 100, 500]

    if labels is None:
        labels = ["Single", "2-12", "13-24", "25-50", "51-100", "100+"]

    df = prepare_episode_score_data(
        details_df, stats_df, max_episodes=max_episodes, bins=bins, labels=labels
    )

    fig, axes, correlation_stats = create_hexbin_violin_plot(
        data=df,
        x_col="episodes",
        y_col="score",
        category_col="episode_category",
        title_left="Density Hotspots",
        title_right="Score Distribution by Category",
        x_title="Number of Episodes",
        y_title="Score",
        figsize=figsize,
        gridsize=10,
        palette="viridis",
    )

    return fig, axes, correlation_stats
