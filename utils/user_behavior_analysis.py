"""
User Behavior Analysis Functions
=================================

Analysis of user viewing patterns, completion/dropout behavior, and persona-based preferences.
"""


import matplotlib.pyplot as plt
import numpy as np

from .data_prep import (
    prepare_viewing_status_data,
    prepare_active_users_with_personas,
    prepare_persona_genre_data,
)

from .viz_components import (
    create_heatmap,
)


def viewing_status_distribution(stats_df, figsize=(20, 12)):
    """
    Visualize the overall distribution of viewing statuses using the helper function.
    """
    # Prepare Data using your helper function
    status_df = prepare_viewing_status_data(stats_df)
    
    # Sort by count descending to make the chart look organized (Big slices first)
    status_df = status_df.sort_values(by='count', ascending=False)
    
    counts = status_df['count']
    labels = [s.replace('_', ' ').title() for s in status_df['status']]

    # Create Visualization
    fig, ax = plt.subplots(figsize=figsize)
    
    # Use 'tab10' colormap for high contrast
    colors = plt.get_cmap('tab10').colors

    # Draw Donut Chart
    wedges, texts, autotexts = ax.pie(
        counts,
        labels=None,
        autopct='%1.1f%%',
        startangle=90,
        pctdistance=0.82,
        colors=colors,
        wedgeprops=dict(width=0.4, edgecolor='white'),
        textprops={'fontsize': 14, 'weight': 'bold', 'color': 'white'}
    )

    total = counts.sum()

    # Titles and Center Text
    ax.set_title("Viewing Status Distribution", fontsize=24, fontweight='bold')
    
    ax.text(0, 0, f"Total\n{total:,.0f}", 
            ha='center', va='center', fontsize=18, fontweight='bold', color='#333')

    # Side Legend
    ax.legend(
        wedges, labels,
        title="Status",
        title_fontsize=16,
        fontsize=14,
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1),
        frameon=False
    )

    return fig, ax


def genre_preferences_by_persona(
    profiles_df, favs_df, details_df, min_anime=10, top_n=15, figsize=(10, 8)
):
    """
    Analyze and visualize genre preferences by fan persona.

    Creates a heatmap showing which genres are preferred by different user personas
    (Completionist, Hardcore Fan, Balanced Enthusiast, Casual Viewer, Selective Critic).

    Parameters
    ----------
    profiles_df : pd. DataFrame
        User profiles with viewing statistics
    favs_df :  pd.DataFrame
        User favorites dataset
    details_df : pd. DataFrame
        Anime details with genres
    min_anime : int, default=10
        Minimum anime watched to be considered active user
    top_n : int, default=15
        Number of top genres to display
    figsize : tuple, default=(10, 8)
        Figure size (width, height)

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax :  matplotlib.axes.Axes
    """
    # Prepare active users with personas
    active_users = prepare_active_users_with_personas(profiles_df, min_anime=min_anime)

    # Prepare genre-persona data
    heatmap_data = prepare_persona_genre_data(
        active_users, favs_df, details_df, top_n=top_n
    )

    # Create visualization
    fig, ax = create_heatmap(
        data=heatmap_data,
        title=f"Genre Preferences by Fan Persona (Top {top_n} Genres)",
        x_label="Fan Persona",
        y_label="Genre",
        cmap="rocket",
        figsize=figsize,
        annot_format=".1f",
    )

    return fig, ax
