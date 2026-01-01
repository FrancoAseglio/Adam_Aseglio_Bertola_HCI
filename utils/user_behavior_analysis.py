"""
User Behavior Analysis Functions
=================================

Analysis of user viewing patterns, completion/dropout behavior, and persona-based preferences.
"""

from .data_prep import (
    prepare_viewing_status_data,
    prepare_active_users_with_personas,
    prepare_persona_genre_data,
)

from .viz_components import (
    create_heatmap,
    create_ring_chart,
)


def viewing_status_distribution(stats_df, figsize=(20, 12)):
    """
    Visualize the overall distribution of viewing statuses using a generic Ring Chart.
    """
    # Prepare Data
    status_df = prepare_viewing_status_data(stats_df)
    
    # Clean Labels specific to this dataset
    status_df['status_label'] = status_df['status'].astype(str).str.replace('_', ' ').str.title()
    
    return create_ring_chart(
        data=status_df,
        value_col='count',
        label_col='status_label',
        title="Viewing Status Distribution",
        figsize=figsize,
        colormap="tab10",
        center_text="Total Interactions"
    )


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
