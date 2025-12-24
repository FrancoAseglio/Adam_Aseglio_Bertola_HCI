"""
Anime Analysis Utilities Package
=================================
"""

from .character_analysis import (
    character_popularity_by_role,
    character_popularity_by_genre_bars,
    voice_actor_character_popularity,
)

from .engagement_analysis import (
    completion_dropout_by_type,
    completion_dropout_by_genre,
    completion_dropout_by_episode_count,
    completion_dropout_distribution,
)

from .data_prep import (
    prepare_engagement_data,
    prepare_character_favorites,
    explode_genres,
    bin_episodes,
    prepare_completion_drop_data,
)

from .viz_components import (
    create_dual_line_distribution,
)

__all__ = [
    # Character Analysis
    "character_popularity_by_role",
    "character_popularity_by_genre_bars",
    "voice_actor_character_popularity",

    # Engagement Analysis
    "completion_dropout_by_type",
    "completion_dropout_by_genre",
    "completion_dropout_by_episode_count",
    "completion_dropout_distribution",

    # Data Preparation
    "prepare_engagement_data",
    "prepare_character_favorites",
    "explode_genres",
    "bin_episodes",
    "prepare_completion_drop_data",

    # Visualization Components
    "create_dual_line_distribution",
]
