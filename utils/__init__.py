"""
Anime Analysis Utilities Package
=================================
"""

from .graphs_utils import (
    clear_cache,
)

from .data_prep import (
    prepare_engagement_data,
    prepare_character_favorites,
    explode_genres,
    bin_episodes,
    prepare_completion_drop_data,
)

from .studio_analysis import (
    studio_performance_analysis,
)

from .episode_analysis import (
    anime_episode_range_analysis,
    episode_score_correlation_analysis,
)

from .genre_analysis import (
    genre_comparison_watched_vs_liked,
    genre_evolution_over_time,
)

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

from .geo_analysis import (
    create_animated_map,
)

from .user_behavior_analysis import (
    viewing_status_distribution,
    genre_preferences_by_persona,
)

from .recommendation_analysis import (
    recommendation_genre_cooccurrence,
    recommendation_type_patterns,
)

from .viz_components import (
    create_dual_line_distribution,
)


__all__ = [
    "clear_cache",
    "studio_performance_analysis",
    "anime_episode_range_analysis",
    "episode_score_correlation_analysis",
    "genre_comparison_watched_vs_liked",
    "genre_evolution_over_time",
    "character_popularity_by_role",
    "character_popularity_by_genre_bars",
    "voice_actor_character_popularity",
    "completion_dropout_by_type",
    "completion_dropout_by_genre",
    "completion_dropout_by_episode_count",
    "completion_dropout_distribution",
    "prepare_engagement_data",
    "prepare_character_favorites",
    "explode_genres",
    "bin_episodes",
    "prepare_completion_drop_data",
    "create_dual_line_distribution",
    "create_animated_map",
    "recommendation_genre_cooccurrence",
    "recommendation_type_patterns",
    "viewing_status_distribution",
    "genre_preferences_by_persona",
]
