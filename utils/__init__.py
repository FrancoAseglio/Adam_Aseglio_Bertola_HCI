"""
Anime Analysis Utilities Package
=================================
"""

from .character_analysis import (
    character_popularity_by_role,
    character_popularity_by_genre_bars,
    voice_actor_character_popularity,
)

__all__ = [
    # Character Analysis
    "character_popularity_by_role",
    "character_popularity_by_genre_bars",
    "voice_actor_character_popularity",
]
