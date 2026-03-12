"""CxG (Contextual Expected Goals) pipeline package."""

from .pipeline import (
    build_shots_dataset,
    add_geometric_features,
    add_context_features,
    assign_zone,
    build_opponent_profiles,
    run_pipeline,
)

__all__ = [
    "build_shots_dataset",
    "add_geometric_features",
    "add_context_features",
    "assign_zone",
    "build_opponent_profiles",
    "run_pipeline",
]
