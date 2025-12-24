"""CxA / cxA analysis utilities.

Analysis-only exports. For pipeline builders (baselines, dataset),
use `opponent_adjusted.pipelines.cxa`.
"""

from opponent_adjusted.analysis.cxa.baseline_analysis import run_cxa_baseline_analysis
from opponent_adjusted.analysis.cxa.summarization import (
    summarize_cxa_baselines,
    summarize_cxa_baselines_from_csv,
)
from opponent_adjusted.analysis.cxa.player_analysis import (
    run_player_analysis,
    compute_xa_vs_assists,
    compute_player_profiles,
    cluster_players,
)
from opponent_adjusted.analysis.cxa.spatial_analysis import (
    run_spatial_analysis,
    compute_zone_stats,
    compute_corridor_matrix,
)
from opponent_adjusted.analysis.cxa.team_analysis import (
    run_team_analysis,
    compute_team_profiles,
    cluster_teams,
)

__all__ = [
    # Baseline analysis
    "run_cxa_baseline_analysis",
    "summarize_cxa_baselines",
    "summarize_cxa_baselines_from_csv",
    # Player analysis
    "run_player_analysis",
    "compute_xa_vs_assists",
    "compute_player_profiles",
    "cluster_players",
    # Spatial analysis
    "run_spatial_analysis",
    "compute_zone_stats",
    "compute_corridor_matrix",
    # Team analysis
    "run_team_analysis",
    "compute_team_profiles",
    "cluster_teams",
]
