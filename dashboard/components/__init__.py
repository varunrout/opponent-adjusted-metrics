"""
Components package initialization
"""

from .data_loader import (
    load_shot_data,
    load_team_aggregates,
    load_match_aggregates,
    load_model_metrics,
    load_feature_effects,
    get_available_runs,
    add_team_names,
    TEAM_NAMES
)

from .pitch_plots import (
    create_shot_map,
    create_shot_heatmap,
    create_cxg_zones_map,
    create_goal_rate_comparison,
    create_team_finishing_chart
)

from .charts import (
    create_model_comparison_chart,
    create_reliability_diagram,
    create_finishing_delta_chart,
    create_scatter_goals_vs_cxg,
    create_feature_importance_chart,
    create_game_state_heatmap
)

__all__ = [
    # Data loaders
    'load_shot_data',
    'load_team_aggregates',
    'load_match_aggregates',
    'load_model_metrics',
    'load_feature_effects',
    'get_available_runs',
    'add_team_names',
    'TEAM_NAMES',
    # Pitch plots
    'create_shot_map',
    'create_shot_heatmap',
    'create_cxg_zones_map',
    'create_goal_rate_comparison',
    'create_team_finishing_chart',
    # Charts
    'create_model_comparison_chart',
    'create_reliability_diagram',
    'create_finishing_delta_chart',
    'create_scatter_goals_vs_cxg',
    'create_feature_importance_chart',
    'create_game_state_heatmap',
]
