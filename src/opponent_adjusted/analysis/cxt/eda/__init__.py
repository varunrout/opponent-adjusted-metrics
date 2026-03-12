"""CxT EDA Module.

Provides comprehensive exploratory data analysis for CxT feature store.

Phases:
- phase0_data_overview: Dataset summary, null analysis, schema validation
- phase1_progressions_eda: Action type distributions, xT delta patterns
- phase2_outcomes_eda: Goal probability, turnover rates by zone
- phase3_opponent_eda: Opponent context analysis
- phase4_gamestate_eda: Game state effects on progressions
"""

from __future__ import annotations

from .run_all_eda import run_full_eda

__all__ = ["run_full_eda"]
