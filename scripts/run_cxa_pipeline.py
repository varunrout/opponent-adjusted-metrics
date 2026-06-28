#!/usr/bin/env python
"""Compatibility entrypoint for the baseline CxA action feature pipeline."""

from __future__ import annotations

from opponent_adjusted.features.cxa.action_features import (
    ACTION_FEATURES_FILENAME,
    ELIGIBLE_ACTION_TYPES,
    FEATURE_STORE_DIR,
    MAX_ACTIONS_TO_SHOT,
    MAX_SECONDS_TO_SHOT,
    SMOKE_MAX_MATCHES,
    build_action_features,
    build_action_features_from_database,
    main,
    run_pipeline,
    save_action_features,
)

__all__ = [
    "ACTION_FEATURES_FILENAME",
    "ELIGIBLE_ACTION_TYPES",
    "FEATURE_STORE_DIR",
    "MAX_ACTIONS_TO_SHOT",
    "MAX_SECONDS_TO_SHOT",
    "SMOKE_MAX_MATCHES",
    "build_action_features",
    "build_action_features_from_database",
    "run_pipeline",
    "save_action_features",
]


if __name__ == "__main__":  # pragma: no cover
    main()
