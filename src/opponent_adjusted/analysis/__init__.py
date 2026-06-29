"""Analysis utilities that sit between feature engineering and modelling."""

from .cxa import (
    CxAAnalysisResult,
    build_pre_model_cxa_analysis,
    detect_cxa_target_column,
    load_action_feature_dataset,
    run_pre_model_cxa_analysis,
)
from .cxg import (
    CxGAnalysisResult,
    build_pre_model_cxg_analysis,
    load_shot_feature_dataset,
    run_pre_model_cxg_analysis,
)
from .cxt import (
    CxTAnalysisResult,
    build_pre_model_cxt_analysis,
    detect_target_proxy_column,
    load_progression_feature_dataset,
    run_pre_model_cxt_analysis,
)

__all__ = [
    "CxAAnalysisResult",
    "CxGAnalysisResult",
    "CxTAnalysisResult",
    "build_pre_model_cxa_analysis",
    "build_pre_model_cxg_analysis",
    "build_pre_model_cxt_analysis",
    "detect_cxa_target_column",
    "detect_target_proxy_column",
    "load_action_feature_dataset",
    "load_progression_feature_dataset",
    "load_shot_feature_dataset",
    "run_pre_model_cxa_analysis",
    "run_pre_model_cxg_analysis",
    "run_pre_model_cxt_analysis",
]
