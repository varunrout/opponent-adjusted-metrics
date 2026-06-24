"""CxT Modeling Package"""

from opponent_adjusted.modeling.cxt.contextual_model import (
    train_cxt_model,
    evaluate_cxt_model,
    CxTModel,
)
from opponent_adjusted.modeling.cxt.cxt_api import (
    CxTPredictor,
    CxTResult,
    PlayerCxTSummary,
    TeamCxTSummary,
    get_cxt_predictor,
    predict_cxt,
)

__all__ = [
    # Core model
    "train_cxt_model",
    "evaluate_cxt_model",
    "CxTModel",
    # API
    "CxTPredictor",
    "CxTResult",
    "PlayerCxTSummary",
    "TeamCxTSummary",
    "get_cxt_predictor",
    "predict_cxt",
]
