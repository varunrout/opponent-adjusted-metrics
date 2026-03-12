"""Backward-compatible re-export — prediction modules moved to modeling.cxg.prediction."""

from opponent_adjusted.modeling.cxg.prediction.run_pipeline import (  # noqa: F401
    PredictionArtifacts,
    run_prediction_pipeline,
)
from opponent_adjusted.modeling.cxg.prediction.score_dataset import score_dataset  # noqa: F401
from opponent_adjusted.modeling.cxg.prediction.aggregate_reports import (  # noqa: F401
    aggregate_to_matches,
    aggregate_to_table,
)
