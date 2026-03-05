"""CxA data pipelines for pass-level analysis."""

from opponent_adjusted.pipelines.cxa.pass_data import build_pass_dataset
from opponent_adjusted.pipelines.cxa.shot_data import build_shot_dataset
from opponent_adjusted.pipelines.cxa.lineup_data import build_lineup_dataset
from opponent_adjusted.pipelines.cxa.pass_sequences import (
    build_pass_sequences,
    compute_sequence_xA,
    get_sequence_summary,
)
from opponent_adjusted.pipelines.cxa.possession_data import (
    build_possession_dataset,
    save_possession_dataset,
)
from opponent_adjusted.pipelines.cxa.sequence_data import (
    build_sequence_dataset,
    save_sequence_dataset,
    aggregate_player_sequences,
)

__all__ = [
    "build_pass_dataset",
    "build_shot_dataset",
    "build_lineup_dataset",
    "build_pass_sequences",
    "compute_sequence_xA",
    "get_sequence_summary",
    "build_possession_dataset",
    "save_possession_dataset",
    "build_sequence_dataset",
    "save_sequence_dataset",
    "aggregate_player_sequences",
]
