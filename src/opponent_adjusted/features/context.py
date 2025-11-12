"""Context feature calculations for shots."""

from typing import Optional
from opponent_adjusted.utils.time import get_minute_bucket


def calculate_game_state(score_diff: int) -> dict:
    """Calculate game state flags based on score difference.

    Args:
        score_diff: Team goals - opponent goals at time of shot

    Returns:
        Dictionary with is_leading, is_trailing, is_drawing flags
    """
    return {
        "is_leading": score_diff > 0,
        "is_trailing": score_diff < 0,
        "is_drawing": score_diff == 0,
    }


def calculate_minute_bucket_label(minute: int) -> str:
    """Get minute bucket label for a given minute.

    Args:
        minute: Match minute

    Returns:
        Bucket label (e.g., "0-15", "16-30", etc.)
    """
    return get_minute_bucket(minute)


def calculate_possession_features(
    possession_events: list,
    current_event_index: int,
    current_timestamp: float,
) -> dict:
    """Calculate possession-based features.

    Args:
        possession_events: List of events in current possession
        current_event_index: Index of current event in possession
        current_timestamp: Timestamp of current event in seconds

    Returns:
        Dictionary with possession features
    """
    if not possession_events or current_event_index < 0:
        return {
            "possession_sequence_length": 0,
            "possession_duration": 0.0,
            "previous_action_gap": 0.0,
        }

    # Sequence length is number of events before and including current
    sequence_length = current_event_index + 1

    # Duration from first event to current
    if len(possession_events) > 0 and current_event_index < len(possession_events):
        first_timestamp = possession_events[0].get("timestamp_seconds", current_timestamp)
        duration = current_timestamp - first_timestamp
    else:
        duration = 0.0

    # Gap between previous and current event
    if current_event_index > 0 and current_event_index < len(possession_events):
        prev_timestamp = possession_events[current_event_index - 1].get(
            "timestamp_seconds", current_timestamp
        )
        action_gap = current_timestamp - prev_timestamp
    else:
        action_gap = 0.0

    return {
        "possession_sequence_length": sequence_length,
        "possession_duration": max(0.0, duration),
        "previous_action_gap": max(0.0, action_gap),
    }


def calculate_pressure_features(
    possession_events: list,
    current_event_index: int,
    under_pressure: bool,
    lookback_count: int = 5,
) -> dict:
    """Calculate pressure-related features.

    Args:
        possession_events: List of events in current possession
        current_event_index: Index of current event
        under_pressure: Whether the shot is under pressure
        lookback_count: Number of recent events to check for defensive actions

    Returns:
        Dictionary with pressure features
    """
    # Count recent defensive actions by opponent
    def_actions_count = 0

    # Look at the last N events before current
    start_idx = max(0, current_event_index - lookback_count)
    for i in range(start_idx, current_event_index):
        if i < len(possession_events):
            event = possession_events[i]
            event_type = event.get("type", "")
            # Defensive actions: Pressure, Block, Interception, Clearance, etc.
            if event_type in [
                "Pressure",
                "Block",
                "Interception",
                "Clearance",
                "Duel",
                "Tackle",
            ]:
                def_actions_count += 1

    return {
        "recent_def_actions_count": def_actions_count,
        "under_pressure": under_pressure,
    }


def calculate_pressure_proxy_score(
    under_pressure: bool,
    recent_def_actions_count: int,
    possession_duration: float,
    mean_def_actions: float = 1.0,
    std_def_actions: float = 1.0,
    mean_duration: float = 5.0,
    std_duration: float = 3.0,
) -> float:
    """Calculate composite pressure proxy score.

    Args:
        under_pressure: Whether shot is under pressure
        recent_def_actions_count: Count of recent defensive actions
        possession_duration: Duration of possession in seconds
        mean_def_actions: Mean of defensive actions (for z-score)
        std_def_actions: Std dev of defensive actions
        mean_duration: Mean possession duration
        std_duration: Std dev of possession duration

    Returns:
        Composite pressure score
    """
    # Weight components
    pressure_weight = 0.5
    def_actions_weight = 0.3
    duration_weight = 0.2

    # Calculate z-scores
    def_actions_z = (
        (recent_def_actions_count - mean_def_actions) / std_def_actions
        if std_def_actions > 0
        else 0
    )

    # Inverse of duration (higher pressure = shorter possession)
    duration_inverse = 1.0 / (1.0 + possession_duration)
    duration_z = (
        (duration_inverse - (1.0 / (1.0 + mean_duration))) / (std_duration / (mean_duration**2))
        if std_duration > 0
        else 0
    )

    # Composite score
    score = (
        pressure_weight * (1.0 if under_pressure else 0.0)
        + def_actions_weight * def_actions_z
        + duration_weight * duration_z
    )

    return float(score)
