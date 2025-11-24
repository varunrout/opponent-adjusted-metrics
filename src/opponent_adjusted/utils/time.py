"""Time utilities for the opponent-adjusted metrics project."""

from datetime import datetime, timedelta
from typing import Optional


def parse_statsbomb_timestamp(timestamp: str) -> datetime:
    """Parse StatsBomb timestamp string to datetime.

    Args:
        timestamp: StatsBomb timestamp string (e.g., "00:00:15.123")

    Returns:
        datetime object
    """
    # StatsBomb timestamps are in HH:MM:SS.mmm format
    parts = timestamp.split(":")
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds_parts = parts[2].split(".")
    seconds = int(seconds_parts[0])
    milliseconds = int(seconds_parts[1]) if len(seconds_parts) > 1 else 0

    # Convert to total seconds
    total_seconds = hours * 3600 + minutes * 60 + seconds
    total_microseconds = milliseconds * 1000

    return datetime(1900, 1, 1) + timedelta(seconds=total_seconds, microseconds=total_microseconds)


def timestamp_to_seconds(timestamp: str) -> float:
    """Convert StatsBomb timestamp to seconds.

    Args:
        timestamp: StatsBomb timestamp string (e.g., "00:00:15.123")

    Returns:
        Total seconds as float
    """
    parts = timestamp.split(":")
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds_parts = parts[2].split(".")
    seconds = int(seconds_parts[0])
    milliseconds = int(seconds_parts[1]) if len(seconds_parts) > 1 else 0

    return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000


def get_minute_bucket(minute: int) -> str:
    """Get minute bucket label for a given minute.

    Args:
        minute: Match minute

    Returns:
        Bucket label (e.g., "0-15", "16-30", etc.)
    """
    if minute <= 15:
        return "0-15"
    elif minute <= 30:
        return "16-30"
    elif minute <= 45:
        return "31-45"
    elif minute <= 60:
        return "46-60"
    elif minute <= 75:
        return "61-75"
    elif minute <= 90:
        return "76-90"
    else:
        return "90+"


def calculate_possession_duration(start_time: str, end_time: str) -> float:
    """Calculate possession duration in seconds.

    Args:
        start_time: Start timestamp
        end_time: End timestamp

    Returns:
        Duration in seconds
    """
    start_seconds = timestamp_to_seconds(start_time)
    end_seconds = timestamp_to_seconds(end_time)
    return end_seconds - start_seconds
