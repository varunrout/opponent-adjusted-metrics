"""Output helpers for reproducible analysis artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_dir(path: Path | str) -> Path:
    """Create and return a directory path."""

    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def write_csv(df: pd.DataFrame, path: Path | str) -> Path:
    """Write a DataFrame to CSV after creating the parent directory."""

    resolved = Path(path)
    ensure_dir(resolved.parent)
    df.to_csv(resolved, index=False)
    return resolved


def write_json(obj: dict[str, Any], path: Path | str) -> Path:
    """Write a JSON object after creating the parent directory."""

    resolved = Path(path)
    ensure_dir(resolved.parent)
    resolved.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    return resolved


def write_markdown(text: str, path: Path | str) -> Path:
    """Write markdown text after creating the parent directory."""

    resolved = Path(path)
    ensure_dir(resolved.parent)
    resolved.write_text(text, encoding="utf-8")
    return resolved
