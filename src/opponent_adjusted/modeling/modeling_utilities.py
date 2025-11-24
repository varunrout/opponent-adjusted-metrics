"""Shared utilities for modeling workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from opponent_adjusted.config import settings


def get_modeling_output_dir(model_name: str, subdir: str | None = None) -> Path:
    """Return (and create) the standard output directory for modeling artifacts."""

    root = settings.data_root.parent / "outputs" / "modeling" / model_name
    if subdir:
        root = root / subdir
    root.mkdir(parents=True, exist_ok=True)
    return root


def load_dataset_from_versions(
    model_name: str,
    filenames: Iterable[str],
) -> pd.DataFrame:
    base_dir = get_modeling_output_dir(model_name)
    for filename in filenames:
        path = base_dir / filename
        if not path.exists():
            continue
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        if path.suffix == ".csv":
            return pd.read_csv(path)
    raise FileNotFoundError(
        f"None of the dataset files {list(filenames)} were found under {base_dir}."
    )


def load_cxg_modeling_dataset(prefer_filtered: bool = True) -> pd.DataFrame:
    """Load the CxG dataset, preferring the filtered version when available."""

    filenames = []
    if prefer_filtered:
        filenames.extend(["cxg_dataset_filtered.parquet", "cxg_dataset_filtered.csv"])
    filenames.extend(["cxg_dataset.parquet", "cxg_dataset.csv"])
    return load_dataset_from_versions("cxg", filenames)
