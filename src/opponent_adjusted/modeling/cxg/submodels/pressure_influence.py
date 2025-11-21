"""Pressure influence analysis controlling for geometry buckets."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from opponent_adjusted.modeling.modeling_utilities import (
    get_modeling_output_dir,
    load_cxg_modeling_dataset,
)

DISTANCE_BINS = [0, 12, 18, 24, 40]
ANGLE_BINS = [0, 20, 40, 60, 90]


def _assign_bins(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["distance_bin"] = pd.cut(
        df["shot_distance"], DISTANCE_BINS, labels=["0-12", "12-18", "18-24", "24+"], right=False
    )
    df["angle_bin"] = pd.cut(
        df["shot_angle"], ANGLE_BINS, labels=["0-20", "20-40", "40-60", "60+"], right=False
    )
    return df


def _summarize(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["pressure_state", "distance_bin", "angle_bin"], observed=False)
        .agg(
            shots=("shot_id", "count"),
            goals=("is_goal", "sum"),
            xg_total=("statsbomb_xg", "sum"),
        )
        .reset_index()
    )
    summary = summary[summary["shots"] >= 30]
    summary["goal_rate"] = summary["goals"] / summary["shots"].clip(lower=1)
    summary["xg_per_shot"] = summary["xg_total"] / summary["shots"].clip(lower=1)
    summary["lift_vs_xg"] = summary["goal_rate"] - summary["xg_per_shot"]
    summary = summary.sort_values(["distance_bin", "angle_bin", "pressure_state"])
    return summary


def main() -> None:
    df = load_cxg_modeling_dataset()
    df = _assign_bins(df)
    summary = _summarize(df)
    out_dir = get_modeling_output_dir("cxg", subdir="submodels")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "pressure_influence_summary.csv"
    summary.to_csv(path, index=False)
    print(f"Saved {path}")


if __name__ == "__main__":  # pragma: no cover
    main()
