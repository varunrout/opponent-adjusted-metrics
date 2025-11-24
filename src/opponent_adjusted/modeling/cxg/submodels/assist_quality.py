"""Assist quality summaries by type and pass style."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from opponent_adjusted.modeling.modeling_utilities import (
    get_modeling_output_dir,
    load_cxg_modeling_dataset,
)


def _summarize(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["assist_category", "pass_style", "pressure_state"], observed=False)
        .agg(
            shots=("shot_id", "count"),
            goals=("is_goal", "sum"),
            xg_total=("statsbomb_xg", "sum"),
            avg_distance=("shot_distance", "mean"),
            avg_angle=("shot_angle", "mean"),
        )
        .reset_index()
    )
    summary = summary[summary["shots"] >= 25]
    summary["goal_rate"] = summary["goals"] / summary["shots"].clip(lower=1)
    summary["xg_per_shot"] = summary["xg_total"] / summary["shots"].clip(lower=1)
    summary["lift_vs_xg"] = summary["goal_rate"] - summary["xg_per_shot"]
    summary = summary.sort_values("lift_vs_xg", ascending=False)
    return summary


def main() -> None:
    df = load_cxg_modeling_dataset()
    df = df[df["assist_category"].notna()].copy()
    summary = _summarize(df)
    out_dir = get_modeling_output_dir("cxg", subdir="submodels")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "assist_quality_summary.csv"
    summary.to_csv(path, index=False)
    print(f"Saved {path}")


if __name__ == "__main__":  # pragma: no cover
    main()
