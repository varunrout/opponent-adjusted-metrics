"""Defensive trigger uplift summaries."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from opponent_adjusted.modeling.modeling_utilities import (
    get_modeling_output_dir,
    load_cxg_modeling_dataset,
)

GAP_BINS = [0, 2, 5, 10, 30]
GAP_LABELS = ["0-2s", "2-5s", "5-10s", "10s+"]


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["time_gap_bucket"] = pd.cut(df["time_gap_seconds"], GAP_BINS, labels=GAP_LABELS, right=False)
    df.loc[df["time_gap_bucket"].isna(), "time_gap_bucket"] = "10s+"
    return df


def _summarize(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["def_label", "time_gap_bucket", "possession_match"], observed=False)
        .agg(
            shots=("shot_id", "count"),
            goals=("is_goal", "sum"),
            xg_total=("statsbomb_xg", "sum"),
        )
        .reset_index()
    )
    summary = summary[summary["shots"] >= 25]
    summary["goal_rate"] = summary["goals"] / summary["shots"].clip(lower=1)
    summary["xg_per_shot"] = summary["xg_total"] / summary["shots"].clip(lower=1)
    summary["lift_vs_xg"] = summary["goal_rate"] - summary["xg_per_shot"]
    summary = summary.sort_values(["lift_vs_xg"], ascending=False)
    return summary


def main() -> None:
    df = load_cxg_modeling_dataset()
    df = _prepare(df)
    summary = _summarize(df)
    out_dir = get_modeling_output_dir("cxg", subdir="submodels")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "defensive_trigger_uplift.csv"
    summary.to_csv(path, index=False)
    print(f"Saved {path}")


if __name__ == "__main__":  # pragma: no cover
    main()
