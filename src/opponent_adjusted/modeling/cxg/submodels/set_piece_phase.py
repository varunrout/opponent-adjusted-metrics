"""Set-piece phase uplift summaries."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from opponent_adjusted.modeling.modeling_utilities import (
    get_modeling_output_dir,
    load_cxg_modeling_dataset,
)


COLUMNS = [
    "set_piece_category",
    "set_piece_phase",
    "score_state",
    "minute_bucket_label",
]


def _summarize(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(COLUMNS, observed=False)
        .agg(
            shots=("shot_id", "count"),
            goals=("is_goal", "sum"),
            xg_total=("statsbomb_xg", "sum"),
        )
        .reset_index()
    )
    summary = summary[summary["shots"] >= 10]
    summary["goal_rate"] = summary["goals"] / summary["shots"].clip(lower=1)
    summary["xg_per_shot"] = summary["xg_total"] / summary["shots"].clip(lower=1)
    summary["lift_vs_xg"] = summary["goal_rate"] - summary["xg_per_shot"]
    summary = summary.sort_values(["set_piece_category", "set_piece_phase", "lift_vs_xg"], ascending=[True, True, False])
    return summary


def main() -> None:
    df = load_cxg_modeling_dataset()
    set_piece_df = df[df["set_piece_category"] != "Open Play"].copy()
    summary = _summarize(set_piece_df)
    out_dir = get_modeling_output_dir("cxg", subdir="submodels")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "set_piece_phase_uplift.csv"
    summary.to_csv(path, index=False)
    print(f"Saved {path}")


if __name__ == "__main__":  # pragma: no cover
    main()
