"""Team finishing and opponent concession bias summaries."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from opponent_adjusted.db.models import Team
from opponent_adjusted.db.session import SessionLocal
from opponent_adjusted.modeling.modeling_utilities import (
    get_modeling_output_dir,
    load_cxg_modeling_dataset,
)


def _team_lookup() -> dict[int, str]:
    with SessionLocal() as session:
        rows = session.query(Team.id, Team.name).all()
    return {row.id: row.name for row in rows}


def _summarize(df: pd.DataFrame, group_col: str, label: str) -> pd.DataFrame:
    agg = (
        df.groupby(group_col, observed=False)
        .agg(
            shots=("shot_id", "count"),
            goals=("is_goal", "sum"),
            xg_total=("statsbomb_xg", "sum"),
        )
        .reset_index()
    )
    agg["goal_rate"] = agg["goals"] / agg["shots"].clip(lower=1)
    agg["xg_per_shot"] = agg["xg_total"] / agg["shots"].clip(lower=1)
    agg["lift_vs_xg"] = agg["goal_rate"] - agg["xg_per_shot"]
    lookup = _team_lookup()
    agg[label] = agg[group_col].map(lookup)
    agg = agg[[label, group_col, "shots", "goals", "xg_total", "goal_rate", "xg_per_shot", "lift_vs_xg"]]
    agg = agg.sort_values("lift_vs_xg", ascending=False)
    return agg


def main() -> None:
    df = load_cxg_modeling_dataset()
    out_dir = get_modeling_output_dir("cxg", subdir="submodels")
    out_dir.mkdir(parents=True, exist_ok=True)

    team_summary = _summarize(df, "team_id", "team_name")
    opp_summary = _summarize(df, "opponent_team_id", "opponent_name")

    team_path = out_dir / "finishing_bias_by_team.csv"
    opp_path = out_dir / "concession_bias_by_opponent.csv"
    team_summary.to_csv(team_path, index=False)
    opp_summary.to_csv(opp_path, index=False)
    print(f"Saved {team_path} and {opp_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
