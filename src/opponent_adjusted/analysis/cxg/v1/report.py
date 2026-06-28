"""DB-backed CxG v1 football analysis report."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from opponent_adjusted.analysis.shared.io import ensure_dir, write_csv, write_markdown
from opponent_adjusted.analysis.shared.loaders import (
    load_player_aggregates,
    load_shots_with_predictions,
    load_team_aggregates,
    resolve_latest_model_version,
)
from opponent_adjusted.analysis.shared.plots import save_histogram, save_scatter

DEFAULT_OUTPUT_DIR = Path("outputs") / "analysis" / "cxg"


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _mean(df: pd.DataFrame, column: str) -> float | None:
    if df.empty or column not in df:
        return None
    value = _numeric(df[column]).mean()
    return None if pd.isna(value) else float(value)


def _sum(df: pd.DataFrame, column: str) -> float:
    if df.empty or column not in df:
        return 0.0
    return float(_numeric(df[column]).fillna(0).sum())


def _count_goals(df: pd.DataFrame) -> int:
    if df.empty or "is_goal" not in df:
        return 0
    return int(_numeric(df["is_goal"]).fillna(0).sum())


def _summary_frame(values: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame([values])


def _distribution_summary(series: pd.Series, metric_name: str) -> pd.DataFrame:
    values = _numeric(series).dropna()
    if values.empty:
        return pd.DataFrame(
            [
                {
                    "metric": metric_name,
                    "count": 0,
                    "mean": None,
                    "std": None,
                    "min": None,
                    "p25": None,
                    "median": None,
                    "p75": None,
                    "max": None,
                }
            ]
        )
    return pd.DataFrame(
        [
            {
                "metric": metric_name,
                "count": int(values.count()),
                "mean": float(values.mean()),
                "std": float(values.std()) if values.count() > 1 else 0.0,
                "min": float(values.min()),
                "p25": float(values.quantile(0.25)),
                "median": float(values.quantile(0.5)),
                "p75": float(values.quantile(0.75)),
                "max": float(values.max()),
            }
        ]
    )


def _outcome_summary(shots: pd.DataFrame) -> pd.DataFrame:
    if shots.empty or "outcome" not in shots:
        return pd.DataFrame(columns=["outcome", "shots", "goals", "goal_rate", "mean_cxg"])
    grouped = (
        shots.groupby("outcome", dropna=False)
        .agg(
            shots=("shot_id", "count"),
            goals=("is_goal", "sum"),
            mean_cxg=("cxg_raw", "mean"),
        )
        .reset_index()
    )
    grouped["goal_rate"] = grouped["goals"] / grouped["shots"].clip(lower=1)
    return grouped[["outcome", "shots", "goals", "goal_rate", "mean_cxg"]].sort_values(
        "shots", ascending=False
    )


def _slice_summary(shots: pd.DataFrame, column: str) -> pd.DataFrame:
    return (
        shots.groupby(column, dropna=False)
        .agg(
            shots=("shot_id", "count"),
            goals=("is_goal", "sum"),
            mean_cxg=("cxg_raw", "mean"),
            mean_provider_xg=("provider_xg", "mean"),
        )
        .reset_index()
        .assign(goal_rate=lambda frame: frame["goals"] / frame["shots"].clip(lower=1))
        .sort_values("shots", ascending=False)
    )


def _prepare_player_aggregates(shots: pd.DataFrame, db_path: Path | None) -> pd.DataFrame:
    players = load_player_aggregates("cxg", db_path=db_path)
    if players.empty and not shots.empty and "player_id" in shots:
        players = (
            shots.groupby("player_id", dropna=False)
            .agg(
                shots_count=("shot_id", "count"),
                goals=("is_goal", "sum"),
                summed_cxg=("cxg_raw", "sum"),
            )
            .reset_index()
        )
    if players.empty:
        return players
    if "player_name" not in players:
        players["player_name"] = players.get("player_id", pd.Series(dtype=object)).astype(str)
    players["avg_cxg_per_shot"] = _numeric(players.get("summed_cxg", 0)) / _numeric(
        players.get("shots_count", 0)
    ).replace(0, pd.NA)
    return players


def _prepare_team_aggregates(shots: pd.DataFrame, db_path: Path | None) -> pd.DataFrame:
    teams = load_team_aggregates("cxg", db_path=db_path)
    if teams.empty and not shots.empty and "team_id" in shots:
        teams = (
            shots.groupby("team_id", dropna=False)
            .agg(
                shots_count=("shot_id", "count"),
                goals=("is_goal", "sum"),
                summed_cxg=("cxg_raw", "sum"),
            )
            .reset_index()
        )
    if teams.empty:
        return teams
    if "team_name" not in teams:
        teams["team_name"] = teams.get("team_id", pd.Series(dtype=object)).astype(str)
    teams["avg_cxg_per_shot"] = _numeric(teams.get("summed_cxg", 0)) / _numeric(
        teams.get("shots_count", 0)
    ).replace(0, pd.NA)
    return teams


def _top_table(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    if df.empty or value_col not in df:
        return df
    return df.sort_values(value_col, ascending=False).head(50)


def _markdown_report(
    *,
    model_version: str | None,
    shots: pd.DataFrame,
    skipped_slices: list[str],
    player_rows: int,
    team_rows: int,
) -> str:
    shot_count = len(shots)
    goal_count = _count_goals(shots)
    goal_rate = goal_count / shot_count if shot_count else 0.0
    mean_cxg = _mean(shots, "cxg_raw")
    mean_adjustment = _mean(shots, "cxg_opp_adjusted_diff")
    return f"""# CxG V1 Analysis Report

## Dataset Coverage

- Model version: `{model_version or "unavailable"}`
- Shots analysed: {shot_count:,}
- Goals: {goal_count:,}
- Goal rate: {goal_rate:.4f}
- Mean CxG: {mean_cxg if mean_cxg is not None else "unavailable"}

## Football Questions

- How many shots and goals are in the current v1 CxG output?
- How is CxG distributed across shots?
- How large are opponent/context neutralization adjustments?
- Which players and teams combine shot volume with shot quality?
- Which optional slices are available from the DB-backed feature layer?

## CxG Distribution

The histogram in `distributions/plots/cxg_distribution.png` shows shot-quality spread. A healthy CxG distribution should contain many low-value shots and fewer high-value chances.

## Opponent Adjustment Distribution

Mean opponent/context adjustment difference: {mean_adjustment if mean_adjustment is not None else "unavailable"}.

The histogram in `distributions/plots/opponent_adjustment_distribution.png` shows how far observed-context CxG differs from neutralized CxG.

## Players And Teams

- Player aggregate rows: {player_rows:,}
- Team aggregate rows: {team_rows:,}

Player and team scatter plots compare shot volume with average CxG per shot.

## Skipped Optional Slices

{", ".join(skipped_slices) if skipped_slices else "No optional slices were skipped."}

## Limitations

- This report reads local SQLite model-output tables and is only as current as the last generated run.
- CxG is an event-data baseline and does not use tracking data.
- Opponent adjustment fields are model/context outputs, not causal opponent-defence estimates.
- Missing optional columns produce skipped slice tables rather than failed analysis runs.
"""


def run_cxg_analysis(
    output_dir: Path | None = None,
    db_path: Path | None = None,
) -> dict[str, Any]:
    """Run the DB-backed CxG v1 analysis report."""

    output_root = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_DIR
    ensure_dir(output_root)
    model_version = resolve_latest_model_version("cxg", db_path=db_path)
    shots = load_shots_with_predictions(model_version=model_version, db_path=db_path)

    paths: dict[str, str] = {}
    skipped_slices: list[str] = []

    population = _summary_frame(
        {
            "model_version": model_version,
            "shot_count": len(shots),
            "goal_count": _count_goals(shots),
            "goal_rate": _count_goals(shots) / len(shots) if len(shots) else 0.0,
            "mean_cxg": _mean(shots, "cxg_raw"),
            "mean_provider_xg": _mean(shots, "provider_xg"),
            "mean_neutral_cxg": _mean(shots, "cxg_neutral"),
        }
    )
    paths["shot_population_summary"] = str(
        write_csv(population, output_root / "eda" / "tables" / "shot_population_summary.csv")
    )
    paths["shot_outcome_summary"] = str(
        write_csv(
            _outcome_summary(shots),
            output_root / "eda" / "tables" / "shot_outcome_summary.csv",
        )
    )

    paths["cxg_distribution_summary"] = str(
        write_csv(
            _distribution_summary(shots.get("cxg_raw", pd.Series(dtype=float)), "cxg_raw"),
            output_root / "distributions" / "tables" / "cxg_distribution_summary.csv",
        )
    )
    paths["opponent_adjustment_summary"] = str(
        write_csv(
            _distribution_summary(
                shots.get("cxg_opp_adjusted_diff", pd.Series(dtype=float)),
                "cxg_opp_adjusted_diff",
            ),
            output_root / "distributions" / "tables" / "opponent_adjustment_summary.csv",
        )
    )
    paths["cxg_distribution_plot"] = str(
        save_histogram(
            shots.get("cxg_raw", pd.Series(dtype=float)),
            output_root / "distributions" / "plots" / "cxg_distribution.png",
            "CxG Distribution",
            "CxG",
        )
    )
    paths["opponent_adjustment_distribution_plot"] = str(
        save_histogram(
            shots.get("cxg_opp_adjusted_diff", pd.Series(dtype=float)),
            output_root / "distributions" / "plots" / "opponent_adjustment_distribution.png",
            "Opponent Adjustment Distribution",
            "Raw CxG - Neutral CxG",
        )
    )

    slice_specs = {
        "by_body_part": "body_part",
        "by_pressure": "under_pressure",
        "by_minute_bucket": "minute_bucket",
        "by_opponent": "opponent_team_id",
    }
    for filename, column in slice_specs.items():
        if shots.empty or column not in shots or shots[column].isna().all():
            skipped_slices.append(filename)
            continue
        paths[filename] = str(
            write_csv(
                _slice_summary(shots, column),
                output_root / "slices" / "tables" / f"{filename}.csv",
            )
        )

    players = _prepare_player_aggregates(shots, db_path)
    top_players = _top_table(players, "summed_cxg")
    paths["top_players_by_cxg"] = str(
        write_csv(top_players, output_root / "players" / "tables" / "top_players_by_cxg.csv")
    )
    paths["player_shot_quality_vs_volume"] = str(
        write_csv(
            players,
            output_root / "players" / "tables" / "shot_quality_vs_volume.csv",
        )
    )
    paths["player_shot_quality_vs_volume_plot"] = str(
        save_scatter(
            players,
            "shots_count",
            "avg_cxg_per_shot",
            output_root / "players" / "plots" / "player_shot_quality_vs_volume.png",
            "Player Shot Quality vs Volume",
            xlabel="Shots",
            ylabel="Average CxG per shot",
        )
    )

    teams = _prepare_team_aggregates(shots, db_path)
    top_teams = _top_table(teams, "summed_cxg")
    paths["top_teams_by_cxg"] = str(
        write_csv(top_teams, output_root / "teams" / "tables" / "top_teams_by_cxg.csv")
    )
    paths["team_quality_vs_volume"] = str(
        write_csv(teams, output_root / "teams" / "tables" / "team_quality_vs_volume.csv")
    )
    paths["team_shot_quality_vs_volume_plot"] = str(
        save_scatter(
            teams,
            "shots_count",
            "avg_cxg_per_shot",
            output_root / "teams" / "plots" / "team_shot_quality_vs_volume.png",
            "Team Shot Quality vs Volume",
            xlabel="Shots",
            ylabel="Average CxG per shot",
        )
    )

    report = _markdown_report(
        model_version=model_version,
        shots=shots,
        skipped_slices=skipped_slices,
        player_rows=len(players),
        team_rows=len(teams),
    )
    paths["report"] = str(write_markdown(report, output_root / "report.md"))

    return {
        "output_dir": str(output_root),
        "model_version": model_version,
        "shot_count": len(shots),
        "skipped_slices": skipped_slices,
        "paths": paths,
    }
