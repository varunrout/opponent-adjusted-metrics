"""Aggregate scored CxG predictions into reporting tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _load_scored_dataset(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def aggregate_to_matches(scored_path: Path, output_path: Path) -> pd.DataFrame:
    df = _load_scored_dataset(scored_path)
    required = {"match_id", "team_id", "opponent_team_id", "is_goal", "cxg_prediction", "statsbomb_xg"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns in scored dataset: {missing}")

    grouped = (
        df.groupby(["match_id", "team_id", "opponent_team_id"], as_index=False)
        .agg(
            cxg_total=("cxg_prediction", "sum"),
            goals=("is_goal", "sum"),
            provider_xg=("statsbomb_xg", "sum"),
            shots=("shot_id", "count"),
        )
    )
    grouped["cxg_delta"] = grouped["goals"] - grouped["cxg_total"]
    grouped["provider_delta"] = grouped["goals"] - grouped["provider_xg"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(output_path, index=False)
    print(f"Saved match aggregates to {output_path}")
    return grouped


def aggregate_to_table(match_agg: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    for_side = (
        match_agg.groupby("team_id", as_index=False)
        .agg(
            matches=("match_id", "nunique"),
            shots_for=("shots", "sum"),
            cxg_for=("cxg_total", "sum"),
            provider_xg_for=("provider_xg", "sum"),
            goals_for=("goals", "sum"),
        )
        .rename(columns={"team_id": "team_id"})
    )

    against_side = (
        match_agg.groupby("opponent_team_id", as_index=False)
        .agg(
            shots_against=("shots", "sum"),
            cxg_against=("cxg_total", "sum"),
            provider_xg_against=("provider_xg", "sum"),
            goals_against=("goals", "sum"),
        )
        .rename(columns={"opponent_team_id": "team_id"})
    )

    aggregations = for_side.merge(against_side, on="team_id", how="left")
    for col in ("shots_against", "cxg_against", "provider_xg_against", "goals_against"):
        aggregations[col] = aggregations[col].fillna(0)

    aggregations["cxg_diff"] = aggregations["cxg_for"] - aggregations["cxg_against"]
    aggregations["goals_minus_cxg"] = aggregations["goals_for"] - aggregations["cxg_for"]
    aggregations["goal_diff"] = aggregations["goals_for"] - aggregations["goals_against"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    aggregations.to_csv(output_path, index=False)
    print(f"Saved team aggregates to {output_path}")
    return aggregations


if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="Aggregate scored CxG to matches and table")
    parser.add_argument("scored", type=Path, help="Path to scored dataset")
    parser.add_argument("match_output", type=Path, help="CSV path for match aggregates")
    parser.add_argument("team_output", type=Path, help="CSV path for team aggregates")
    args = parser.parse_args()

    match_df = aggregate_to_matches(args.scored, args.match_output)
    aggregate_to_table(match_df, args.team_output)
