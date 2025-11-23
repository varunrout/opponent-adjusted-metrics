"""Build a unified per-shot dataset for CxG modelling.

This script joins core shot features, sequence context (chains, assists,
set pieces), game state, and defensive overlay into a single table suitable
for downstream modelling. It deliberately reuses the logic already present
in the exploratory analysis modules rather than re-implementing joins from
scratch.

Usage (from repo root):

    poetry run python -m opponent_adjusted.modeling.cxg.build_cxg_dataset

The resulting parquet/CSV files are written under
`outputs/modeling/cxg/`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from opponent_adjusted.analysis.cxg_analysis import (
    pass_value_chain as pvc,
    game_state_lens as gsl,
    set_piece_lens as spl,
    assist_context as acl,
    defensive_overlay as dol,
)
from opponent_adjusted.modeling.modeling_utilities import get_modeling_output_dir


def _load_base_shots(feature_version: str = "v1") -> pd.DataFrame:
    """Load the base shot table used by multiple analysis modules.

    We rely on `pass_value_chain.load_shot_data` as the canonical source
    because it already joins `Shot`, `ShotFeature`, `Event`, and `RawEvent`.
    """

    return pvc.load_shot_data(feature_version=feature_version)


def _attach_chains(df: pd.DataFrame) -> pd.DataFrame:
    """Attach pass-value chain labels and chain-level fields.

    Reuses the `classify_chains` helper from `pass_value_chain`.
    """

    chains = pvc.classify_chains(df)
    # `classify_chains` returns the input with added columns; keep only
    # chain-specific fields to avoid duplicating everything.
    cols_to_keep = [
        "shot_id",
        "chain_label",
        "pass_style",
    ]
    chain_df = chains[cols_to_keep].drop_duplicates("shot_id")
    return df.merge(chain_df, on="shot_id", how="left")


def _attach_game_state(df: pd.DataFrame) -> pd.DataFrame:
    """Attach score and minute state features.

    Uses `game_state_lens.prepare_game_state_features` which consumes
    the same columns as `pass_value_chain` outputs (score_diff_at_shot,
    is_leading/is_trailing/is_drawing, minute, feature_minute_bucket).
    """

    enriched = gsl.prepare_game_state_features(df)
    cols = [
        "shot_id",
        "score_state",
        "simple_state",
        "minute_bucket_label",
    ]
    state_df = enriched[cols].drop_duplicates("shot_id")
    return df.merge(state_df, on="shot_id", how="left")


def _attach_set_piece(df: pd.DataFrame, feature_version: str = "v1") -> pd.DataFrame:
    """Attach set-piece category and phase for each shot.

    We call `set_piece_lens.load_shot_data` separately and then map
    its identifiers back to the base `shot_id` domain.
    """

    sp_df = spl.load_shot_data(feature_version=feature_version)
    if not sp_df.empty:
        sp_df = spl.prepare_set_piece_features(sp_df)
    cols = [
        "shot_id",
        "set_piece_category",
        "set_piece_phase",
    ]
    sp_df = sp_df[cols].drop_duplicates("shot_id")
    return df.merge(sp_df, on="shot_id", how="left")


def _attach_assist_context(df: pd.DataFrame, feature_version: str = "v1") -> pd.DataFrame:
    """Attach assist-category level context for each shot.

    The assist_context module works off its own loader, so we join
    only the high-level categorical context back to the base table.
    """

    ac_df = acl.load_shot_and_assist_data(feature_version=feature_version)
    cols = [
        "shot_id",
        "assist_category",
        "under_pressure",
    ]
    if "assist_category" not in ac_df.columns:
        # If the analysis module has not yet emitted an explicit category,
        # we can still reuse the raw play patterns later; for now, skip.
        return df

    ac_df = ac_df[cols].drop_duplicates("shot_id")
    # Rename under_pressure to explicit pressure_state string for modelling.
    ac_df["pressure_state"] = ac_df["under_pressure"].map(
        {True: "Under pressure", False: "Not under pressure"}
    )
    ac_df = ac_df.drop(columns=["under_pressure"])
    return df.merge(ac_df, on="shot_id", how="left")


def _attach_defensive_overlay(df: pd.DataFrame, feature_version: str = "v1") -> pd.DataFrame:
    """Attach defensive trigger labels from the defensive overlay analysis.

    We construct the defensive overlay dataset and then join by shot_id.
    """

    overlay_df = dol.build_overlay_dataset(feature_version=feature_version)
    cols = [
        "shot_id",
        "def_label",
        "seconds_since_def_action",
        "possession_alignment",
    ]
    missing_cols = [col for col in cols if col not in overlay_df.columns]
    if missing_cols:
        for col in missing_cols:
            overlay_df[col] = pd.NA
    overlay_df = overlay_df[cols].drop_duplicates("shot_id")
    overlay_df = overlay_df.rename(
        columns={
            "seconds_since_def_action": "time_gap_seconds",
            "possession_alignment": "possession_match",
        }
    )
    return df.merge(overlay_df, on="shot_id", how="left")


def build_cxg_dataset(feature_version: str = "v1") -> pd.DataFrame:
    """Construct the full CxG modelling dataset as a pandas DataFrame."""

    df = _load_base_shots(feature_version=feature_version)
    df = _attach_chains(df)
    df = _attach_game_state(df)
    df = _attach_set_piece(df, feature_version=feature_version)
    df = _attach_assist_context(df, feature_version=feature_version)
    df = _attach_defensive_overlay(df, feature_version=feature_version)

    # Minimal selection of columns likely to be used in modelling; we
    # retain IDs so the table can be joined to other artefacts.
    keep_cols = [
        "shot_id",
        "match_id",
        "team_id",
        "opponent_team_id",
        "player_id",
        "is_home",
        "match_date",
        "competition_id",
        "competition_name",
        "competition_season",
        "competition_statsbomb_id",
        "statsbomb_xg",
        "is_goal",
        "shot_distance",
        "shot_angle",
        "location_x",
        "location_y",
        "shot_body_part",
        "shot_outcome",
        "score_diff_at_shot",
        "is_leading",
        "is_trailing",
        "is_drawing",
        "minute",
        "feature_minute_bucket",
        "score_state",
        "simple_state",
        "minute_bucket_label",
        "chain_label",
        "pass_style",
        "assist_category",
        "pressure_state",
        "set_piece_category",
        "set_piece_phase",
        "def_label",
        "time_gap_seconds",
        "possession_match",
    ]

    existing = [c for c in keep_cols if c in df.columns]
    return df[existing].copy()


def main() -> None:
    out_dir = get_modeling_output_dir("cxg")

    dataset = build_cxg_dataset(feature_version="v1")

    parquet_path = out_dir / "cxg_dataset.parquet"
    csv_path = out_dir / "cxg_dataset.csv"

    dataset.to_parquet(parquet_path, index=False)
    dataset.to_csv(csv_path, index=False)

    print(f"Wrote CxG dataset with {len(dataset)} rows to {parquet_path} and {csv_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
