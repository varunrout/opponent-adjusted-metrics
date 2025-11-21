"""Augment the CxG dataset with modeled sub-model priors."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd

from opponent_adjusted.modeling.modeling_utilities import (
    get_modeling_output_dir,
    load_cxg_modeling_dataset,
)

SUBMODEL_DIR = get_modeling_output_dir("cxg", subdir="submodels")
LOGGER = logging.getLogger(__name__)

SET_PIECE_BUCKET_COLS = [
    "set_piece_category",
    "set_piece_phase",
    "score_state",
    "minute_bucket_label",
]
ASSIST_BUCKET_COLS = ["assist_category", "pass_style", "pressure_state"]
PRESSURE_BUCKET_COLS = ["pressure_state", "distance_bin", "angle_bin"]
DEF_TRIGGER_BUCKET_COLS = ["def_label", "time_gap_bucket", "possession_state"]

PRESSURE_DISTANCE_BINS = [0.0, 12.0, 18.0, 24.0, float("inf")]
PRESSURE_DISTANCE_LABELS = ["0-12", "12-18", "18-24", "24+"]
PRESSURE_ANGLE_BINS = [0.0, 20.0, 40.0, float("inf")]
PRESSURE_ANGLE_LABELS = ["0-20", "20-40", "40+"]
DEF_TIME_BINS = [0.0, 2.0, 5.0, 10.0, 30.0, float("inf")]
DEF_TIME_LABELS = ["0-2s", "2-5s", "5-10s", "10-30s", "30s+"]


def _compose_bucket(df: pd.DataFrame, cols: Iterable[str], bucket_name: str) -> pd.DataFrame:
    subset = df[list(cols)].copy()
    for col in subset.columns:
        subset[col] = subset[col].astype("object").fillna("Unknown").astype(str)
    df[bucket_name] = subset.agg("|".join, axis=1)
    return df


def _load_prior(filename: str) -> pd.DataFrame | None:
    path = SUBMODEL_DIR / filename
    if not path.exists():
        LOGGER.warning("Skipping %s because file was not found", path)
        return None
    return pd.read_csv(path)


def _merge_finishing_bias(df: pd.DataFrame) -> pd.DataFrame:
    priors = _load_prior("finishing_bias_modeled.csv")
    if priors is None:
        df["finishing_bias_logit"] = 0.0
        df["finishing_bias_multiplier"] = 1.0
        return df
    cols = ["team_id", "bias_logit", "odds_multiplier"]
    priors = priors[cols].rename(
        columns={
            "bias_logit": "finishing_bias_logit",
            "odds_multiplier": "finishing_bias_multiplier",
        }
    )
    merged = df.merge(priors, on="team_id", how="left")
    merged["finishing_bias_logit"] = merged["finishing_bias_logit"].fillna(0.0)
    merged["finishing_bias_multiplier"] = merged["finishing_bias_multiplier"].fillna(1.0)
    return merged


def _merge_concession_bias(df: pd.DataFrame) -> pd.DataFrame:
    priors = _load_prior("concession_bias_modeled.csv")
    if priors is None:
        df["concession_bias_logit"] = 0.0
        df["concession_bias_multiplier"] = 1.0
        return df
    cols = ["opponent_team_id", "bias_logit", "odds_multiplier"]
    priors = priors[cols].rename(
        columns={
            "bias_logit": "concession_bias_logit",
            "odds_multiplier": "concession_bias_multiplier",
        }
    )
    merged = df.merge(priors, on="opponent_team_id", how="left")
    merged["concession_bias_logit"] = merged["concession_bias_logit"].fillna(0.0)
    merged["concession_bias_multiplier"] = merged["concession_bias_multiplier"].fillna(1.0)
    return merged


def _merge_set_piece(df: pd.DataFrame) -> pd.DataFrame:
    priors = _load_prior("set_piece_phase_modeled.csv")
    if priors is None:
        df["set_piece_logit"] = 0.0
        df["set_piece_multiplier"] = 1.0
        df["set_piece_modeled_prob"] = np.nan
        return df
    priors = _compose_bucket(priors, SET_PIECE_BUCKET_COLS, "set_piece_bucket")
    priors = priors[[
        "set_piece_bucket",
        "bucket_logit",
        "odds_multiplier",
        "modeled_prob",
    ]].rename(
        columns={
            "bucket_logit": "set_piece_logit",
            "odds_multiplier": "set_piece_multiplier",
            "modeled_prob": "set_piece_modeled_prob",
        }
    )
    df = _compose_bucket(df, SET_PIECE_BUCKET_COLS, "set_piece_bucket")
    merged = df.merge(priors, on="set_piece_bucket", how="left")
    merged["set_piece_logit"] = merged["set_piece_logit"].fillna(0.0)
    merged["set_piece_multiplier"] = merged["set_piece_multiplier"].fillna(1.0)
    return merged


def _merge_assist_quality(df: pd.DataFrame) -> pd.DataFrame:
    priors = _load_prior("assist_quality_modeled.csv")
    if priors is None:
        df["assist_quality_logit"] = 0.0
        df["assist_quality_multiplier"] = 1.0
        df["assist_quality_modeled_prob"] = np.nan
        return df
    priors = _compose_bucket(priors, ASSIST_BUCKET_COLS, "assist_bucket")
    priors = priors[[
        "assist_bucket",
        "bucket_logit",
        "odds_multiplier",
        "modeled_prob",
    ]].rename(
        columns={
            "bucket_logit": "assist_quality_logit",
            "odds_multiplier": "assist_quality_multiplier",
            "modeled_prob": "assist_quality_modeled_prob",
        }
    )
    df = _compose_bucket(df, ASSIST_BUCKET_COLS, "assist_bucket")
    merged = df.merge(priors, on="assist_bucket", how="left")
    merged["assist_quality_logit"] = merged["assist_quality_logit"].fillna(0.0)
    merged["assist_quality_multiplier"] = merged["assist_quality_multiplier"].fillna(1.0)
    return merged


def _bucketize_pressure(df: pd.DataFrame) -> pd.DataFrame:
    distance = pd.cut(
        df["shot_distance"],
        bins=PRESSURE_DISTANCE_BINS,
        labels=PRESSURE_DISTANCE_LABELS,
        include_lowest=True,
        right=False,
    )
    angle = pd.cut(
        df["shot_angle"],
        bins=PRESSURE_ANGLE_BINS,
        labels=PRESSURE_ANGLE_LABELS,
        include_lowest=True,
        right=False,
    )
    df["distance_bin"] = distance.astype("object").fillna("Unknown").astype(str)
    df["angle_bin"] = angle.astype("object").fillna("Unknown").astype(str)
    return df


def _merge_pressure(df: pd.DataFrame) -> pd.DataFrame:
    priors = _load_prior("pressure_influence_modeled.csv")
    if priors is None:
        df["pressure_logit"] = 0.0
        df["pressure_multiplier"] = 1.0
        df["pressure_modeled_prob"] = np.nan
        return df
    priors = _compose_bucket(priors, PRESSURE_BUCKET_COLS, "pressure_bucket")
    priors = priors[[
        "pressure_bucket",
        "bucket_logit",
        "odds_multiplier",
        "modeled_prob",
    ]].rename(
        columns={
            "bucket_logit": "pressure_logit",
            "odds_multiplier": "pressure_multiplier",
            "modeled_prob": "pressure_modeled_prob",
        }
    )
    df = _bucketize_pressure(df)
    df = _compose_bucket(df, PRESSURE_BUCKET_COLS, "pressure_bucket")
    merged = df.merge(priors, on="pressure_bucket", how="left")
    merged["pressure_logit"] = merged["pressure_logit"].fillna(0.0)
    merged["pressure_multiplier"] = merged["pressure_multiplier"].fillna(1.0)
    return merged


def _bucketize_def_trigger(df: pd.DataFrame) -> pd.DataFrame:
    time_gap = pd.cut(
        df["time_gap_seconds"],
        bins=DEF_TIME_BINS,
        labels=DEF_TIME_LABELS,
        include_lowest=True,
        right=False,
    )
    df["time_gap_bucket"] = time_gap.astype("object").fillna("Unknown").astype(str)

    def _map_possession(val: object) -> str:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "Unknown"
        return "Same" if bool(val) else "Different"

    df["possession_state"] = df["possession_match"].map(_map_possession)
    return df


def _merge_defensive_trigger(df: pd.DataFrame) -> pd.DataFrame:
    priors = _load_prior("defensive_trigger_modeled.csv")
    if priors is None:
        df["def_trigger_logit"] = 0.0
        df["def_trigger_multiplier"] = 1.0
        df["def_trigger_modeled_prob"] = np.nan
        return df
    priors = _compose_bucket(priors, DEF_TRIGGER_BUCKET_COLS, "def_trigger_bucket")
    priors = priors[[
        "def_trigger_bucket",
        "bucket_logit",
        "odds_multiplier",
        "modeled_prob",
    ]].rename(
        columns={
            "bucket_logit": "def_trigger_logit",
            "odds_multiplier": "def_trigger_multiplier",
            "modeled_prob": "def_trigger_modeled_prob",
        }
    )
    df = _bucketize_def_trigger(df)
    df = _compose_bucket(df, DEF_TRIGGER_BUCKET_COLS, "def_trigger_bucket")
    merged = df.merge(priors, on="def_trigger_bucket", how="left")
    merged["def_trigger_logit"] = merged["def_trigger_logit"].fillna(0.0)
    merged["def_trigger_multiplier"] = merged["def_trigger_multiplier"].fillna(1.0)
    return merged


def enrich_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = _merge_finishing_bias(df)
    df = _merge_concession_bias(df)
    df = _merge_set_piece(df)
    df = _merge_assist_quality(df)
    df = _merge_pressure(df)
    df = _merge_defensive_trigger(df)
    return df


def main() -> None:
    dataset = load_cxg_modeling_dataset()
    enriched = enrich_dataset(dataset)

    out_dir = get_modeling_output_dir("cxg")
    parquet_path = out_dir / "cxg_dataset_enriched.parquet"
    csv_path = out_dir / "cxg_dataset_enriched.csv"

    enriched.to_parquet(parquet_path, index=False)
    enriched.to_csv(csv_path, index=False)

    print(
        "Wrote enriched CxG dataset with "
        f"{len(enriched)} rows to {parquet_path} and {csv_path}"
    )


if __name__ == "__main__":  # pragma: no cover
    main()
