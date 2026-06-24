"""cXA-xG Scorer and Credit Allocator.

This module provides the cXA-xG metric:
- Trains a scorer predicting is_final_action_before_shot on ALL shots
- Allocates credit via softmax over log-odds within each shot's action window
- Weights by shot xG (cXA-xG) or 1.0 for goals (cXA-Goals)

The key insight: by training on all shots (not just goals), we get a stable
scorer that represents "how likely is this action to be the immediate setup
for any shot attempt?" - i.e., chance creation regardless of outcome.

Output modes:
- cXA-xG: Sum of player credit weighted by shot xG. Sums to total xG.
- cXA-Goals: Sum of player credit weighted by 1.0 for goals only. Sums to goal count.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

# Features for the scorer (per-action)
ACTION_FEATURES = [
    "end_x",
    "end_y",
    "distance_to_goal",
    "angle_to_goal",
    "is_pass",
    "is_carry",
    "is_dribble",
    "is_into_box",
    "under_pressure",
    "seconds_to_shot",
]


def _get_repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def load_shot_windows(path: Optional[Path] = None) -> pd.DataFrame:
    """Load the shot action windows parquet."""
    if path is None:
        path = _get_repo_root() / "feature_store" / "cxa" / "shot_action_windows.parquet"
    return pd.read_parquet(path)


def _melt_windows_to_actions(windows_df: pd.DataFrame, max_actions: int = 8) -> pd.DataFrame:
    """Convert wide window format to long action format for training.

    Returns a DataFrame with one row per action, including the shot context.
    """
    action_rows = []

    for _, row in windows_df.iterrows():
        shot_id = row["shot_id"]
        num_actions = int(row["num_actions"])
        statsbomb_xg = row["statsbomb_xg"]
        is_goal = row["is_goal"]

        for i in range(1, num_actions + 1):
            prefix = f"action{i}_"

            action_row = {
                "shot_id": shot_id,
                "action_position": i,  # 1 = closest to shot
                "is_final_action": i == 1,  # Target: is this the last action before shot?
                "statsbomb_xg": statsbomb_xg,
                "is_goal": is_goal,
                "player_id": row.get(f"{prefix}player_id"),
                "action_type": row.get(f"{prefix}type"),
            }

            # Features
            for feat in ACTION_FEATURES:
                col = f"{prefix}{feat}"
                val = row.get(col)
                action_row[feat] = val if pd.notna(val) else 0.0

            action_rows.append(action_row)

    return pd.DataFrame(action_rows)


class CxAScorer:
    """Scorer for cXA that predicts is_final_action_before_shot.

    The scorer learns which actions are most likely to be the immediate
    setup for a shot. We then use softmax over log-odds to allocate
    credit within each shot's action window.
    """

    def __init__(self):
        self.pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(max_iter=500, random_state=42)),
            ]
        )
        self._is_fitted = False

    def fit(self, windows_df: pd.DataFrame) -> "CxAScorer":
        """Fit the scorer on all shots' action windows.

        Args:
            windows_df: Wide-format shot windows from shot_action_windows.py
        """
        # Melt to long format
        actions_df = _melt_windows_to_actions(windows_df)

        logger.info(
            f"Training scorer on {len(actions_df):,} actions from {windows_df['shot_id'].nunique():,} shots"
        )

        # Prepare features and target
        X = actions_df[ACTION_FEATURES].fillna(0).values
        y = actions_df["is_final_action"].astype(int).values

        self.pipeline.fit(X, y)
        self._is_fitted = True

        # Log some stats
        y_pred = self.pipeline.predict(X)
        accuracy = (y_pred == y).mean()
        logger.info(f"  Training accuracy: {accuracy:.3f}")

        return self

    def score_actions(self, actions_df: pd.DataFrame) -> np.ndarray:
        """Score actions, returning log-odds (logit) values.

        Higher log-odds = more likely to be the final action before shot.
        """
        if not self._is_fitted:
            raise RuntimeError("Scorer not fitted. Call fit() first.")

        X = actions_df[ACTION_FEATURES].fillna(0).values

        # Get log-odds (logit) via log(p / (1-p))
        proba = self.pipeline.predict_proba(X)[:, 1]

        # Clip to avoid log(0) or log(inf)
        proba = np.clip(proba, 1e-6, 1 - 1e-6)
        log_odds = np.log(proba / (1 - proba))

        return log_odds


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()


def allocate_credit(
    windows_df: pd.DataFrame,
    scorer: CxAScorer,
    mode: str = "xG",
) -> pd.DataFrame:
    """Allocate creation credit across actions for each shot.

    Args:
        windows_df: Wide-format shot windows
        scorer: Fitted CxAScorer
        mode: "xG" to weight by statsbomb_xg, "goals" to weight by is_goal (1/0)

    Returns:
        DataFrame with columns: shot_id, action_position, player_id, action_type,
        raw_score, credit_share, weighted_credit
    """
    # Melt to long
    actions_df = _melt_windows_to_actions(windows_df)

    if actions_df.empty:
        return pd.DataFrame()

    # Score all actions
    actions_df["raw_score"] = scorer.score_actions(actions_df)

    # Allocate credit per shot via softmax
    credits = []

    for shot_id, group in actions_df.groupby("shot_id"):
        scores = group["raw_score"].values
        shares = _softmax(scores)

        shot_row = windows_df[windows_df["shot_id"] == shot_id].iloc[0]

        if mode == "xG":
            weight = float(shot_row["statsbomb_xg"]) if pd.notna(shot_row["statsbomb_xg"]) else 0.0
        else:  # goals
            weight = 1.0 if shot_row["is_goal"] else 0.0

        for i, (idx, row) in enumerate(group.iterrows()):
            credits.append(
                {
                    "shot_id": shot_id,
                    "action_position": row["action_position"],
                    "player_id": row["player_id"],
                    "action_type": row["action_type"],
                    "raw_score": row["raw_score"],
                    "credit_share": shares[i],
                    "weighted_credit": shares[i] * weight,
                    "is_goal": shot_row["is_goal"],
                    "statsbomb_xg": shot_row["statsbomb_xg"],
                }
            )

    return pd.DataFrame(credits)


def compute_cxa_xg(
    windows_df: pd.DataFrame,
    scorer: Optional[CxAScorer] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute cXA-xG and cXA-Goals from shot windows.

    Args:
        windows_df: Wide-format shot windows
        scorer: Pre-fitted scorer (will train one if None)

    Returns:
        Tuple of (cxa_xg_credits, cxa_goals_credits) DataFrames
    """
    if scorer is None:
        scorer = CxAScorer()
        scorer.fit(windows_df)

    logger.info("Computing cXA-xG (all shots weighted by xG)...")
    cxa_xg = allocate_credit(windows_df, scorer, mode="xG")

    logger.info("Computing cXA-Goals (goals only, weight=1)...")
    cxa_goals = allocate_credit(windows_df, scorer, mode="goals")

    return cxa_xg, cxa_goals


def player_leaderboard(
    credits_df: pd.DataFrame,
    mode: str = "xG",
) -> pd.DataFrame:
    """Aggregate credits to player level.

    Args:
        credits_df: Output from allocate_credit()
        mode: "xG" or "goals" (for column naming)

    Returns:
        DataFrame with player totals, sorted descending
    """
    agg = (
        credits_df.groupby("player_id")
        .agg(
            total_credit=("weighted_credit", "sum"),
            num_actions=("shot_id", "count"),
            num_shots=("shot_id", "nunique"),
        )
        .reset_index()
    )

    # Rename for clarity
    credit_col = f"cXA_{mode}"
    agg = agg.rename(columns={"total_credit": credit_col})

    return agg.sort_values(credit_col, ascending=False)


def action_type_summary(credits_df: pd.DataFrame, mode: str = "xG") -> pd.DataFrame:
    """Summarize credit by action type."""
    agg = (
        credits_df.groupby("action_type")
        .agg(
            total_credit=("weighted_credit", "sum"),
            num_actions=("shot_id", "count"),
            mean_share=("credit_share", "mean"),
        )
        .reset_index()
    )

    total = agg["total_credit"].sum()
    agg["pct_of_total"] = 100.0 * agg["total_credit"] / total if total > 0 else 0.0

    return agg.sort_values("total_credit", ascending=False)


def main() -> int:
    """Run cXA-xG computation and print summary."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    # Load windows
    windows_df = load_shot_windows()
    logger.info(f"Loaded {len(windows_df):,} shot windows")

    # Compute
    cxa_xg, cxa_goals = compute_cxa_xg(windows_df)

    # Summaries
    logger.info("\n=== cXA-xG Summary ===")
    logger.info(f"Total xG attributed: {cxa_xg['weighted_credit'].sum():.2f}")
    logger.info(f"Total shots: {windows_df['shot_id'].nunique():,}")

    xg_by_type = action_type_summary(cxa_xg, "xG")
    logger.info("\nCredit by action type (cXA-xG):")
    print(xg_by_type.to_string(index=False))

    logger.info("\n=== cXA-Goals Summary ===")
    logger.info(f"Total goals attributed: {cxa_goals['weighted_credit'].sum():.2f}")

    goals_by_type = action_type_summary(cxa_goals, "goals")
    logger.info("\nCredit by action type (cXA-Goals):")
    print(goals_by_type.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
