"""xA+ (Actions) - Softmax credit distribution across action sequences.

For sequences ending in a goal (is_goal=True), distribute 1.0 credit across
all actions (passes, carries, dribbles) using softmax weighting.

Carry → Pass₁ → Dribble → Pass₂ (assist) → Shot → GOAL
  ↓       ↓        ↓          ↓
 0.10    0.20     0.15       0.55  (softmax weights, sum=1.0)

Target: is_goal (bool) - used to identify goal-scoring sequences
Input: action_sequences.parquet (without opposition metrics)
"""

from __future__ import annotations

import logging
from typing import Tuple, Optional, List

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def distance_to_goal(x: float, y: float) -> float:
    """Calculate distance from (x, y) to goal center (120, 40)."""
    return np.sqrt((120 - x) ** 2 + (40 - y) ** 2)


class XAPlusActionModel:
    """xA+ Action Model: Softmax credit distribution across action sequences."""

    def __init__(self, temperature: float = 1.0):
        """
        Args:
            temperature: Softmax temperature. Lower = more peaked distribution.
        """
        self.scorer = None
        self.scaler = StandardScaler()
        self.temperature = temperature
        self.feature_weights: Optional[np.ndarray] = None
        self.feature_names: List[str] = []
        self.is_fitted = False

    def _extract_actions(self, sequences_df: pd.DataFrame) -> pd.DataFrame:
        """Extract individual actions from wide-format sequences.

        Converts action1_*, action2_*, ... columns to long format.
        """

        def _get(obj, name: str, default=None):
            return getattr(obj, name) if hasattr(obj, name) else default

        actions_list = []

        # itertuples is much faster than iterrows for wide frames
        for seq_idx, seq in enumerate(sequences_df.itertuples(index=False)):
            is_goal = bool(_get(seq, "is_goal", False))
            sequence_id = _get(seq, "sequence_id", seq_idx)
            num_actions = int(_get(seq, "num_actions", 1) or 1)
            shot_id = _get(seq, "shot_id")

            for i in range(1, min(num_actions + 1, 6)):
                action_type = _get(seq, f"action{i}_type")
                if pd.isna(action_type):
                    continue

                # Action 1 is the assist (closest to shot)
                is_assist = (i == 1) and is_goal

                end_x = _get(seq, f"action{i}_end_x", 60)
                end_y = _get(seq, f"action{i}_end_y", 40)
                player_id = _get(seq, f"action{i}_player_id")
                player_name = _get(seq, f"action{i}_player_name")

                actions_list.append(
                    {
                        "sequence_id": sequence_id,
                        "sequence_idx": seq_idx,
                        "shot_id": shot_id,
                        "action_num": i,
                        "action_type": action_type,
                        "is_goal": is_goal,
                        "is_assist": is_assist,
                        "player_id": player_id,
                        "player_name": player_name,
                        "start_x": _get(seq, f"action{i}_start_x", 50),
                        "start_y": _get(seq, f"action{i}_start_y", 40),
                        "end_x": end_x,
                        "end_y": end_y,
                        "is_cross": _get(seq, f"action{i}_is_cross", False),
                        "is_through_ball": _get(seq, f"action{i}_is_through_ball", False),
                        "under_pressure": _get(seq, f"action{i}_under_pressure", False),
                        "distance_to_goal": distance_to_goal(float(end_x), float(end_y)),
                        "is_pass": action_type == "Pass",
                        "is_carry": action_type == "Carry",
                        "is_dribble": action_type == "Dribble",
                    }
                )

        return pd.DataFrame(actions_list)

    def _prepare_features(self, actions_df: pd.DataFrame) -> np.ndarray:
        """Prepare feature matrix from actions DataFrame."""
        # Feature columns
        feature_cols = [
            "end_x",
            "end_y",
            "distance_to_goal",
            "is_cross",
            "is_through_ball",
            "under_pressure",
            "is_pass",
            "is_carry",
            "is_dribble",
        ]

        # Ensure all columns exist (vectorized, avoids per-column frame mutation overhead)
        df = actions_df.copy()
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0

        X_df = df[feature_cols].fillna(0)
        X_df = X_df.astype(float)

        self.feature_names = feature_cols
        return X_df.values

    def fit(self, actions_df: pd.DataFrame) -> "XAPlusActionModel":
        """Fit the action scoring model.

        Uses logistic regression to learn which action features predict
        being the assist action (is_assist=True).

        Args:
            actions_df: Actions DataFrame with is_assist column

        Returns:
            Self for chaining
        """
        logger.info("Fitting xA+ Action Model (softmax weighting)...")

        df = actions_df.copy()

        num_actions = len(df)
        num_assists = df["is_assist"].sum()
        logger.info(f"  Total actions: {num_actions:,}")
        logger.info(f"  Assist actions: {num_assists}")

        # Prepare features
        X = self._prepare_features(df)
        y = df["is_assist"].astype(int).values

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Train logistic regression
        self.scorer = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
        self.scorer.fit(X_scaled, y)

        # Store feature weights
        self.feature_weights = self.scorer.coef_[0]

        logger.info("  Feature weights (contribution to assist probability):")
        for i, feat in enumerate(self.feature_names):
            logger.info(f"    {feat}: {self.feature_weights[i]:.3f}")

        self.is_fitted = True
        return self

    def score_actions(self, actions_df: pd.DataFrame) -> np.ndarray:
        """Score each action based on learned feature weights."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X = self._prepare_features(actions_df)
        X_scaled = self.scaler.transform(X)

        scores = self.scorer.predict_proba(X_scaled)[:, 1]
        return scores

    def distribute_credit(self, sequence_actions: pd.DataFrame) -> np.ndarray:
        """Distribute 1.0 credit across actions in a sequence using softmax.

        Args:
            sequence_actions: DataFrame of actions in one sequence

        Returns:
            Array of credits (sum=1.0)
        """
        if len(sequence_actions) == 0:
            return np.array([])

        scores = self.score_actions(sequence_actions)

        # Apply softmax with temperature
        exp_scores = np.exp(scores / self.temperature)
        weights = exp_scores / exp_scores.sum()

        return weights


def compute_xa_plus_actions(
    sequences_df: pd.DataFrame,
    temperature: float = 1.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, XAPlusActionModel]:
    """Compute xA+ for actions using softmax credit distribution.

    Only sequences ending in GOALS (is_goal=True) receive credit.

    Args:
        sequences_df: DataFrame from action_sequences.parquet
        temperature: Softmax temperature

    Returns:
        Tuple of:
        - sequences DataFrame with xa_plus columns added
        - actions DataFrame (long format) with xa_plus column
        - fitted model
    """
    logger.info("Computing xA+ (Actions)...")

    df = sequences_df.copy()

    # Initialize model
    model = XAPlusActionModel(temperature=temperature)

    # Extract actions to long format
    actions_df = model._extract_actions(df)
    logger.info(f"  Extracted {len(actions_df)} actions from {len(df)} sequences")

    # Count action types
    action_counts = actions_df["action_type"].value_counts()
    logger.info("  Action types:")
    for atype, count in action_counts.items():
        logger.info(f"    {atype}: {count}")

    # Fit model on all actions
    model.fit(actions_df)

    # Score all actions once (avoid per-sequence feature prep)
    actions_df["score"] = model.score_actions(actions_df)

    # Initialize xA+ columns
    actions_df["xa_plus"] = 0.0
    for i in range(1, 6):
        df[f"xa_plus_action{i}"] = 0.0

    # Get goal-scoring sequences
    goal_seq_indices = df[df["is_goal"].fillna(False).astype(bool)].index
    num_goals = len(goal_seq_indices)
    logger.info(f"  Goal-scoring sequences: {num_goals}")

    # Distribute credit across actions per goal sequence using softmax over scores
    goal_actions = actions_df[actions_df["sequence_idx"].isin(goal_seq_indices)].copy()

    def _softmax_group(scores: pd.Series) -> pd.Series:
        z = scores.to_numpy(dtype=float) / max(temperature, 1e-9)
        z = z - np.max(z)
        e = np.exp(z)
        w = e / np.sum(e)
        return pd.Series(w, index=scores.index)

    if len(goal_actions) > 0:
        goal_actions["xa_plus"] = goal_actions.groupby("sequence_idx")["score"].transform(
            _softmax_group
        )
        actions_df.loc[goal_actions.index, "xa_plus"] = goal_actions["xa_plus"]

        # Assign back to sequences wide-format columns
        for seq_idx, group in goal_actions.groupby("sequence_idx"):
            for action_num, weight in zip(
                group["action_num"].astype(int), group["xa_plus"].astype(float)
            ):
                df.loc[seq_idx, f"xa_plus_action{action_num}"] = weight

        credits_assigned = float(goal_actions["xa_plus"].sum())
    else:
        credits_assigned = 0.0

    # Create total column
    df["xa_plus_total"] = sum(df[f"xa_plus_action{i}"] for i in range(1, 6))

    # Summary stats
    logger.info(f"  Total xA+ credit assigned: {credits_assigned:.1f}")
    logger.info(f"  Expected (goals): {num_goals}")
    logger.info(f"  Ratio: {credits_assigned/max(num_goals, 1):.2f}")

    # Credit distribution by action position
    logger.info("  Credit by action position:")
    for i in range(1, 6):
        total = df[f"xa_plus_action{i}"].sum()
        if total > 0:
            pct = total / max(credits_assigned, 1) * 100
            logger.info(f"    Action {i}: {total:.1f} ({pct:.1f}%)")

    # Credit by action type
    logger.info("  Credit by action type:")
    for atype in actions_df["action_type"].unique():
        type_credit = actions_df[actions_df["action_type"] == atype]["xa_plus"].sum()
        if type_credit > 0:
            pct = type_credit / max(credits_assigned, 1) * 100
            logger.info(f"    {atype}: {type_credit:.1f} ({pct:.1f}%)")

    return df, actions_df, model
