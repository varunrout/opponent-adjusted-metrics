"""xA+ (Passes) - Softmax credit distribution across pass sequences.

For sequences ending in a goal (is_goal=True), distribute 1.0 credit
across all passes in the sequence using softmax weighting based on pass features.

Pass₁ → Pass₂ → Pass₃ (Assist) → Shot → GOAL
  ↓       ↓       ↓
 0.15    0.25    0.60  (softmax weights, sum=1.0)

Training: sequences.parquet (one row per sequence, wide format)
         - Convert to long format for training (position-agnostic)
Target: is_assist (last pass in goal-scoring sequence)
Features: end_x, end_xt, xt_delta, is_through_ball, etc. (same for all passes)
"""

from __future__ import annotations

import logging
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


# Features used to score each pass (position-agnostic)
PASS_FEATURES = [
    "end_x",  # Pass destination x
    "end_y",  # Pass destination y
    "end_xt",  # Positional threat at destination
    "xt_delta",  # Change in threat from pass
    "is_cross",  # Cross into box
    "is_through_ball",  # Through ball behind defense
    "is_into_box",  # Pass ends in penalty area
    "is_progressive",  # Pass moves ball toward goal
]

# Max passes in a sequence
MAX_PASSES = 3


class XAPlusPassModel:
    """xA+ Pass Model: Softmax credit distribution across pass sequences.

    Trained on passes (long format, position-agnostic) to learn what pass features
    predict an assist. Then uses learned weights to distribute credit.
    """

    def __init__(self, temperature: float = 1.0):
        """
        Args:
            temperature: Softmax temperature. Lower = more peaked distribution.
        """
        self.scorer = None
        self.scaler = StandardScaler()
        self.temperature = temperature
        self.feature_weights: Optional[np.ndarray] = None
        self.is_fitted = False

    def _sequences_to_passes(self, seq_df: pd.DataFrame) -> pd.DataFrame:
        """Convert wide sequence format to long pass format for training.

        Each pass becomes a row with the same features, regardless of position.
        """

        def _get(obj, name: str, default=np.nan):
            return getattr(obj, name) if hasattr(obj, name) else default

        rows = []

        # itertuples is substantially faster than iterrows for wide frames
        for seq in seq_df.itertuples(index=False):
            num_passes = int(_get(seq, "num_passes_in_sequence", 0))
            is_goal = bool(_get(seq, "is_goal", False))
            sequence_id = _get(seq, "sequence_id")
            shot_id = _get(seq, "shot_id")

            for pass_num in range(1, MAX_PASSES + 1):
                pass_id = _get(seq, f"pass{pass_num}_id")

                # Skip if pass doesn't exist
                if pd.isna(pass_id):
                    continue

                # Is this the assist? (last pass in a goal-scoring sequence)
                is_assist = (pass_num == num_passes) and is_goal

                row = {
                    "sequence_id": sequence_id,
                    "shot_id": shot_id,
                    "pass_num": pass_num,
                    "pass_id": pass_id,
                    "is_assist": is_assist,
                    "is_goal": is_goal,
                }

                # Add features (position-agnostic names)
                for feat in PASS_FEATURES:
                    row[feat] = _get(seq, f"pass{pass_num}_{feat}", np.nan)

                # Add player info
                row["player_id"] = _get(seq, f"pass{pass_num}_player_id")
                row["player_name"] = _get(seq, f"pass{pass_num}_player_name")

                rows.append(row)

        return pd.DataFrame(rows)

    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare feature matrix from passes DataFrame."""
        X = df[PASS_FEATURES].fillna(0).values
        return X

    def fit(self, sequences_df: pd.DataFrame) -> "XAPlusPassModel":
        """Fit the pass scoring model on sequences.

        Converts sequences to long format, then trains logistic regression
        to learn which pass features predict being an assist.

        Args:
            sequences_df: Sequences with pass1_*, pass2_*, pass3_* features

        Returns:
            Self for chaining
        """
        logger.info("Fitting xA+ Pass Model (softmax weighting)...")

        # Convert to long format (position-agnostic)
        passes_df = self._sequences_to_passes(sequences_df)
        self.passes_df_ = passes_df  # Store for later use

        num_passes = len(passes_df)
        num_assists = passes_df["is_assist"].sum()
        logger.info(f"  Total sequences: {len(sequences_df):,}")
        logger.info(f"  Total passes (long format): {num_passes:,}")
        logger.info(f"  Assist passes: {num_assists}")

        # Build features
        X = self._prepare_features(passes_df)
        y = passes_df["is_assist"].astype(int).values

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Train logistic regression to learn feature importance
        self.scorer = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
        self.scorer.fit(X_scaled, y)

        # Store feature weights for interpretation
        self.feature_weights = self.scorer.coef_[0]

        logger.info("  Feature weights (contribution to assist probability):")
        for i, feat in enumerate(PASS_FEATURES):
            logger.info(f"    {feat}: {self.feature_weights[i]:.3f}")

        self.is_fitted = True
        return self

    def score_pass(self, pass_features: dict) -> float:
        """Score a single pass based on its features.

        Returns probability of being an assist.
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        # Build feature vector
        features = []
        for feat in PASS_FEATURES:
            val = pass_features.get(feat, 0)
            features.append(0.0 if pd.isna(val) else float(val))

        X = np.array(features).reshape(1, -1)
        X_scaled = self.scaler.transform(X)

        # Return probability of assist
        return self.scorer.predict_proba(X_scaled)[0, 1]

    def distribute_credit(self, seq_row: pd.Series) -> dict:
        """Distribute 1.0 credit across passes in a sequence using softmax.

        Args:
            seq_row: Single sequence row (wide format)

        Returns:
            Dict mapping pass_num -> credit
        """
        # Get scores for each pass that exists
        scores = {}
        for pass_num in range(1, MAX_PASSES + 1):
            pass_id_col = f"pass{pass_num}_id"

            # Skip if pass doesn't exist
            if pass_id_col not in seq_row.index or pd.isna(seq_row[pass_id_col]):
                continue

            # Build features dict for this pass
            pass_features = {}
            for feat in PASS_FEATURES:
                col = f"pass{pass_num}_{feat}"
                pass_features[feat] = seq_row.get(col, 0)

            scores[pass_num] = self.score_pass(pass_features)

        if not scores:
            return {}

        # Apply softmax
        pass_nums = list(scores.keys())
        raw_scores = np.array([scores[p] for p in pass_nums])

        # Softmax with temperature (use log-odds for better spread)
        log_odds = np.log(raw_scores / (1 - raw_scores + 1e-10))
        exp_scores = np.exp((log_odds - log_odds.max()) / self.temperature)
        weights = exp_scores / exp_scores.sum()

        return {pass_nums[i]: weights[i] for i in range(len(pass_nums))}


def compute_xa_plus_passes(
    sequences_df: pd.DataFrame,
    temperature: float = 1.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, XAPlusPassModel]:
    """Compute xA+ for passes using softmax credit distribution.

    Trains on sequences.parquet (wide format), then distributes credit
    only for goal-scoring sequences.

    Args:
        sequences_df: DataFrame from sequences.parquet (wide format)
        temperature: Softmax temperature

    Returns:
        Tuple of:
        - sequences DataFrame with xa_plus_pass1, xa_plus_pass2, xa_plus_pass3 columns
        - passes DataFrame (long format) with xa_plus column
        - fitted model
    """
    logger.info("Computing xA+ (Passes)...")

    df = sequences_df.copy()

    # Fit the scoring model
    model = XAPlusPassModel(temperature=temperature)
    model.fit(df)

    # Initialize xA+ columns
    for pass_num in range(1, MAX_PASSES + 1):
        df[f"xa_plus_pass{pass_num}"] = 0.0
    df["xa_plus_total"] = 0.0

    # Get goal-scoring sequences
    goal_sequences = df[df["is_goal"].fillna(False).astype(bool)]
    num_goals = len(goal_sequences)
    logger.info(f"  Goal-scoring sequences: {num_goals}")

    # Distribute credit for each goal sequence
    credits_assigned = 0

    for idx, row in goal_sequences.iterrows():
        credits = model.distribute_credit(row)

        total = 0.0
        for pass_num, credit in credits.items():
            df.loc[idx, f"xa_plus_pass{pass_num}"] = credit
            total += credit

        df.loc[idx, "xa_plus_total"] = total
        credits_assigned += total

    # Summary stats
    logger.info(f"  Total xA+ credit assigned: {credits_assigned:.1f}")
    logger.info(f"  Expected (goals): {num_goals}")
    logger.info(f"  Ratio: {credits_assigned/max(num_goals, 1):.2f}")

    # Credit by pass position
    logger.info("  Credit by pass position:")
    for pass_num in range(1, MAX_PASSES + 1):
        total_credit = df[f"xa_plus_pass{pass_num}"].sum()
        pct = total_credit / max(credits_assigned, 1) * 100
        logger.info(f"    Pass {pass_num}: {total_credit:.1f} ({pct:.1f}%)")

    # Also create long-format passes DataFrame with credits
    passes_long = _create_passes_long_format(df)

    # Calculate assist credit
    assist_credit = passes_long[passes_long["is_assist"].fillna(False).astype(bool)][
        "xa_plus"
    ].sum()
    logger.info(
        f"  Credit to assists: {assist_credit:.1f} ({assist_credit/max(credits_assigned,1)*100:.1f}%)"
    )

    return df, passes_long, model


def _create_passes_long_format(sequences_df: pd.DataFrame) -> pd.DataFrame:
    """Convert wide sequence format to long pass format.

    Returns DataFrame with one row per pass, including xa_plus credit.
    """

    def _get(obj, name: str, default=np.nan):
        return getattr(obj, name) if hasattr(obj, name) else default

    rows = []

    for seq in sequences_df.itertuples(index=False):
        num_passes = int(_get(seq, "num_passes_in_sequence", 0))
        is_goal = bool(_get(seq, "is_goal", False))
        sequence_id = _get(seq, "sequence_id")
        shot_id = _get(seq, "shot_id")

        for pass_num in range(1, MAX_PASSES + 1):
            pass_id = _get(seq, f"pass{pass_num}_id")

            if pd.isna(pass_id):
                continue

            row = {
                "sequence_id": sequence_id,
                "shot_id": shot_id,
                "pass_num": pass_num,
                "pass_id": pass_id,
                "player_id": _get(seq, f"pass{pass_num}_player_id"),
                "player_name": _get(seq, f"pass{pass_num}_player_name"),
                "end_x": _get(seq, f"pass{pass_num}_end_x"),
                "end_y": _get(seq, f"pass{pass_num}_end_y"),
                "end_xt": _get(seq, f"pass{pass_num}_end_xt"),
                "xt_delta": _get(seq, f"pass{pass_num}_xt_delta"),
                "is_cross": _get(seq, f"pass{pass_num}_is_cross"),
                "is_through_ball": _get(seq, f"pass{pass_num}_is_through_ball"),
                "is_into_box": _get(seq, f"pass{pass_num}_is_into_box"),
                "is_progressive": _get(seq, f"pass{pass_num}_is_progressive"),
                "is_goal": is_goal,
                "is_assist": (pass_num == num_passes) and is_goal,
                "xa_plus": _get(seq, f"xa_plus_pass{pass_num}", 0.0),
            }
            rows.append(row)

    return pd.DataFrame(rows)
