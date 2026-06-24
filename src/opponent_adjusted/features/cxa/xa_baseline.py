"""xA Baseline - Logistic regression model for assist probability.

xA Baseline = P(is_assist | pass_features)

Uses logistic regression to predict the probability that a pass becomes an assist
based on its features (end location, pass type, etc.).

The model is calibrated so that Sum(xA_baseline) = Total assists.

Target: is_assist (bool) - the last pass before a goal
Input: pass_sequences.parquet
"""

from __future__ import annotations

import logging
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


# Features for the logistic regression model
BASELINE_FEATURES = [
    "end_x",  # Pass destination x (0-120, 120 = opponent goal)
    "end_y",  # Pass destination y (0-80)
    "is_cross",  # Cross into box
    "is_through_ball",  # Through ball behind defense
    "is_into_box",  # Pass ends in penalty area
    "is_progressive",  # Pass moves ball toward goal
]


class XABaselineModel:
    """xA Baseline: Logistic regression for P(is_assist | features).

    The model learns relative feature importance, then calibrates
    probabilities so they sum to the actual number of assists.
    """

    def __init__(self):
        self.model: Optional[LogisticRegression] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_weights: Optional[np.ndarray] = None
        self.calibration_factor: float = 1.0
        self.is_fitted = False

    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare feature matrix from DataFrame."""
        prep_df = df.copy()

        # Ensure boolean flags exist and are numeric
        for col in ["is_cross", "is_through_ball", "is_into_box", "is_progressive"]:
            if col not in prep_df.columns:
                prep_df[col] = False
            prep_df[col] = prep_df[col].fillna(False).astype(float)

        # Ensure numeric columns
        for col in ["end_x", "end_y"]:
            if col not in prep_df.columns:
                prep_df[col] = 0.0
            prep_df[col] = prep_df[col].fillna(0.0)

        # Build feature matrix
        X = prep_df[BASELINE_FEATURES].values

        return X

    def fit(self, passes_df: pd.DataFrame) -> "XABaselineModel":
        """Fit xA baseline model using logistic regression.

        Args:
            passes_df: DataFrame with all passes

        Returns:
            Self for chaining
        """
        logger.info("Fitting xA Baseline (logistic regression)...")

        df = passes_df.copy()

        # Create is_assist if not present
        if "is_assist" not in df.columns:
            df["is_assist"] = (df["is_key_pass"].fillna(False).astype(bool)) & (
                df["sequence_resulted_goal"].fillna(False).astype(bool)
            )

        num_passes = len(df)
        num_assists = df["is_assist"].sum()
        logger.info(f"  Total passes: {num_passes:,}")
        logger.info(f"  Assists: {num_assists}")

        # Prepare features
        X = self._prepare_features(df)
        y = df["is_assist"].astype(int).values

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Fit logistic regression (no class weighting - we'll calibrate instead)
        self.model = LogisticRegression(max_iter=1000, random_state=42)
        self.model.fit(X_scaled, y)

        # Store feature weights
        self.feature_weights = self.model.coef_[0]

        logger.info("  Feature weights (contribution to assist probability):")
        for i, feat in enumerate(BASELINE_FEATURES):
            logger.info(f"    {feat}: {self.feature_weights[i]:.3f}")

        # Calibrate: adjust probabilities so they sum to actual assists
        raw_probas = self.model.predict_proba(X_scaled)[:, 1]
        self.calibration_factor = num_assists / raw_probas.sum()
        logger.info(f"  Calibration factor: {self.calibration_factor:.4f}")

        self.is_fitted = True
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Predict P(is_assist) for each pass.

        Args:
            df: DataFrame with pass features

        Returns:
            Array of assist probabilities (calibrated)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X = self._prepare_features(df)
        X_scaled = self.scaler.transform(X)

        # Get raw probability and apply calibration
        raw_probas = self.model.predict_proba(X_scaled)[:, 1]
        calibrated_probas = raw_probas * self.calibration_factor

        # Clip to [0, 1]
        calibrated_probas = np.clip(calibrated_probas, 0, 1)

        return calibrated_probas


def compute_xa_baseline(
    passes_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, XABaselineModel]:
    """Compute xA Baseline for all passes using logistic regression.

    Args:
        passes_df: DataFrame from pass_sequences.parquet

    Returns:
        Tuple of (passes DataFrame with xa_baseline column, fitted model)
    """
    logger.info("Computing xA Baseline...")

    df = passes_df.copy()

    # Create is_assist if not present
    if "is_assist" not in df.columns:
        df["is_assist"] = (df["is_key_pass"].fillna(False).astype(bool)) & (
            df["sequence_resulted_goal"].fillna(False).astype(bool)
        )

    # Fit the model
    model = XABaselineModel()
    model.fit(df)

    # Predict probabilities for all passes
    df["xa_baseline"] = model.predict_proba(df)

    # Summary stats
    total_xa = df["xa_baseline"].sum()
    actual_assists = df["is_assist"].sum()

    logger.info(f"  Sum(xa_baseline): {total_xa:.1f}")
    logger.info(f"  Actual assists: {actual_assists}")
    logger.info(f"  Ratio: {total_xa/actual_assists:.2f}")
    logger.info(f"  Mean xa_baseline: {df['xa_baseline'].mean():.6f}")

    return df, model
