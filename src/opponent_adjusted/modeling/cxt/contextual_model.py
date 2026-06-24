"""
Contextual xT (CxT) Model

Predicts expected threat value of progressions (passes, carries, dribbles)
adjusted for opponent defensive strength and game context.

CxT = P(completion) * E[xT_delta | completion]
     + (1 - P(completion)) * E[xT_delta | incomplete]
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)


# Feature columns - must match actual column names in featured dataset
NUMERIC_FEATURES = [
    "start_xt",
    "xt_delta",
    "minute_normalized",
    "opponent_global_rating",
    "opponent_zone_rating",
    "opponent_global_block_rate",
    "opponent_zone_block_rate",
]

BINARY_FEATURES = [
    "under_pressure",
    "is_late_game",
    "is_first_half",
    "is_second_half",
    "is_extra_time",
    "is_very_late",
    "is_early_game",
    "start_is_central",
    "is_progressive",
    "is_into_final_third",
    "is_into_penalty_area",
    "moved_to_att_third",
    "moved_wide_to_central",
    "zone_changed",
    "pressure_flag",
    "is_pass",
    "is_carry",
    "is_dribble",
    "opponent_is_strong",
    "opponent_is_weak",
]

CATEGORICAL_FEATURES = [
    "action_type",
    "start_third",
    "macro_zone_start",
]


@dataclass
class CxTModel:
    """Container for trained CxT model components."""

    completion_model: Pipeline
    xt_gain_model: Pipeline
    completion_features: list[str]  # Features for completion model
    gain_features: list[str]  # Features for xT gain model (excludes xt_delta)
    numeric_features: list[str] = field(default_factory=list)
    binary_features: list[str] = field(default_factory=list)
    categorical_features: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    # Keep feature_columns for backward compatibility
    @property
    def feature_columns(self) -> list[str]:
        return self.gain_features

    def predict_completion_prob(self, X: pd.DataFrame) -> np.ndarray:
        """Predict completion probability."""
        available = [c for c in self.completion_features if c in X.columns]
        return self.completion_model.predict_proba(X[available])[:, 1]

    def predict_xt_gain(self, X: pd.DataFrame) -> np.ndarray:
        """Predict expected xT gain (for completed actions)."""
        available = [c for c in self.gain_features if c in X.columns]
        return self.xt_gain_model.predict(X[available])

    def predict_cxt(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict contextual xT value.

        CxT = P(complete) * E[xT_delta | complete]

        For incomplete actions, we assume xT_delta ≈ 0 (ball lost).
        """
        p_complete = self.predict_completion_prob(X)
        xt_if_complete = self.predict_xt_gain(X)

        # CxT is expected xT weighted by completion probability
        return p_complete * xt_if_complete

    def save(self, output_dir: Path) -> None:
        """Save model to disk."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.completion_model, output_dir / "completion_model.joblib")
        joblib.dump(self.xt_gain_model, output_dir / "xt_gain_model.joblib")

        config = {
            "completion_features": self.completion_features,
            "gain_features": self.gain_features,
            "numeric_features": self.numeric_features,
            "binary_features": self.binary_features,
            "categorical_features": self.categorical_features,
            "metadata": self.metadata,
        }
        with open(output_dir / "model_config.json", "w") as f:
            json.dump(config, f, indent=2, default=str)

        logger.info(f"Model saved to {output_dir}")

    @classmethod
    def load(cls, model_dir: Path) -> "CxTModel":
        """Load model from disk."""
        model_dir = Path(model_dir)

        completion_model = joblib.load(model_dir / "completion_model.joblib")
        xt_gain_model = joblib.load(model_dir / "xt_gain_model.joblib")

        with open(model_dir / "model_config.json") as f:
            config = json.load(f)

        return cls(
            completion_model=completion_model,
            xt_gain_model=xt_gain_model,
            completion_features=config.get(
                "completion_features", config.get("feature_columns", [])
            ),
            gain_features=config.get("gain_features", config.get("feature_columns", [])),
            numeric_features=config.get("numeric_features", []),
            binary_features=config.get("binary_features", []),
            categorical_features=config.get("categorical_features", []),
            metadata=config.get("metadata", {}),
        )


def build_preprocessor(
    numeric_features: list[str],
    binary_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    """Build sklearn preprocessing pipeline."""

    transformers = []

    if numeric_features:
        numeric_transformer = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("num", numeric_transformer, numeric_features))

    if binary_features:
        binary_transformer = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
            ]
        )
        transformers.append(("bin", binary_transformer, binary_features))

    if categorical_features:
        categorical_transformer = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        transformers.append(("cat", categorical_transformer, categorical_features))

    return ColumnTransformer(transformers, remainder="drop")


def train_cxt_model(
    df: pd.DataFrame,
    numeric_features: list[str] | None = None,
    binary_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
    n_splits: int = 5,
    random_state: int = 42,
) -> tuple[CxTModel, dict[str, Any]]:
    """
    Train CxT model with cross-validation.

    Args:
        df: Featured dataframe with success column
        numeric_features: Numeric feature columns
        binary_features: Binary feature columns
        categorical_features: Categorical feature columns
        n_splits: Number of CV folds
        random_state: Random state for reproducibility

    Returns:
        (model, metrics) tuple
    """
    logger.info("=" * 60)
    logger.info("TRAINING CxT MODEL")
    logger.info("=" * 60)

    # Default features
    numeric_features = numeric_features or [c for c in NUMERIC_FEATURES if c in df.columns]
    binary_features = binary_features or [c for c in BINARY_FEATURES if c in df.columns]
    categorical_features = categorical_features or [
        c for c in CATEGORICAL_FEATURES if c in df.columns
    ]

    feature_columns = numeric_features + binary_features + categorical_features
    logger.info(f"Features: {len(feature_columns)} total")
    logger.info(f"  Numeric: {len(numeric_features)}")
    logger.info(f"  Binary: {len(binary_features)}")
    logger.info(f"  Categorical: {len(categorical_features)}")

    # Prepare data
    df = df.copy()

    # Ensure success column - use action_success if available
    if "success" not in df.columns:
        if "action_success" in df.columns:
            df["success"] = df["action_success"].astype(int)
        elif "action_outcome" in df.columns:
            df["success"] = (df["action_outcome"] == "Complete").astype(int)
        else:
            raise ValueError("No success column found")

    # For xT gain model, only use completed actions
    df_complete = df[df["success"] == 1].copy()

    # Groups for CV (by match)
    if "match_id" in df.columns:
        groups = df["match_id"].values
        groups_complete = df_complete["match_id"].values
    else:
        groups = np.arange(len(df))
        groups_complete = np.arange(len(df_complete))

    logger.info(f"Training data: {len(df):,} rows")
    logger.info(f"Completed actions: {len(df_complete):,} rows")
    logger.info(f"Success rate: {df['success'].mean():.1%}")

    # ----- COMPLETION MODEL -----
    logger.info("\nTraining completion model...")

    # IMPORTANT: Exclude xt_delta from completion model to prevent data leakage
    # xt_delta is only known after action completes, so it would leak completion info
    completion_numeric_features = [c for c in numeric_features if c != "xt_delta"]
    completion_features = [c for c in feature_columns if c != "xt_delta"]

    # Build preprocessor for completion model (without xt_delta)
    completion_preprocessor = build_preprocessor(
        completion_numeric_features, binary_features, categorical_features
    )

    completion_pipeline = Pipeline(
        [
            ("preprocessor", completion_preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=random_state,
                    class_weight="balanced",
                    C=1.0,
                ),
            ),
        ]
    )

    # CV for completion model
    cv = GroupKFold(n_splits=n_splits)
    completion_metrics = _cv_classification(
        completion_pipeline, df, completion_features, "success", groups, cv
    )

    # Fit final model
    completion_pipeline.fit(df[completion_features], df["success"])

    logger.info(
        f"  Completion AUC: {completion_metrics['auc_mean']:.3f} ± {completion_metrics['auc_std']:.3f}"
    )
    logger.info(f"  Completion Brier: {completion_metrics['brier_mean']:.4f}")

    # ----- XT GAIN MODEL -----
    logger.info("\nTraining xT gain model...")

    # Remove xt_delta from features for gain model (it's the target)
    gain_numeric_features = [c for c in numeric_features if c != "xt_delta"]
    gain_features = [c for c in feature_columns if c != "xt_delta"]

    # Build preprocessor for gain model with adjusted features
    gain_preprocessor = build_preprocessor(
        gain_numeric_features, binary_features, categorical_features
    )

    xt_gain_pipeline = Pipeline(
        [
            ("preprocessor", gain_preprocessor),
            ("regressor", Ridge(alpha=1.0, random_state=random_state)),
        ]
    )

    # CV for gain model
    gain_metrics = _cv_regression(
        xt_gain_pipeline, df_complete, gain_features, "xt_delta", groups_complete, cv
    )

    # Fit final model
    xt_gain_pipeline.fit(df_complete[gain_features], df_complete["xt_delta"])

    logger.info(f"  xT Gain R²: {gain_metrics['r2_mean']:.3f} ± {gain_metrics['r2_std']:.3f}")
    logger.info(f"  xT Gain MAE: {gain_metrics['mae_mean']:.4f}")

    # ----- BUILD MODEL OBJECT -----
    metadata = {
        "train_date": datetime.now(timezone.utc).isoformat(),
        "n_samples": len(df),
        "n_complete": len(df_complete),
        "success_rate": float(df["success"].mean()),
        "completion_metrics": completion_metrics,
        "gain_metrics": gain_metrics,
    }

    model = CxTModel(
        completion_model=completion_pipeline,
        xt_gain_model=xt_gain_pipeline,
        completion_features=completion_features,  # Excludes xt_delta to prevent leakage
        gain_features=gain_features,  # Excludes xt_delta (it's the target)
        numeric_features=numeric_features,
        binary_features=binary_features,
        categorical_features=categorical_features,
        metadata=metadata,
    )

    # Combined metrics
    all_metrics = {
        "completion": completion_metrics,
        "xt_gain": gain_metrics,
    }

    return model, all_metrics


def _cv_classification(
    pipeline: Pipeline,
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    groups: np.ndarray,
    cv: GroupKFold,
) -> dict[str, float]:
    """Run cross-validation for classification."""

    aucs, briers, log_losses = [], [], []

    for train_idx, val_idx in cv.split(df, groups=groups):
        X_train = df.iloc[train_idx][feature_cols]
        y_train = df.iloc[train_idx][target_col]
        X_val = df.iloc[val_idx][feature_cols]
        y_val = df.iloc[val_idx][target_col]

        pipeline.fit(X_train, y_train)
        y_prob = pipeline.predict_proba(X_val)[:, 1]

        aucs.append(roc_auc_score(y_val, y_prob))
        briers.append(brier_score_loss(y_val, y_prob))
        log_losses.append(log_loss(y_val, y_prob))

    return {
        "auc_mean": np.mean(aucs),
        "auc_std": np.std(aucs),
        "brier_mean": np.mean(briers),
        "brier_std": np.std(briers),
        "logloss_mean": np.mean(log_losses),
        "logloss_std": np.std(log_losses),
    }


def _cv_regression(
    pipeline: Pipeline,
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    groups: np.ndarray,
    cv: GroupKFold,
) -> dict[str, float]:
    """Run cross-validation for regression."""

    r2s, maes, rmses = [], [], []

    for train_idx, val_idx in cv.split(df, groups=groups):
        X_train = df.iloc[train_idx][feature_cols]
        y_train = df.iloc[train_idx][target_col]
        X_val = df.iloc[val_idx][feature_cols]
        y_val = df.iloc[val_idx][target_col]

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_val)

        r2s.append(r2_score(y_val, y_pred))
        maes.append(mean_absolute_error(y_val, y_pred))
        rmses.append(np.sqrt(mean_squared_error(y_val, y_pred)))

    return {
        "r2_mean": np.mean(r2s),
        "r2_std": np.std(r2s),
        "mae_mean": np.mean(maes),
        "mae_std": np.std(maes),
        "rmse_mean": np.mean(rmses),
        "rmse_std": np.std(rmses),
    }


def evaluate_cxt_model(
    model: CxTModel,
    df: pd.DataFrame,
) -> dict[str, Any]:
    """
    Evaluate CxT model on a dataset.

    Returns:
        Dictionary with evaluation metrics
    """
    logger.info("Evaluating CxT model...")

    df = df.copy()

    # Ensure success column - use action_success if available
    if "success" not in df.columns:
        if "action_success" in df.columns:
            df["success"] = df["action_success"].astype(int)
        elif "action_outcome" in df.columns:
            df["success"] = (df["action_outcome"] == "Complete").astype(int)
        else:
            raise ValueError("No success column found")

    # Predictions
    p_complete = model.predict_completion_prob(df)
    model.predict_xt_gain(df)
    cxt = model.predict_cxt(df)

    # Completion metrics (full dataset)
    completion_auc = roc_auc_score(df["success"], p_complete)
    completion_brier = brier_score_loss(df["success"], p_complete)

    # xT gain metrics (completed only)
    df_complete = df[df["success"] == 1]
    pred_xt_complete = model.predict_xt_gain(df_complete)

    gain_r2 = r2_score(df_complete["xt_delta"], pred_xt_complete)
    gain_mae = mean_absolute_error(df_complete["xt_delta"], pred_xt_complete)

    # CxT correlation with actual xT delta
    # For completed: actual = xt_delta, for incomplete: actual ≈ 0
    actual_xt = np.where(df["success"] == 1, df["xt_delta"], 0)
    cxt_corr = np.corrcoef(cxt, actual_xt)[0, 1]

    metrics = {
        "completion_auc": completion_auc,
        "completion_brier": completion_brier,
        "xt_gain_r2": gain_r2,
        "xt_gain_mae": gain_mae,
        "cxt_actual_corr": cxt_corr,
        "cxt_mean": float(cxt.mean()),
        "cxt_std": float(cxt.std()),
    }

    logger.info(f"  Completion AUC: {completion_auc:.3f}")
    logger.info(f"  xT Gain R²: {gain_r2:.3f}")
    logger.info(f"  CxT-Actual Correlation: {cxt_corr:.3f}")

    return metrics
