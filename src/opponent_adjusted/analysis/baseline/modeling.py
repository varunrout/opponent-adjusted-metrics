"""Core fitting/scoring logic for the dumb baseline and v1 kitchen-sink logistic regression.

Kept as pure, testable functions (fit on train-only artifacts, transform any split with them)
so the orchestration script (`scripts/materialize_cxg_baseline_v1.py`) stays a thin BigQuery
read/write wrapper around this module -- same separation used by `bivariate/testing.py`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

# Narrow imports (not `statsmodels.api`) -- see bivariate/testing.py for why: the full api
# module fails to import under this environment's pandas/statsmodels version combination.
from statsmodels.discrete.discrete_model import Logit
from statsmodels.tools.tools import add_constant


@dataclass(frozen=True)
class FittedV1Model:
    """Train-only-fitted artifacts: imputer + scaler for continuous features, fixed
    one-hot category levels for the categorical feature, and the fitted Logit model.
    `transform` reapplies all of these unchanged to any split (val/test), never refit."""

    continuous_cols: tuple[str, ...]
    categorical_col: str | None
    categorical_levels: tuple[str, ...]  # levels present in train, drop_first already applied
    imputer: SimpleImputer
    scaler: StandardScaler
    design_columns: tuple[str, ...]  # column order the fitted model expects (post add_constant)
    result: object  # statsmodels LogitResults


def _design_matrix(
    df: pd.DataFrame,
    continuous_cols: tuple[str, ...],
    categorical_col: str | None,
    categorical_levels: tuple[str, ...] | None,
    imputer: SimpleImputer,
    scaler: StandardScaler,
) -> pd.DataFrame:
    X_cont_raw = df[list(continuous_cols)].to_numpy(dtype=float)
    X_cont = scaler.transform(imputer.transform(X_cont_raw))
    parts = [pd.DataFrame(X_cont, columns=continuous_cols, index=df.index)]

    if categorical_col is not None:
        dummies = pd.get_dummies(df[categorical_col].astype(str), prefix=categorical_col)
        dummy_cols = [f"{categorical_col}_{lvl}" for lvl in categorical_levels]
        for c in dummy_cols:
            if c not in dummies.columns:
                dummies[c] = 0.0
        parts.append(dummies[dummy_cols].astype(float))

    return pd.concat(parts, axis=1)


def fit_v1_model(
    df_train: pd.DataFrame,
    y_col: str,
    continuous_cols: tuple[str, ...],
    categorical_col: str | None = None,
) -> FittedV1Model:
    """Fit median-impute + StandardScaler (train-only) for continuous features, drop-first
    one-hot for the categorical feature (train's observed levels, minus the first
    alphabetically -- fixed at fit time), then a standard additive logistic regression."""
    X_cont_raw = df_train[list(continuous_cols)].to_numpy(dtype=float)
    imputer = SimpleImputer(strategy="median").fit(X_cont_raw)
    scaler = StandardScaler().fit(imputer.transform(X_cont_raw))

    categorical_levels: tuple[str, ...] = ()
    if categorical_col is not None:
        all_levels = sorted(df_train[categorical_col].astype(str).unique())
        categorical_levels = tuple(all_levels[1:])  # drop first level (reference category)

    X = _design_matrix(df_train, continuous_cols, categorical_col, categorical_levels, imputer, scaler)
    X = add_constant(X, has_constant="add")
    y = df_train[y_col].astype(int)

    model = Logit(y, X).fit(disp=0, maxiter=200)

    return FittedV1Model(
        continuous_cols=continuous_cols,
        categorical_col=categorical_col,
        categorical_levels=categorical_levels,
        imputer=imputer,
        scaler=scaler,
        design_columns=tuple(X.columns),
        result=model,
    )


def predict_v1(fitted: FittedV1Model, df: pd.DataFrame) -> np.ndarray:
    X = _design_matrix(
        df, fitted.continuous_cols, fitted.categorical_col, fitted.categorical_levels,
        fitted.imputer, fitted.scaler,
    )
    X = add_constant(X, has_constant="add")
    X = X.reindex(columns=fitted.design_columns, fill_value=0.0)
    return np.asarray(fitted.result.predict(X))


def coefficient_table(fitted: FittedV1Model) -> pd.DataFrame:
    r = fitted.result
    return pd.DataFrame(
        {
            "feature": r.params.index,
            "coefficient": r.params.values,
            "std_error": r.bse.values,
            "p_value": r.pvalues.values,
        }
    )


def dumb_baseline_prob(df_train: pd.DataFrame, y_col: str) -> float:
    return float(df_train[y_col].astype(int).mean())


def score_predictions(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    """log_loss/brier always computed; roc_auc is None when y_pred is (numerically) constant
    -- a constant predictor has no meaningful ROC-AUC, reporting 0.5 there would misleadingly
    imply "random but calibrated" rather than "no discriminative signal by construction"."""
    y_true_arr = np.asarray(y_true).astype(int)
    y_pred_arr = np.clip(np.asarray(y_pred, dtype=float), 1e-15, 1 - 1e-15)

    ll = float(log_loss(y_true_arr, y_pred_arr, labels=[0, 1]))
    brier = float(brier_score_loss(y_true_arr, y_pred_arr))

    auc = None
    if np.ptp(y_pred_arr) > 1e-12 and len(np.unique(y_true_arr)) == 2:
        auc = float(roc_auc_score(y_true_arr, y_pred_arr))

    return {"n": len(y_true_arr), "log_loss": ll, "brier_score": brier, "roc_auc": auc}


def calibration_table(y_true: pd.Series, y_pred: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Predicted-probability decile vs. actual goal rate. Rank-based binning (not a raw
    `pd.qcut` on the values) so this stays well-defined even for the dumb baseline's constant
    predictions, where every row ties on the raw value -- all 10 bins get the same predicted
    probability (correctly) but can show different actual rates from sampling, which is itself
    the honest illustration of "the floor has no discriminative power"."""
    df = pd.DataFrame({"y": np.asarray(y_true).astype(int), "p": np.asarray(y_pred, dtype=float)})
    ranks = df["p"].rank(method="first")
    df["decile"] = pd.qcut(ranks, n_bins, labels=list(range(1, n_bins + 1)))
    grouped = df.groupby("decile", observed=True).agg(n=("y", "size"), mean_predicted=("p", "mean"), actual_rate=("y", "mean"))
    return grouped.reset_index()
