"""v3 model: track-generic ridge/plain logistic regression, reused unchanged for BOTH CxG
event-wide and CxG+ (unlike v2model/modeling.py, which is CxG+-only and hardcodes the
archetype-collapse/gap-log transform decisions). Not a fork of v2model -- v2 stays frozen and
untouched; this is new, general-purpose code for a track-agnostic pool.

Missing-categorical handling reuses v1/v2's exact convention unchanged: each categorical
column is cast to a string via `_to_missing_safe_str` BEFORE computing dummy levels, so a
missing value becomes the literal category `"nan"` with its own explicit dummy indicator
(unless it sorts first alphabetically, in which case it IS the reference level) -- never
silently absorbed into the reference category, never row-dropped.

Missing-CONTINUOUS handling (new in v3, needed for `cross_match_defensive_rate`'s cold-start
nulls, which are a real "no prior match observed" state, not a data-quality gap): an explicit
boolean indicator column `<col>_was_missing` is added to the continuous design matrix, and
the underlying value is then median-imputed (fit train-only, same as every other continuous
column) so the row is never dropped -- the model can learn a distinct effect for
"cold-start, imputed" versus "genuinely near-median value observed," rather than collapsing
the two into an indistinguishable state.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from statsmodels.discrete.discrete_model import Logit
from statsmodels.tools.tools import add_constant


@dataclass(frozen=True)
class V3ModelSpec:
    continuous_cols: tuple[str, ...]
    categorical_cols: tuple[str, ...] = ()
    interactions: tuple[tuple[str, str], ...] = ()
    log1p_cols: tuple[str, ...] = ()
    missing_indicator_cols: tuple[str, ...] = ()


def apply_transforms(df: pd.DataFrame, spec: V3ModelSpec) -> pd.DataFrame:
    """Column-level transforms only (log1p) -- returns a new frame with the same column
    names, ready for `_design_matrix`. Imputation/scaling/missing-indicators happen per-split
    inside `_build_matrix` (fit on train only)."""
    df = df.copy()
    for col in spec.log1p_cols:
        if col in df.columns:
            df[col] = np.log1p(df[col])
    return df


@dataclass(frozen=True)
class FittedV3Model:
    spec: V3ModelSpec
    imputer: SimpleImputer
    scaler: StandardScaler
    categorical_levels: dict[str, tuple[str, ...]]
    design_columns: tuple[str, ...]
    kind: str  # "ridge" | "plain"
    model: object


def _to_missing_safe_str(series: pd.Series) -> pd.Series:
    """Same convention as `baseline/modeling.py`/`v2model/modeling.py`: maps every value
    through Python's own `str()` (missing values first, as the literal string "nan"), so the
    result is always a plain string regardless of pandas' internal string backend -- Arrow-
    backed pandas' `.astype(str)` alone leaves NaN as a float object instead of "nan"."""
    return series.map(lambda v: "nan" if pd.isna(v) else str(v))


def _categorical_dummies(series: pd.Series, name: str, levels: tuple[str, ...] | None) -> pd.DataFrame:
    str_series = _to_missing_safe_str(series)
    dummies = pd.get_dummies(str_series, prefix=name).astype(float)
    if levels is None:
        return dummies
    cols = [f"{name}_{lvl}" for lvl in levels]
    for c in cols:
        if c not in dummies.columns:
            dummies[c] = 0.0
    return dummies[cols]


def _continuous_design(
    df: pd.DataFrame, spec: V3ModelSpec, imputer: SimpleImputer, scaler: StandardScaler
) -> pd.DataFrame:
    missing_flags = pd.DataFrame(
        {f"{c}_was_missing": df[c].isna().astype(float) for c in spec.missing_indicator_cols}, index=df.index
    )
    X_cont_raw = df[list(spec.continuous_cols)].to_numpy(dtype=float)
    X_cont = scaler.transform(imputer.transform(X_cont_raw))
    cont_df = pd.DataFrame(X_cont, columns=spec.continuous_cols, index=df.index)
    return pd.concat([cont_df, missing_flags], axis=1) if len(missing_flags.columns) else cont_df


def _build_matrix(
    df: pd.DataFrame, spec: V3ModelSpec, imputer: SimpleImputer, scaler: StandardScaler,
    categorical_levels: dict[str, tuple[str, ...]],
) -> pd.DataFrame:
    cont_df = _continuous_design(df, spec, imputer, scaler)

    cat_dfs = {col: _categorical_dummies(df[col], col, categorical_levels[col]) for col in spec.categorical_cols}
    parts = [cont_df] + list(cat_dfs.values())

    for a, b in spec.interactions:
        a_is_cat, b_is_cat = a in spec.categorical_cols, b in spec.categorical_cols
        if a_is_cat and b_is_cat:
            inter = pd.DataFrame(
                {f"{ac}:{bc}": cat_dfs[a][ac] * cat_dfs[b][bc] for ac in cat_dfs[a].columns for bc in cat_dfs[b].columns},
                index=df.index,
            )
        elif a_is_cat and not b_is_cat:
            inter = cat_dfs[a].mul(cont_df[b], axis=0)
            inter.columns = [f"{c}:{b}" for c in cat_dfs[a].columns]
        elif b_is_cat and not a_is_cat:
            inter = cat_dfs[b].mul(cont_df[a], axis=0)
            inter.columns = [f"{a}:{c}" for c in cat_dfs[b].columns]
        else:
            inter = pd.DataFrame({f"{a}:{b}": cont_df[a] * cont_df[b]}, index=df.index)
        parts.append(inter)

    return pd.concat(parts, axis=1)


def fit_v3_model(df_train: pd.DataFrame, y_col: str, spec: V3ModelSpec, kind: str, C: float = 1.0) -> FittedV3Model:
    df_train = apply_transforms(df_train, spec)
    X_cont_raw = df_train[list(spec.continuous_cols)].to_numpy(dtype=float)
    imputer = SimpleImputer(strategy="median").fit(X_cont_raw)
    scaler = StandardScaler().fit(imputer.transform(X_cont_raw))

    categorical_levels = {
        col: tuple(sorted(_to_missing_safe_str(df_train[col]).unique()))[1:]
        for col in spec.categorical_cols
    }

    X = _build_matrix(df_train, spec, imputer, scaler, categorical_levels)
    y = df_train[y_col].astype(int)

    if kind == "ridge":
        model = LogisticRegression(penalty="l2", C=C, solver="lbfgs", max_iter=2000)
        model.fit(X.to_numpy(dtype=float), y.to_numpy())
        design_columns = tuple(X.columns)
    elif kind == "plain":
        X_const = add_constant(X, has_constant="add")
        model = Logit(y, X_const).fit(disp=0, maxiter=200)
        design_columns = tuple(X_const.columns)
    else:
        raise ValueError(f"unknown kind: {kind!r}")

    return FittedV3Model(spec, imputer, scaler, categorical_levels, design_columns, kind, model)


def predict_v3(fitted: FittedV3Model, df: pd.DataFrame) -> np.ndarray:
    df = apply_transforms(df, fitted.spec)
    X = _build_matrix(df, fitted.spec, fitted.imputer, fitted.scaler, fitted.categorical_levels)
    if fitted.kind == "ridge":
        X = X.reindex(columns=list(fitted.design_columns), fill_value=0.0)
        return fitted.model.predict_proba(X.to_numpy(dtype=float))[:, 1]
    X_const = add_constant(X, has_constant="add")
    X_const = X_const.reindex(columns=fitted.design_columns, fill_value=0.0)
    return np.asarray(fitted.model.predict(X_const))


def coefficient_table(fitted: FittedV3Model) -> pd.DataFrame:
    if fitted.kind == "plain":
        r = fitted.model
        return pd.DataFrame({"feature": r.params.index, "coefficient": r.params.values, "std_error": r.bse.values, "p_value": r.pvalues.values})
    coefs = list(fitted.model.coef_[0])
    features = ["const"] + list(fitted.design_columns)
    values = [float(fitted.model.intercept_[0])] + [float(c) for c in coefs]
    return pd.DataFrame({"feature": features, "coefficient": values, "std_error": [None] * len(features), "p_value": [None] * len(features)})
