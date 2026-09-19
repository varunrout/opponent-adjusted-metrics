"""v2 kitchen-sink-plus-interactions model: ridge-regularized (L2) logistic regression over
the enlarged 21-feature CxG+ pool, with explicit confirmed Tier 1 interaction terms.

Missing-categorical handling reuses v1's exact convention unchanged
(`baseline/modeling.py:_design_matrix`): each categorical column is cast to `str` BEFORE
computing dummy levels, so a missing value becomes the literal category `"nan"` and gets its
own explicit dummy indicator (unless it happens to sort first alphabetically, in which case
it IS the reference level) -- not silently absorbed into the reference category, not
row-dropped. No new missing-category mechanism invented here; same precedent, same reasoning.

Ridge fit uses `sklearn.linear_model.LogisticRegression(penalty="l2")` (`C` selected on the
validation split, not defaulted). The plain (unpenalized) comparison fit reuses
`statsmodels.discrete.discrete_model.Logit` (same library/convention as v1 and every prior
interaction test in this pipeline) so coefficient std_error/p_value stay directly comparable
to v1's `cxg_baseline_v1_coefficients`. Ridge coefficients do not get a classical p-value --
penalization deliberately introduces bias exactly where the standard asymptotic variance
formula stops applying cleanly; reported as `NULL` with the reason recorded, not
approximated, matching this project's "record why, don't fabricate" convention.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from statsmodels.discrete.discrete_model import Logit
from statsmodels.tools.tools import add_constant

DEEP_BLOCK_CLEARER = "deep_block_clearer"
ARCHETYPE_COLLAPSED_SUFFIX = "_is_deep_block_clearer"


@dataclass(frozen=True)
class V2ModelSpec:
    continuous_cols: tuple[str, ...]
    categorical_cols: tuple[str, ...]
    interactions: tuple[tuple[str, str], ...]
    log_gap: bool = False
    collapse_archetype: bool = False
    archetype_cols: tuple[str, ...] = ("nearest_defender_style_archetype", "second_nearest_defender_style_archetype")
    gap_col: str = "nearest_defender_gap"


def apply_transforms(df: pd.DataFrame, spec: V2ModelSpec) -> pd.DataFrame:
    """Column-level transforms only (log1p, archetype collapsing) -- returns a new frame with
    the same column names, ready for `_design_matrix`. Does not impute/scale/impute (that
    happens per-split inside `_design_matrix`, fit on train only)."""
    df = df.copy()
    if spec.log_gap and spec.gap_col in df.columns:
        df[spec.gap_col] = np.log1p(df[spec.gap_col])
    if spec.collapse_archetype:
        for col in spec.archetype_cols:
            if col in df.columns:
                df[col] = df[col].apply(lambda v: DEEP_BLOCK_CLEARER if v == DEEP_BLOCK_CLEARER else ("other" if pd.notna(v) else v))
    return df


@dataclass(frozen=True)
class FittedV2Model:
    spec: V2ModelSpec
    imputer: SimpleImputer
    scaler: StandardScaler
    categorical_levels: dict[str, tuple[str, ...]]
    design_columns: tuple[str, ...]
    kind: str  # "ridge" | "plain"
    model: object  # sklearn LogisticRegression | statsmodels LogitResults


def _to_missing_safe_str(series: pd.Series) -> pd.Series:
    """v1's exact missing-category convention: a missing value becomes the literal string
    "nan". Deliberately not `series.astype(str)` alone -- under this environment's
    Arrow-backed pandas string dtype, `.astype(str)` on a column with NaN/None leaves those
    entries as an actual `float('nan')` object instead of converting them to the string
    "nan" (confirmed by direct inspection before writing this), which would silently break
    the "explicit missing category, not a fabricated/dropped value" guarantee. This maps
    every value through Python's own `str()` (missing values first), so the result is always
    a plain string regardless of pandas' internal string backend."""
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


def _build_matrix(
    df: pd.DataFrame, spec: V2ModelSpec, imputer: SimpleImputer, scaler: StandardScaler,
    categorical_levels: dict[str, tuple[str, ...]],
) -> pd.DataFrame:
    X_cont_raw = df[list(spec.continuous_cols)].to_numpy(dtype=float)
    X_cont = scaler.transform(imputer.transform(X_cont_raw))
    cont_df = pd.DataFrame(X_cont, columns=spec.continuous_cols, index=df.index)

    cat_dfs = {}
    for col in spec.categorical_cols:
        cat_dfs[col] = _categorical_dummies(df[col], col, categorical_levels[col])

    parts = [cont_df] + list(cat_dfs.values())

    for a, b in spec.interactions:
        a_is_cat, b_is_cat = a in spec.categorical_cols, b in spec.categorical_cols
        if a_is_cat and not b_is_cat:
            inter = cat_dfs[a].mul(cont_df[b], axis=0)
            inter.columns = [f"{c}:{b}" for c in cat_dfs[a].columns]
        elif b_is_cat and not a_is_cat:
            inter = cat_dfs[b].mul(cont_df[a], axis=0)
            inter.columns = [f"{a}:{c}" for c in cat_dfs[b].columns]
        elif not a_is_cat and not b_is_cat:
            inter = pd.DataFrame({f"{a}:{b}": cont_df[a] * cont_df[b]}, index=df.index)
        else:
            raise NotImplementedError(f"categorical x categorical interaction not needed for this pool: {a} x {b}")
        parts.append(inter)

    return pd.concat(parts, axis=1)


def fit_v2_model(df_train: pd.DataFrame, y_col: str, spec: V2ModelSpec, kind: str, C: float = 1.0) -> FittedV2Model:
    df_train = apply_transforms(df_train, spec)
    X_cont_raw = df_train[list(spec.continuous_cols)].to_numpy(dtype=float)
    imputer = SimpleImputer(strategy="median").fit(X_cont_raw)
    scaler = StandardScaler().fit(imputer.transform(X_cont_raw))

    # Drop the alphabetically-first level as the reference category (matches v1's exact
    # convention, `baseline/modeling.py:fit_v1_model`) -- without this, the full set of
    # dummy columns for a k-level categorical sums to 1 everywhere, perfectly collinear with
    # the intercept, which is exactly what produced NaN std_errors in this module's own
    # tests before this fix was caught.
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

    return FittedV2Model(spec, imputer, scaler, categorical_levels, design_columns, kind, model)


def predict_v2(fitted: FittedV2Model, df: pd.DataFrame) -> np.ndarray:
    df = apply_transforms(df, fitted.spec)
    X = _build_matrix(df, fitted.spec, fitted.imputer, fitted.scaler, fitted.categorical_levels)
    if fitted.kind == "ridge":
        X = X.reindex(columns=[c for c in fitted.design_columns], fill_value=0.0)
        return fitted.model.predict_proba(X.to_numpy(dtype=float))[:, 1]
    X_const = add_constant(X, has_constant="add")
    X_const = X_const.reindex(columns=fitted.design_columns, fill_value=0.0)
    return np.asarray(fitted.model.predict(X_const))


def coefficient_table(fitted: FittedV2Model) -> pd.DataFrame:
    if fitted.kind == "plain":
        r = fitted.model
        return pd.DataFrame({"feature": r.params.index, "coefficient": r.params.values, "std_error": r.bse.values, "p_value": r.pvalues.values})
    # ridge: coefficients only, no classical std_error/p_value (see module docstring).
    coefs = list(fitted.model.coef_[0])
    features = ["const"] + list(fitted.design_columns)
    values = [float(fitted.model.intercept_[0])] + [float(c) for c in coefs]
    return pd.DataFrame({"feature": features, "coefficient": values, "std_error": [None] * len(features), "p_value": [None] * len(features)})
