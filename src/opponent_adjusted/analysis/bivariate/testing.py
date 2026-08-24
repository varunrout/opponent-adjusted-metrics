"""Additive-vs-interaction logistic regression testing, shared by every tier.

Method: fit (a) `y ~ 1 + a + b` and (b) `y ~ 1 + a + b + a:b` via `statsmodels.Logit`,
likelihood-ratio test comparing the two (chi-square on the degrees-of-freedom difference,
which is 1 for two numeric features and k for a k-level categorical interaction term).

A tested pair always gets a result -- fit failures (perfect separation, non-convergence) or
insufficient post-listwise-deletion sample size are recorded with `fit_status` set
accordingly and all statistics `None`, never silently omitted (matches this project's
standing "recorded negative, not deletion" principle).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

# Narrow imports (not `statsmodels.api`) -- the full api module pulls in
# `statsmodels.graphics.tsaplots`, which fails to import under this environment's installed
# pandas/statsmodels version combination (an unrelated, pre-existing compatibility issue in
# a submodule this code never uses). Logit + add_constant alone import cleanly.
from statsmodels.discrete.discrete_model import Logit
from statsmodels.tools.tools import add_constant

MIN_N = 50


@dataclass(frozen=True)
class InteractionResult:
    n_train: int
    interaction_coef: float | None
    interaction_se: float | None
    interaction_p_raw: float | None
    lr_stat: float | None
    main_effect_a_coef: float | None
    main_effect_b_coef: float | None
    fit_status: str  # ok | insufficient_data | fit_failed


def _design(series: pd.Series, name: str, is_categorical: bool) -> pd.DataFrame:
    if is_categorical:
        dummies = pd.get_dummies(series, prefix=name, drop_first=True)
        return dummies.astype(float)
    return series.to_frame(name).astype(float)


def fit_interaction(
    df: pd.DataFrame,
    y_col: str,
    a_col: str,
    b_col: str,
    a_categorical: bool = False,
    b_categorical: bool = False,
) -> InteractionResult:
    sub = df[[y_col, a_col, b_col]].dropna().copy()
    n = len(sub)
    if n < MIN_N or sub[y_col].nunique() < 2:
        return InteractionResult(n, None, None, None, None, None, None, "insufficient_data")

    y = sub[y_col].astype(int)
    A = _design(sub[a_col], "a", a_categorical)
    B = _design(sub[b_col], "b", b_categorical)
    if A.shape[1] == 0 or B.shape[1] == 0 or A.nunique().min() < 2 or B.nunique().min() < 2:
        return InteractionResult(n, None, None, None, None, None, None, "insufficient_data")

    interaction = {f"{ac}:{bc}": A[ac] * B[bc] for ac in A.columns for bc in B.columns}
    inter_df = pd.DataFrame(interaction)

    X_add = add_constant(pd.concat([A, B], axis=1), has_constant="add")
    X_int = add_constant(pd.concat([A, B, inter_df], axis=1), has_constant="add")

    try:
        m_add = Logit(y, X_add).fit(disp=0, maxiter=100)
        m_int = Logit(y, X_int).fit(disp=0, maxiter=100)
    except Exception:
        return InteractionResult(n, None, None, None, None, None, None, "fit_failed")

    if not (np.isfinite(m_add.llf) and np.isfinite(m_int.llf)):
        return InteractionResult(n, None, None, None, None, None, None, "fit_failed")

    lr_stat = float(2 * (m_int.llf - m_add.llf))
    dof = X_int.shape[1] - X_add.shape[1]
    if lr_stat < 0 or dof <= 0:
        return InteractionResult(n, None, None, None, None, None, None, "fit_failed")
    p_raw = float(scipy_stats.chi2.sf(lr_stat, dof))

    if len(interaction) == 1:
        term = next(iter(interaction))
        interaction_coef = float(m_int.params[term])
        interaction_se = float(m_int.bse[term])
    else:
        interaction_coef = None
        interaction_se = None

    main_a = float(m_add.params[A.columns[0]]) if A.shape[1] == 1 else None
    main_b = float(m_add.params[B.columns[0]]) if B.shape[1] == 1 else None

    return InteractionResult(n, interaction_coef, interaction_se, p_raw, lr_stat, main_a, main_b, "ok")


def fit_categorical_interaction_saturated(
    df: pd.DataFrame, y_col: str, a_col: str, b_col: str
) -> InteractionResult:
    """Fallback for a categorical x categorical pair whose standard interaction-model MLE
    fit fails (`fit_interaction` returns `fit_status="fit_failed"`) because of sparse or
    structurally-empty cross-tab cells -- e.g. `nearest_defender_role x
    second_nearest_defender_role` has a literal zero-count GK x GK cell (a team has one
    goalkeeper, so it can never be both a shot's nearest and second-nearest defender), which
    makes the interaction design matrix rank-deficient; other cells with a handful of shots
    and zero goals cause complete/quasi-complete separation (observed as `overflow in exp` /
    `divide by zero in log` from statsmodels' unregularized MLE).

    Compares the additive model (A+B, no interaction -- identical in spirit to
    `fit_interaction`'s own `m_add`, and typically still fits fine: far fewer parameters,
    no column that is guaranteed all-zero) against the SATURATED cell-means model. The
    saturated model's log-likelihood has a closed form -- each cell's MLE probability is
    just its own empirical goal rate -- so it never requires solving a system that could be
    singular, sidestepping the failure mode entirely. This is the standard deviance /
    goodness-of-fit test for an interaction in a contingency-table setting, and is at least
    as statistically appropriate for two categoricals as the logistic-interaction-term LR
    test used for continuous/mixed pairs (per Cameron & Trivedi's treatment of saturated
    binary-response models; 0*log(0) terms are handled by the standard convention that they
    contribute 0 to the log-likelihood)."""
    sub = df[[y_col, a_col, b_col]].dropna().copy()
    n = len(sub)
    if n < MIN_N or sub[y_col].nunique() < 2:
        return InteractionResult(n, None, None, None, None, None, None, "insufficient_data")

    y = sub[y_col].astype(int)
    A = _design(sub[a_col], "a", True)
    B = _design(sub[b_col], "b", True)
    if A.shape[1] == 0 or B.shape[1] == 0:
        return InteractionResult(n, None, None, None, None, None, None, "insufficient_data")

    X_add = add_constant(pd.concat([A, B], axis=1), has_constant="add")
    try:
        m_add = Logit(y, X_add).fit(disp=0, maxiter=100)
    except Exception:
        return InteractionResult(n, None, None, None, None, None, None, "fit_failed")
    if not np.isfinite(m_add.llf):
        return InteractionResult(n, None, None, None, None, None, None, "fit_failed")

    cell = sub.groupby([a_col, b_col], observed=True)[y_col].agg(["size", "sum"])
    n_cells = len(cell)
    llf_sat = 0.0
    for cell_n, cell_g in zip(cell["size"], cell["sum"]):
        p = cell_g / cell_n
        if cell_g > 0:
            llf_sat += cell_g * np.log(p)
        if cell_n - cell_g > 0:
            llf_sat += (cell_n - cell_g) * np.log(1 - p)

    dof = n_cells - X_add.shape[1]
    if dof <= 0:
        return InteractionResult(n, None, None, None, None, None, None, "fit_failed")

    lr_stat = float(2 * (llf_sat - m_add.llf))
    if lr_stat < -1e-6:
        return InteractionResult(n, None, None, None, None, None, None, "fit_failed")
    lr_stat = max(lr_stat, 0.0)
    p_raw = float(scipy_stats.chi2.sf(lr_stat, dof))

    return InteractionResult(n, None, None, p_raw, lr_stat, None, None, "ok_saturated_fallback")


def validates_categorical_fallback_on_split(
    df: pd.DataFrame, y_col: str, a_col: str, b_col: str, alpha: float = 0.05
) -> bool:
    """Validation-split check for the saturated-fallback method: refit on the held-out
    split, True if it still fits (no re-triggering of the same sparsity failure there) and
    is still significant at `alpha`. No sign-check (matches `validates_on_split`'s own
    behavior for categorical x categorical pairs, where `interaction_coef` is always None --
    multi-df, no single coefficient to compare)."""
    result = fit_categorical_interaction_saturated(df, y_col, a_col, b_col)
    if result.fit_status != "ok_saturated_fallback" or result.interaction_p_raw is None:
        return False
    return result.interaction_p_raw < alpha


def validates_on_split(
    df: pd.DataFrame,
    y_col: str,
    a_col: str,
    b_col: str,
    train_sign: float | None,
    a_categorical: bool = False,
    b_categorical: bool = False,
    alpha: float = 0.05,
) -> bool:
    """Refit on a held-out split; True if the interaction is still significant at `alpha`
    and (for a single-df numeric:numeric interaction) the sign matches train's."""
    result = fit_interaction(df, y_col, a_col, b_col, a_categorical, b_categorical)
    if result.fit_status != "ok" or result.interaction_p_raw is None:
        return False
    if result.interaction_p_raw >= alpha:
        return False
    if train_sign is not None and result.interaction_coef is not None:
        if np.sign(result.interaction_coef) != np.sign(train_sign):
            return False
    return True
