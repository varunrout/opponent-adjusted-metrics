import numpy as np
import pandas as pd

from opponent_adjusted.analysis.bivariate.testing import (
    fit_categorical_interaction_saturated,
    fit_interaction,
    validates_categorical_fallback_on_split,
    validates_on_split,
)


def _synthetic_interaction_df(n=4000, seed=0, interaction_strength=2.0):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    logit = 0.1 * a + 0.1 * b + interaction_strength * a * b
    p = 1 / (1 + np.exp(-logit))
    y = rng.binomial(1, p)
    return pd.DataFrame({"is_goal": y, "a": a, "b": b})


def _synthetic_null_df(n=4000, seed=1):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    logit = 0.1 * a + 0.1 * b
    p = 1 / (1 + np.exp(-logit))
    y = rng.binomial(1, p)
    return pd.DataFrame({"is_goal": y, "a": a, "b": b})


def test_detects_real_interaction():
    df = _synthetic_interaction_df()
    result = fit_interaction(df, "is_goal", "a", "b")
    assert result.fit_status == "ok"
    assert result.interaction_p_raw < 0.001
    assert result.interaction_coef is not None and result.interaction_coef > 0


def test_null_interaction_not_significant():
    df = _synthetic_null_df()
    result = fit_interaction(df, "is_goal", "a", "b")
    assert result.fit_status == "ok"
    assert result.interaction_p_raw > 0.05


def test_insufficient_data_recorded_not_dropped():
    df = pd.DataFrame({"is_goal": [1, 0, 1], "a": [0.1, 0.2, 0.3], "b": [1.0, 2.0, 3.0]})
    result = fit_interaction(df, "is_goal", "a", "b")
    assert result.fit_status == "insufficient_data"
    assert result.interaction_p_raw is None
    assert result.n_train == 3  # still recorded, not silently dropped


def test_categorical_interaction_multi_df():
    rng = np.random.default_rng(2)
    n = 3000
    a = rng.normal(size=n)
    cat = rng.choice(["x", "y", "z", "w"], size=n)
    cat_effect = np.select([cat == "x", cat == "y", cat == "z", cat == "w"], [0.0, 1.5, -1.5, 0.5])
    logit = 0.1 * a + 2.0 * a * cat_effect
    p = 1 / (1 + np.exp(-logit))
    y = rng.binomial(1, p)
    df = pd.DataFrame({"is_goal": y, "a": a, "cat": cat})
    result = fit_interaction(df, "is_goal", "a", "cat", b_categorical=True)
    assert result.fit_status == "ok"
    assert result.interaction_p_raw < 0.01
    # categorical interaction is multi-df -- no single coef/se reported
    assert result.interaction_coef is None
    assert result.interaction_se is None
    assert result.lr_stat is not None


def test_validates_on_split_rejects_wrong_sign():
    train_df = _synthetic_interaction_df(seed=10, interaction_strength=2.0)
    train_result = fit_interaction(train_df, "is_goal", "a", "b")
    flipped_sign_df = _synthetic_interaction_df(seed=11, interaction_strength=-2.0)
    ok = validates_on_split(flipped_sign_df, "is_goal", "a", "b", train_sign=train_result.interaction_coef)
    assert ok is False


def test_validates_on_split_accepts_matching_sign():
    train_df = _synthetic_interaction_df(seed=20, interaction_strength=2.0)
    train_result = fit_interaction(train_df, "is_goal", "a", "b")
    val_df = _synthetic_interaction_df(seed=21, interaction_strength=2.0)
    ok = validates_on_split(val_df, "is_goal", "a", "b", train_sign=train_result.interaction_coef)
    assert ok is True


def _df_with_structural_zero_cell(n=3000, seed=0):
    """Mirrors the real nearest_defender_role x second_nearest_defender_role failure: a
    3-level x 3-level categorical pair where one combination never co-occurs (like GK x GK
    being impossible -- a team has one goalkeeper), plus a couple of tiny all-zero-outcome
    cells that trigger complete separation in the unregularized interaction-model MLE."""
    rng = np.random.default_rng(seed)
    a_levels = np.array(["x", "y", "z"])
    b_levels = np.array(["p", "q", "r"])
    a = rng.choice(a_levels, size=n)
    b = np.empty(n, dtype=object)
    for i in range(n):
        if a[i] == "z":
            # "z" never co-occurs with "r" -- the structural zero cell.
            b[i] = rng.choice(["p", "q"])
        else:
            b[i] = rng.choice(b_levels)
    logit = -2.5 + 0.1 * (a == "x") + 0.1 * (b == "p")
    p = 1 / (1 + np.exp(-logit))
    y = rng.binomial(1, p)
    # Force one small cell to be complete-separation (all-zero outcome), like
    # Fullback_WingBack x Fullback_WingBack (n=16, 0 goals) in the real data.
    mask = (a == "y") & (b == "q")
    idx = np.where(mask)[0][:15]
    y[idx] = 0
    return pd.DataFrame({"is_goal": y, "a": a, "b": b})


def test_fit_interaction_fails_on_structural_zero_cell():
    # Confirms the setup actually reproduces the real failure mode before testing the fix.
    df = _df_with_structural_zero_cell()
    result = fit_interaction(df, "is_goal", "a", "b", a_categorical=True, b_categorical=True)
    assert result.fit_status == "fit_failed"


def test_categorical_saturated_fallback_succeeds_where_standard_fit_fails():
    df = _df_with_structural_zero_cell()
    result = fit_categorical_interaction_saturated(df, "is_goal", "a", "b")
    assert result.fit_status == "ok_saturated_fallback"
    assert result.interaction_p_raw is not None
    assert 0.0 <= result.interaction_p_raw <= 1.0
    assert result.lr_stat is not None and result.lr_stat >= 0
    # Multi-df categorical interaction -- no single coefficient, matches the existing
    # convention for categorical x categorical pairs in fit_interaction.
    assert result.interaction_coef is None
    assert result.main_effect_a_coef is None


def test_categorical_saturated_fallback_detects_real_interaction():
    rng = np.random.default_rng(7)
    n = 4000
    a = rng.choice(["x", "y", "z"], size=n)
    b = rng.choice(["p", "q", "r"], size=n)
    # Strong interaction: only (x, p) and (y, q) have elevated goal rate.
    logit = np.where(((a == "x") & (b == "p")) | ((a == "y") & (b == "q")), 2.0, -3.0)
    prob = 1 / (1 + np.exp(-logit))
    y = rng.binomial(1, prob)
    df = pd.DataFrame({"is_goal": y, "a": a, "b": b})
    result = fit_categorical_interaction_saturated(df, "is_goal", "a", "b")
    assert result.fit_status == "ok_saturated_fallback"
    assert result.interaction_p_raw < 0.01


def test_categorical_saturated_fallback_null_when_no_interaction():
    rng = np.random.default_rng(8)
    n = 4000
    a = rng.choice(["x", "y", "z"], size=n)
    b = rng.choice(["p", "q", "r"], size=n)
    a_effect = np.select([a == "x", a == "y", a == "z"], [0.3, -0.2, 0.0])
    b_effect = np.select([b == "p", b == "q", b == "r"], [0.2, -0.1, 0.0])
    logit = -2.5 + a_effect + b_effect  # purely additive, no interaction
    prob = 1 / (1 + np.exp(-logit))
    y = rng.binomial(1, prob)
    df = pd.DataFrame({"is_goal": y, "a": a, "b": b})
    result = fit_categorical_interaction_saturated(df, "is_goal", "a", "b")
    assert result.fit_status == "ok_saturated_fallback"
    assert result.interaction_p_raw > 0.05


def test_validates_categorical_fallback_on_split():
    rng = np.random.default_rng(9)

    def make(seed):
        r = np.random.default_rng(seed)
        n = 4000
        a = r.choice(["x", "y", "z"], size=n)
        b = r.choice(["p", "q", "r"], size=n)
        logit = np.where(((a == "x") & (b == "p")) | ((a == "y") & (b == "q")), 2.0, -3.0)
        prob = 1 / (1 + np.exp(-logit))
        y = r.binomial(1, prob)
        return pd.DataFrame({"is_goal": y, "a": a, "b": b})

    val_df = make(10)
    assert validates_categorical_fallback_on_split(val_df, "is_goal", "a", "b") is True

    # Null (no-interaction) data should not validate.
    n = 4000
    a = rng.choice(["x", "y", "z"], size=n)
    b = rng.choice(["p", "q", "r"], size=n)
    prob = np.full(n, 0.1)
    y = rng.binomial(1, prob)
    null_df = pd.DataFrame({"is_goal": y, "a": a, "b": b})
    assert validates_categorical_fallback_on_split(null_df, "is_goal", "a", "b") is False
