import numpy as np
import pandas as pd

from opponent_adjusted.analysis.bivariate.testing import fit_interaction, validates_on_split


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
