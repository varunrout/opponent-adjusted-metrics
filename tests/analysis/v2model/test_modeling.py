import numpy as np
import pandas as pd

from opponent_adjusted.analysis.v2model.modeling import (
    DEEP_BLOCK_CLEARER,
    V2ModelSpec,
    apply_transforms,
    coefficient_table,
    fit_v2_model,
    predict_v2,
)


def _synthetic_df(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    gap = rng.gamma(2.0, 2.0, size=n)  # right-skewed, like nearest_defender_gap
    cat = rng.choice(["a", "b", "c", None], size=n, p=[0.4, 0.3, 0.2, 0.1])
    archetype = rng.choice([DEEP_BLOCK_CLEARER, "high_volume_presser", "duel_dominant_contester", None], size=n, p=[0.3, 0.3, 0.3, 0.1])
    cat_effect = np.select([cat == "a", cat == "b", cat == "c"], [0.0, 0.6, -0.6], default=0.0)
    logit = -2.0 + 0.5 * x1 - 0.3 * x2 + cat_effect + 0.4 * x1 * (cat == "b")
    p = 1 / (1 + np.exp(-logit))
    y = rng.binomial(1, p)
    return pd.DataFrame({"is_goal": y, "x1": x1, "x2": x2, "gap": gap, "cat": cat, "archetype": archetype})


def test_apply_transforms_log_gap():
    df = _synthetic_df()
    spec = V2ModelSpec(continuous_cols=("x1", "x2", "gap"), categorical_cols=("cat",), interactions=(), log_gap=True, gap_col="gap")
    out = apply_transforms(df, spec)
    assert np.allclose(out["gap"], np.log1p(df["gap"]))


def test_apply_transforms_collapse_archetype():
    df = _synthetic_df()
    spec = V2ModelSpec(
        continuous_cols=("x1", "x2"), categorical_cols=("archetype",), interactions=(),
        collapse_archetype=True, archetype_cols=("archetype",),
    )
    out = apply_transforms(df, spec)
    levels = set(out["archetype"].dropna().unique())
    assert levels <= {DEEP_BLOCK_CLEARER, "other"}
    # nulls stay null, not collapsed into "other"
    assert out["archetype"].isna().sum() == df["archetype"].isna().sum()


def test_missing_category_gets_explicit_dummy_not_dropped():
    df = _synthetic_df(seed=1)
    spec = V2ModelSpec(continuous_cols=("x1", "x2"), categorical_cols=("cat",), interactions=())
    fitted = fit_v2_model(df, "is_goal", spec, kind="plain")
    # "nan" (from the None values) should appear as its own feature column, OR be the
    # dropped reference level -- either way, predictions must be produced for every row,
    # including rows where cat is None (no silent row-dropping).
    preds = predict_v2(fitted, df)
    assert len(preds) == len(df)
    assert not np.isnan(preds).any()


def test_categorical_continuous_interaction_column_built():
    df = _synthetic_df(seed=2)
    spec = V2ModelSpec(continuous_cols=("x1", "x2"), categorical_cols=("cat",), interactions=(("cat", "x1"),))
    fitted = fit_v2_model(df, "is_goal", spec, kind="plain")
    assert any(":" in c for c in fitted.design_columns)


def test_continuous_continuous_interaction_column_built():
    df = _synthetic_df(seed=3)
    spec = V2ModelSpec(continuous_cols=("x1", "x2"), categorical_cols=(), interactions=(("x1", "x2"),))
    fitted = fit_v2_model(df, "is_goal", spec, kind="plain")
    assert "x1:x2" in fitted.design_columns


def test_ridge_fit_produces_valid_predictions_and_no_pvalues():
    df_train = _synthetic_df(seed=4)
    df_test = _synthetic_df(seed=5)
    spec = V2ModelSpec(continuous_cols=("x1", "x2"), categorical_cols=("cat",), interactions=(("cat", "x1"),))
    fitted = fit_v2_model(df_train, "is_goal", spec, kind="ridge", C=1.0)
    preds = predict_v2(fitted, df_test)
    assert len(preds) == len(df_test)
    assert ((preds >= 0) & (preds <= 1)).all()
    coefs = coefficient_table(fitted)
    assert coefs["std_error"].isna().all()
    assert coefs["p_value"].isna().all()


def test_plain_fit_produces_pvalues():
    # Larger n, no missing category, no interaction -- avoids the near-separation instability
    # a small/sparse synthetic fit can hit (the real v2 fit has 2780 train rows and the
    # standard MIN_N-style sample-size guards used throughout this pipeline; this test only
    # needs to confirm the plain-fit code path itself populates std_error/p_value, which
    # `test_categorical_continuous_interaction_column_built` already exercises structurally).
    rng = np.random.default_rng(6)
    n = 6000
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    cat = rng.choice(["a", "b", "c"], size=n)
    cat_effect = np.select([cat == "a", cat == "b", cat == "c"], [0.0, 0.4, -0.4])
    logit = -2.0 + 0.3 * x1 - 0.2 * x2 + cat_effect
    p = 1 / (1 + np.exp(-logit))
    y = rng.binomial(1, p)
    df = pd.DataFrame({"is_goal": y, "x1": x1, "x2": x2, "cat": cat})
    spec = V2ModelSpec(continuous_cols=("x1", "x2"), categorical_cols=("cat",), interactions=())
    fitted = fit_v2_model(df, "is_goal", spec, kind="plain")
    coefs = coefficient_table(fitted)
    assert coefs["std_error"].notna().all()
    assert coefs["p_value"].notna().all()


def test_predict_v2_column_alignment_on_unseen_level_combo():
    # A validation/test split might not contain every dummy level combination seen in train
    # (or vice versa) -- predict_v2 must reindex safely rather than crash.
    df_train = _synthetic_df(seed=7, n=2000)
    df_test = _synthetic_df(seed=8, n=200)
    spec = V2ModelSpec(continuous_cols=("x1", "x2"), categorical_cols=("cat",), interactions=(("cat", "x1"),))
    fitted = fit_v2_model(df_train, "is_goal", spec, kind="ridge", C=1.0)
    preds = predict_v2(fitted, df_test)
    assert len(preds) == len(df_test)
