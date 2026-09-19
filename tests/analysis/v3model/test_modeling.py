import numpy as np
import pandas as pd
import pytest

from opponent_adjusted.analysis.v3model.modeling import (
    V3ModelSpec,
    apply_transforms,
    coefficient_table,
    fit_v3_model,
    predict_v3,
)


def _toy_df(n=200, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    cat = rng.choice(["x", "y", "z"], size=n)
    logit = 0.8 * a - 0.5 * b + (cat == "x").astype(float) * 0.6
    p = 1 / (1 + np.exp(-logit))
    y = rng.binomial(1, p)
    return pd.DataFrame({"a": a, "b": b, "cat": cat, "is_goal": y})


def test_apply_transforms_log1p_only_touches_listed_columns():
    df = pd.DataFrame({"rate": [0.0, 1.0, 3.0], "other": [5.0, 5.0, 5.0]})
    spec = V3ModelSpec(continuous_cols=("rate", "other"), log1p_cols=("rate",))
    out = apply_transforms(df, spec)
    assert out["rate"].tolist() == pytest.approx(np.log1p([0.0, 1.0, 3.0]).tolist())
    assert out["other"].tolist() == [5.0, 5.0, 5.0]


def test_fit_plain_no_categorical_no_interaction():
    df = _toy_df()
    spec = V3ModelSpec(continuous_cols=("a", "b"))
    fitted = fit_v3_model(df, "is_goal", spec, kind="plain")
    preds = predict_v3(fitted, df)
    assert preds.shape == (len(df),)
    assert ((preds >= 0) & (preds <= 1)).all()


def test_fit_ridge_matches_shape_and_bounded_probabilities():
    df = _toy_df()
    spec = V3ModelSpec(continuous_cols=("a", "b"), categorical_cols=("cat",))
    fitted = fit_v3_model(df, "is_goal", spec, kind="ridge", C=1.0)
    preds = predict_v3(fitted, df)
    assert ((preds > 0) & (preds < 1)).all()


def test_continuous_continuous_interaction_term_present():
    df = _toy_df()
    spec = V3ModelSpec(continuous_cols=("a", "b"), interactions=(("a", "b"),))
    fitted = fit_v3_model(df, "is_goal", spec, kind="plain")
    assert "a:b" in fitted.design_columns


def test_categorical_continuous_interaction_term_present():
    df = _toy_df()
    spec = V3ModelSpec(continuous_cols=("a",), categorical_cols=("cat",), interactions=(("cat", "a"),))
    fitted = fit_v3_model(df, "is_goal", spec, kind="plain")
    assert any(c.startswith("cat_") and c.endswith(":a") for c in fitted.design_columns)


def test_missing_indicator_column_added_and_imputed_not_dropped():
    df = _toy_df()
    df.loc[df.index[:20], "a"] = np.nan
    spec = V3ModelSpec(continuous_cols=("a", "b"), missing_indicator_cols=("a",))
    fitted = fit_v3_model(df, "is_goal", spec, kind="plain")
    assert "a_was_missing" in fitted.design_columns
    preds = predict_v3(fitted, df)
    assert preds.shape == (len(df),)  # no row dropped despite the nulls


def test_missing_indicator_flags_exactly_the_null_rows():
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0, np.nan], "b": [1.0, 2.0, 3.0, 4.0], "is_goal": [0, 1, 0, 1]})
    spec = V3ModelSpec(continuous_cols=("a", "b"), missing_indicator_cols=("a",))
    fitted = fit_v3_model(df, "is_goal", spec, kind="ridge")
    from opponent_adjusted.analysis.v3model.modeling import _build_matrix
    X = _build_matrix(df, spec, fitted.imputer, fitted.scaler, fitted.categorical_levels)
    assert X["a_was_missing"].tolist() == [0.0, 1.0, 0.0, 1.0]


def test_missing_categorical_gets_explicit_nan_category_not_dropped():
    df = _toy_df()
    df["cat"] = df["cat"].astype(object)
    df.loc[df.index[:10], "cat"] = None
    spec = V3ModelSpec(continuous_cols=("a",), categorical_cols=("cat",))
    fitted = fit_v3_model(df, "is_goal", spec, kind="plain")
    preds = predict_v3(fitted, df)
    assert preds.shape == (len(df),)


def test_plain_and_ridge_coefficient_tables_have_expected_shape():
    df = _toy_df()
    spec = V3ModelSpec(continuous_cols=("a", "b"))
    plain = coefficient_table(fit_v3_model(df, "is_goal", spec, kind="plain"))
    ridge = coefficient_table(fit_v3_model(df, "is_goal", spec, kind="ridge"))
    assert set(plain.columns) == {"feature", "coefficient", "std_error", "p_value"}
    assert set(ridge.columns) == {"feature", "coefficient", "std_error", "p_value"}
    assert ridge["std_error"].isna().all()
    assert not plain["std_error"].isna().all()


def test_predict_reindexes_missing_dummy_level_to_zero_not_error():
    df = _toy_df()
    spec = V3ModelSpec(continuous_cols=("a",), categorical_cols=("cat",))
    fitted = fit_v3_model(df, "is_goal", spec, kind="ridge")
    unseen = df.copy()
    unseen["cat"] = "brand_new_level_never_seen_in_train"
    preds = predict_v3(fitted, unseen)
    assert preds.shape == (len(unseen),)
