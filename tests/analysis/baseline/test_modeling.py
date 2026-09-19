import numpy as np
import pandas as pd

from opponent_adjusted.analysis.baseline.modeling import (
    calibration_table,
    coefficient_table,
    dumb_baseline_prob,
    fit_v1_model,
    predict_v1,
    score_predictions,
)


def _synthetic_df(n=3000, seed=0, goal_rate=0.12):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    cat = rng.choice(["a", "b", "c"], size=n)
    cat_effect = np.select([cat == "a", cat == "b", cat == "c"], [0.0, 0.8, -0.8])
    logit = -2.0 + 1.5 * x1 - 0.5 * x2 + cat_effect
    p = 1 / (1 + np.exp(-logit))
    y = rng.binomial(1, p)
    return pd.DataFrame({"is_goal": y, "x1": x1, "x2": x2, "cat": cat})


def test_dumb_baseline_prob_is_train_mean():
    df = _synthetic_df()
    prob = dumb_baseline_prob(df, "is_goal")
    assert abs(prob - df["is_goal"].mean()) < 1e-9


def test_fit_v1_model_recovers_signal():
    df = _synthetic_df(seed=1)
    fitted = fit_v1_model(df, "is_goal", ("x1", "x2"), categorical_col="cat")
    coefs = coefficient_table(fitted)
    x1_coef = coefs.loc[coefs["feature"] == "x1", "coefficient"].iloc[0]
    x2_coef = coefs.loc[coefs["feature"] == "x2", "coefficient"].iloc[0]
    assert x1_coef > 0  # matches synthetic positive relationship (scaled, so sign is what matters)
    assert x2_coef < 0
    # drop-first categorical: only 2 of the 3 levels get their own dummy column
    cat_features = [f for f in coefs["feature"] if f.startswith("cat_")]
    assert len(cat_features) == 2


def test_predict_v1_beats_constant_on_log_loss():
    df_train = _synthetic_df(seed=2)
    df_test = _synthetic_df(seed=3)
    fitted = fit_v1_model(df_train, "is_goal", ("x1", "x2"), categorical_col="cat")
    v1_pred = predict_v1(fitted, df_test)
    dumb_pred = np.full(len(df_test), dumb_baseline_prob(df_train, "is_goal"))

    v1_metrics = score_predictions(df_test["is_goal"], v1_pred)
    dumb_metrics = score_predictions(df_test["is_goal"], dumb_pred)
    assert v1_metrics["log_loss"] < dumb_metrics["log_loss"]
    assert v1_metrics["brier_score"] < dumb_metrics["brier_score"]
    assert v1_metrics["roc_auc"] is not None and v1_metrics["roc_auc"] > 0.5


def test_score_predictions_constant_predictor_has_no_auc():
    y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    pred = np.full(10, 0.12)
    metrics = score_predictions(y, pred)
    assert metrics["roc_auc"] is None
    assert metrics["log_loss"] is not None and metrics["brier_score"] is not None


def test_calibration_table_shape_and_bounds():
    rng = np.random.default_rng(4)
    y = pd.Series(rng.binomial(1, 0.15, size=500))
    pred = rng.uniform(0.01, 0.4, size=500)
    table = calibration_table(y, pred, n_bins=10)
    assert len(table) == 10
    assert table["n"].sum() == 500
    assert (table["mean_predicted"] >= 0).all() and (table["mean_predicted"] <= 1).all()


def test_calibration_table_handles_constant_predictions():
    rng = np.random.default_rng(5)
    y = pd.Series(rng.binomial(1, 0.15, size=200))
    pred = np.full(200, 0.15)
    table = calibration_table(y, pred, n_bins=10)
    assert len(table) == 10
    # every bin's predicted value is identical for a constant predictor
    assert table["mean_predicted"].nunique() == 1
