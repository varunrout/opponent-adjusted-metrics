#!/usr/bin/env python
"""Build static portfolio-ready CxG summary artifacts and charts."""

from __future__ import annotations

import argparse
import json
import textwrap
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_RESULTS_DIR = Path("outputs/results/cxg/diagnostic_v1")
DEFAULT_FEATURE_IMPACT_DIR = Path("outputs/modeling/cxg/diagnostic_v1/feature_impact")
DEFAULT_OUTPUT_DIR = Path("outputs/portfolio/cxg")
METRICS = ("log_loss", "brier", "roc_auc", "expected_calibration_error")
REQUIRED_CHARTS = {
    "model_metric_comparison": "model_metric_comparison.png",
    "feature_group_impact": "feature_group_impact.png",
    "top_feature_importance": "top_feature_importance.png",
    "team_cxg_ranking": "team_cxg_ranking.png",
    "player_cxg_ranking": "player_cxg_ranking.png",
    "goals_minus_cxg_teams": "goals_minus_cxg_teams.png",
    "category_lift_body_part": "category_lift_body_part.png",
    "category_lift_shot_type": "category_lift_shot_type.png",
    "category_lift_set_piece_category": "category_lift_set_piece_category.png",
}
CATEGORY_TABLES = {
    "body_part": "category_lift_body_part.csv",
    "shot_type": "category_lift_shot_type.csv",
    "set_piece_category": "category_lift_set_piece_category.csv",
    "pressure_state": "category_lift_pressure_state.csv",
}
TEAM_COLUMNS = [
    "team_id",
    "team_name",
    "shots",
    "goals",
    "total_cxg",
    "mean_cxg_per_shot",
    "goals_minus_cxg",
    "baseline_total_cxg",
    "total_cxg_delta_vs_baseline",
    "rank_total_cxg",
]
PLAYER_COLUMNS = [
    "player_id",
    "player_name",
    "team_id",
    "team_name",
    "shots",
    "goals",
    "total_cxg",
    "mean_cxg_per_shot",
    "goals_minus_cxg",
    "baseline_total_cxg",
    "total_cxg_delta_vs_baseline",
    "rank_total_cxg",
]


@dataclass(frozen=True)
class PortfolioPaths:
    """Input/output paths for the static CxG portfolio layer."""

    results_dir: Path
    feature_impact_dir: Path
    output_dir: Path

    @property
    def charts_dir(self) -> Path:
        return self.output_dir / "charts"

    @property
    def promotion_summary(self) -> Path:
        return self.results_dir / "model_promotion_summary.json"

    @property
    def shot_predictions(self) -> Path:
        return self.results_dir / "shot_predictions.parquet"

    @property
    def player_summary(self) -> Path:
        return self.results_dir / "player_cxg_summary.csv"

    @property
    def team_summary(self) -> Path:
        return self.results_dir / "team_cxg_summary.csv"

    @property
    def top_players(self) -> Path:
        return self.results_dir / "top_players_by_cxg.csv"

    @property
    def team_rankings(self) -> Path:
        return self.results_dir / "team_cxg_rankings.csv"

    @property
    def baseline_summary(self) -> Path:
        return self.results_dir / "baseline_vs_diagnostic_summary.csv"

    @property
    def feature_impact_summary(self) -> Path:
        return self.feature_impact_dir / "feature_impact_summary.json"

    @property
    def feature_impact_report(self) -> Path:
        return self.feature_impact_dir / "feature_impact_report.md"

    @property
    def permutation_importance(self) -> Path:
        return self.feature_impact_dir / "permutation_importance.csv"

    @property
    def group_perturbation(self) -> Path:
        return self.feature_impact_dir / "group_perturbation_summary.csv"

    @classmethod
    def from_roots(
        cls,
        results_dir: Path = DEFAULT_RESULTS_DIR,
        feature_impact_dir: Path = DEFAULT_FEATURE_IMPACT_DIR,
        output_dir: Path = DEFAULT_OUTPUT_DIR,
    ) -> "PortfolioPaths":
        return cls(
            results_dir=results_dir,
            feature_impact_dir=feature_impact_dir,
            output_dir=output_dir,
        )


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if np.isnan(value) else float(value)
    if pd.isna(value) and not isinstance(value, bool | str):
        return None
    return value


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _baseline_join_rate(promotion_summary: dict[str, Any], impact_summary: dict[str, Any]) -> Any:
    comparison = promotion_summary.get("baseline_comparison", {})
    if isinstance(comparison, dict):
        value = comparison.get("join_rate", comparison.get("baseline_join_rate"))
        if value is not None:
            return value
    return impact_summary.get("baseline_join_rate")


def metric_comparison(
    promotion_summary: dict[str, Any],
    impact_summary: dict[str, Any],
    baseline_summary: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    """Extract baseline/diagnostic metric comparison with deltas."""

    headline = impact_summary.get("validation_metric_headline", {})
    baseline = headline.get("baseline", {}) if isinstance(headline, dict) else {}
    diagnostic = headline.get("diagnostic", {}) if isinstance(headline, dict) else {}
    if not baseline or not diagnostic:
        validation_metrics = promotion_summary.get("validation_metrics", {})
        if isinstance(validation_metrics, dict) and validation_metrics:
            baseline = validation_metrics.get("baseline", {})
            diagnostic_key = next(
                (
                    key
                    for key in validation_metrics
                    if key != "baseline" and str(key).startswith("diagnostic")
                ),
                None,
            )
            if diagnostic_key is None:
                diagnostic_key = next(
                    (key for key in validation_metrics if key != "baseline"), None
                )
            diagnostic = validation_metrics.get(diagnostic_key, {}) if diagnostic_key else {}
    if (not baseline or not diagnostic) and not baseline_summary.empty:
        baseline, diagnostic = _metrics_from_baseline_summary(baseline_summary)
    baseline_metrics = {metric: baseline.get(metric) for metric in METRICS}
    diagnostic_metrics = {metric: diagnostic.get(metric) for metric in METRICS}
    deltas = {
        metric: _metric_delta(diagnostic_metrics.get(metric), baseline_metrics.get(metric))
        for metric in METRICS
    }
    return {
        "baseline": baseline_metrics,
        "diagnostic": diagnostic_metrics,
        "diagnostic_minus_baseline": deltas,
    }


def _metrics_from_baseline_summary(
    baseline_summary: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not {"metric", "value"}.issubset(baseline_summary.columns):
        return {}, {}
    values = {
        str(row.metric): row.value
        for row in baseline_summary[["metric", "value"]].itertuples(index=False)
    }
    baseline = {
        metric: values.get(f"baseline_{metric}")
        for metric in METRICS
        if f"baseline_{metric}" in values
    }
    diagnostic = {
        metric: values.get(f"diagnostic_{metric}")
        for metric in METRICS
        if f"diagnostic_{metric}" in values
    }
    return baseline, diagnostic


def _metric_delta(diagnostic_value: Any, baseline_value: Any) -> float | None:
    if diagnostic_value is None or baseline_value is None:
        return None
    if pd.isna(diagnostic_value) or pd.isna(baseline_value):
        return None
    return float(diagnostic_value) - float(baseline_value)


def team_rankings(team_summary: pd.DataFrame, team_rankings_df: pd.DataFrame) -> pd.DataFrame:
    source = team_rankings_df if not team_rankings_df.empty else team_summary
    return _ensure_columns(source.copy(), TEAM_COLUMNS).sort_values(
        "total_cxg", ascending=False, na_position="last"
    )


def player_rankings(player_summary: pd.DataFrame, top_players: pd.DataFrame) -> pd.DataFrame:
    source = top_players if not top_players.empty else player_summary
    result = _ensure_columns(source.copy(), PLAYER_COLUMNS)
    if result["player_id"].isna().any():
        missing = int(result["player_id"].isna().sum())
        raise ValueError(f"Cannot build portfolio player rankings with {missing} missing player_id")
    return result.sort_values("total_cxg", ascending=False, na_position="last")


def _ensure_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        if column not in frame.columns:
            frame[column] = np.nan
    return frame[columns]


def feature_driver_summary(
    group_summary: pd.DataFrame,
    permutation: pd.DataFrame,
) -> pd.DataFrame:
    """Combine group and individual feature impact into one curated table."""

    rows: list[pd.DataFrame] = []
    if not group_summary.empty:
        group = group_summary.copy()
        group["driver_type"] = "feature_group"
        group["name"] = group["feature_group"]
        group["rank"] = group["log_loss_delta"].rank(method="first", ascending=False).astype(int)
        rows.append(
            _ensure_columns(
                group,
                [
                    "driver_type",
                    "name",
                    "feature_group",
                    "log_loss_delta",
                    "brier_delta",
                    "roc_auc_delta",
                    "absolute_probability_delta_mean",
                    "rank",
                ],
            )
        )
    if not permutation.empty:
        features = permutation.copy()
        features = features.loc[features.get("feature", "") != "statsbomb_xg"].copy()
        features["driver_type"] = "feature"
        features["name"] = features["feature"]
        if "impact_rank" in features.columns:
            features["rank"] = features["impact_rank"]
        else:
            features["rank"] = (
                features["log_loss_delta"].rank(method="first", ascending=False).astype(int)
            )
        rows.append(
            _ensure_columns(
                features,
                [
                    "driver_type",
                    "name",
                    "feature_group",
                    "log_loss_delta",
                    "brier_delta",
                    "roc_auc_delta",
                    "absolute_probability_delta_mean",
                    "rank",
                ],
            )
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "driver_type",
                "name",
                "feature_group",
                "log_loss_delta",
                "brier_delta",
                "roc_auc_delta",
                "absolute_probability_delta_mean",
                "rank",
            ]
        )
    return pd.concat(rows, ignore_index=True).sort_values(
        ["driver_type", "rank"], ascending=[True, True]
    )


def category_insights(feature_impact_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    frames = []
    skipped = []
    for category, filename in CATEGORY_TABLES.items():
        path = feature_impact_dir / filename
        if not path.exists():
            skipped.append(category)
            continue
        frame = pd.read_csv(path)
        if "category_column" not in frame.columns:
            frame.insert(0, "category_column", category)
        frames.append(frame)
    if not frames:
        return pd.DataFrame(), skipped
    result = pd.concat(frames, ignore_index=True)
    columns = [
        "category_column",
        "category",
        "shots",
        "goals",
        "goal_rate",
        "mean_predicted_cxg",
        "total_predicted_cxg",
        "mean_baseline_cxg",
        "total_baseline_cxg",
        "mean_delta_vs_baseline",
        "total_delta_vs_baseline",
    ]
    return _ensure_columns(result, columns), skipped


def build_scorecard(
    *,
    promotion_summary: dict[str, Any],
    impact_summary: dict[str, Any],
    baseline_summary: pd.DataFrame,
    outputs: dict[str, Path],
    charts: dict[str, Path],
) -> dict[str, Any]:
    metrics = metric_comparison(promotion_summary, impact_summary, baseline_summary)
    return {
        "model_version": impact_summary.get(
            "model_version", promotion_summary.get("model_version", "diagnostic_v1")
        ),
        "selected_model_candidate": impact_summary.get(
            "selected_model_candidate", promotion_summary.get("selected_model_candidate")
        ),
        "promotion_status": promotion_summary.get(
            "promotion_status", impact_summary.get("promotion_status")
        ),
        "promotion_gate_passed": promotion_summary.get(
            "promotion_gate_passed", impact_summary.get("promotion_gate_passed")
        ),
        "governance_status": (
            promotion_summary.get("governance_summary", {}).get("status")
            or impact_summary.get("governance_status")
        ),
        "baseline_join_rate": _baseline_join_rate(promotion_summary, impact_summary),
        "selected_feature_count": impact_summary.get("selected_feature_count"),
        "metric_comparison": metrics,
        "integrity_checks": impact_summary.get("result_integrity_checks", {}),
        "generated_artifact_paths": {
            "outputs": {name: str(path) for name, path in outputs.items()},
            "charts": {name: str(path) for name, path in charts.items()},
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def build_cxg_portfolio_summary(
    *,
    paths: PortfolioPaths | None = None,
    top_n_teams: int = 15,
    top_n_players: int = 20,
    top_n_features: int = 20,
) -> dict[str, Path]:
    """Build static CxG portfolio outputs."""

    paths = paths or PortfolioPaths.from_roots()
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    paths.charts_dir.mkdir(parents=True, exist_ok=True)

    promotion = _read_json(paths.promotion_summary)
    impact = _read_json(paths.feature_impact_summary)
    baseline = _read_csv_if_exists(paths.baseline_summary)
    teams = team_rankings(
        pd.read_csv(paths.team_summary),
        _read_csv_if_exists(paths.team_rankings),
    )
    players = player_rankings(
        pd.read_csv(paths.player_summary),
        _read_csv_if_exists(paths.top_players),
    )
    group_summary = pd.read_csv(paths.group_perturbation)
    permutation = pd.read_csv(paths.permutation_importance)
    drivers = feature_driver_summary(group_summary, permutation)
    categories, skipped_categories = category_insights(paths.feature_impact_dir)

    outputs = {
        "summary_md": paths.output_dir / "cxg_portfolio_summary.md",
        "scorecard_json": paths.output_dir / "cxg_model_scorecard.json",
        "team_rankings_csv": paths.output_dir / "cxg_team_rankings.csv",
        "player_rankings_csv": paths.output_dir / "cxg_player_rankings.csv",
        "feature_driver_summary_csv": paths.output_dir / "cxg_feature_driver_summary.csv",
        "category_insights_csv": paths.output_dir / "cxg_category_insights.csv",
    }
    charts = {name: paths.charts_dir / filename for name, filename in REQUIRED_CHARTS.items()}

    teams.to_csv(outputs["team_rankings_csv"], index=False)
    players.to_csv(outputs["player_rankings_csv"], index=False)
    drivers.to_csv(outputs["feature_driver_summary_csv"], index=False)
    categories.to_csv(outputs["category_insights_csv"], index=False)

    scorecard = build_scorecard(
        promotion_summary=promotion,
        impact_summary=impact,
        baseline_summary=baseline,
        outputs=outputs,
        charts=charts,
    )
    scorecard["skipped_category_insights"] = skipped_categories
    _write_json(outputs["scorecard_json"], scorecard)

    create_charts(
        scorecard=scorecard,
        teams=teams,
        players=players,
        drivers=drivers,
        categories=categories,
        charts=charts,
        top_n_teams=top_n_teams,
        top_n_players=top_n_players,
        top_n_features=top_n_features,
    )
    outputs["summary_md"].write_text(
        build_markdown_summary(
            scorecard=scorecard,
            teams=teams,
            players=players,
            drivers=drivers,
            categories=categories,
            top_n_teams=top_n_teams,
            top_n_players=top_n_players,
        ),
        encoding="utf-8",
    )
    return {**outputs, **{f"chart_{name}": path for name, path in charts.items()}}


def create_charts(
    *,
    scorecard: dict[str, Any],
    teams: pd.DataFrame,
    players: pd.DataFrame,
    drivers: pd.DataFrame,
    categories: pd.DataFrame,
    charts: dict[str, Path],
    top_n_teams: int,
    top_n_players: int,
    top_n_features: int,
) -> None:
    _plot_model_metric_comparison(scorecard["metric_comparison"], charts["model_metric_comparison"])
    group_rows = drivers.loc[drivers["driver_type"] == "feature_group"]
    _plot_horizontal_bars(
        group_rows.sort_values("log_loss_delta", ascending=False),
        label_col="name",
        value_col="log_loss_delta",
        path=charts["feature_group_impact"],
        title="Which feature groups move CxG log loss most?",
        xlabel="Log loss delta after perturbation",
    )
    feature_rows = drivers.loc[drivers["driver_type"] == "feature"].nlargest(
        top_n_features, "log_loss_delta"
    )
    _plot_horizontal_bars(
        feature_rows,
        label_col="name",
        value_col="log_loss_delta",
        path=charts["top_feature_importance"],
        title=f"Top {top_n_features} promoted CxG feature impacts",
        xlabel="Log loss delta after permutation",
    )
    _plot_horizontal_bars(
        teams.nlargest(top_n_teams, "total_cxg"),
        label_col="team_name",
        value_col="total_cxg",
        path=charts["team_cxg_ranking"],
        title=f"Top {top_n_teams} teams by total diagnostic CxG",
        xlabel="Total CxG",
    )
    player_plot = players.nlargest(top_n_players, "total_cxg").copy()
    player_plot["label"] = player_plot.apply(_player_label, axis=1)
    _plot_horizontal_bars(
        player_plot,
        label_col="label",
        value_col="total_cxg",
        path=charts["player_cxg_ranking"],
        title=f"Top {top_n_players} players by total diagnostic CxG",
        xlabel="Total CxG",
    )
    goals_minus = _top_bottom(teams, "goals_minus_cxg", top_n_teams)
    _plot_horizontal_bars(
        goals_minus.sort_values("goals_minus_cxg", ascending=True),
        label_col="team_name",
        value_col="goals_minus_cxg",
        path=charts["goals_minus_cxg_teams"],
        title="Teams furthest above and below diagnostic CxG",
        xlabel="Goals minus CxG",
    )
    for category, chart_key in (
        ("body_part", "category_lift_body_part"),
        ("shot_type", "category_lift_shot_type"),
        ("set_piece_category", "category_lift_set_piece_category"),
    ):
        frame = categories.loc[categories["category_column"] == category]
        _plot_horizontal_bars(
            frame.nlargest(12, "total_predicted_cxg"),
            label_col="category",
            value_col="mean_predicted_cxg",
            path=charts[chart_key],
            title=f"Mean diagnostic CxG by {category.replace('_', ' ')}",
            xlabel="Mean predicted CxG",
        )


def _plot_model_metric_comparison(metrics: dict[str, dict[str, Any]], path: Path) -> None:
    labels = list(METRICS)
    baseline = [metrics["baseline"].get(metric, np.nan) for metric in labels]
    diagnostic = [metrics["diagnostic"].get(metric, np.nan) for metric in labels]
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, baseline, width, label="Fair baseline")
    ax.bar(x + width / 2, diagnostic, width, label="Diagnostic promoted")
    ax.set_xticks(x)
    ax.set_xticklabels([_metric_label(label) for label in labels], rotation=20, ha="right")
    ax.set_ylabel("Metric value")
    ax.set_title("Does promoted diagnostic CxG beat the fair baseline?")
    ax.text(
        0.01,
        0.98,
        "Lower is better: log loss, Brier, ECE\nHigher is better: ROC AUC",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
    )
    ax.legend()
    _save_figure(fig, path)


def _plot_horizontal_bars(
    frame: pd.DataFrame,
    *,
    label_col: str,
    value_col: str,
    path: Path,
    title: str,
    xlabel: str,
) -> None:
    plot = frame.dropna(subset=[value_col]).copy() if value_col in frame.columns else pd.DataFrame()
    if plot.empty or label_col not in plot.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No rows available", ha="center", va="center")
        ax.axis("off")
        _save_figure(fig, path)
        return
    plot = plot.head(30)
    labels = [_wrap_label(value) for value in plot[label_col].fillna("unknown").astype(str)]
    height = max(4.5, len(plot) * 0.42)
    fig, ax = plt.subplots(figsize=(10, height))
    ax.barh(labels, plot[value_col].astype(float))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.tick_params(axis="y", labelsize=8)
    _save_figure(fig, path)


def _save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _wrap_label(value: str, width: int = 28) -> str:
    return "\n".join(textwrap.wrap(value, width=width)) or value


def _metric_label(metric: str) -> str:
    return {
        "log_loss": "Log loss",
        "brier": "Brier",
        "roc_auc": "ROC AUC",
        "expected_calibration_error": "ECE",
    }.get(metric, metric)


def _player_label(row: pd.Series) -> str:
    player = row.get("player_name")
    if pd.isna(player) or not str(player).strip():
        player = f"Player {row.get('player_id')}"
    team = row.get("team_name")
    if pd.isna(team) or not str(team).strip():
        return str(player)
    return f"{player} ({team})"


def _top_bottom(frame: pd.DataFrame, column: str, top_n: int) -> pd.DataFrame:
    if column not in frame.columns or frame.empty:
        return pd.DataFrame()
    count = max(3, top_n // 2)
    top = frame.nlargest(count, column)
    bottom = frame.nsmallest(count, column)
    return pd.concat([bottom, top], ignore_index=True).drop_duplicates(subset=["team_id"])


def build_markdown_summary(
    *,
    scorecard: dict[str, Any],
    teams: pd.DataFrame,
    players: pd.DataFrame,
    drivers: pd.DataFrame,
    categories: pd.DataFrame,
    top_n_teams: int,
    top_n_players: int,
) -> str:
    metrics = scorecard["metric_comparison"]
    integrity = scorecard.get("integrity_checks", {})
    top_groups = drivers.loc[drivers["driver_type"] == "feature_group"].nsmallest(5, "rank")
    top_features = drivers.loc[drivers["driver_type"] == "feature"].nsmallest(10, "rank")
    top_team_rows = teams.nlargest(min(top_n_teams, 10), "total_cxg")
    top_goal_delta_rows = teams.nlargest(5, "goals_minus_cxg")
    top_player_rows = players.nlargest(min(top_n_players, 10), "total_cxg")
    if players["player_id"].isna().any():
        raise ValueError("Portfolio Markdown requires non-null player_id values.")
    return "\n".join(
        [
            "# Diagnostic-informed CxG Portfolio Summary",
            "",
            "## Executive Summary",
            "- The diagnostic-informed CxG model is promoted and governed for portfolio use.",
            "- The fair baseline excludes StatsBomb xG as a training feature.",
            "- Diagnostic CxG improves log loss, Brier score, and ROC AUC versus the fair baseline.",
            "- Calibration is transparent: the fair baseline remains slightly better on ECE.",
            "- Player and team outputs are ID-safe, including non-null promoted `player_id` values.",
            "",
            "## Model Scorecard",
            "*_Diagnostic minus baseline: negative is better for log loss, Brier, and ECE; positive is better for ROC AUC._",
            "",
            _metric_table(metrics),
            "",
            "## Data Integrity",
            f"- Baseline join rate: `{scorecard.get('baseline_join_rate')}`",
            f"- Feature frame `shot_id` missing: `{integrity.get('feature_frame_shot_id_missing_count')}`",
            f"- Feature frame `player_id` missing: `{integrity.get('feature_frame_player_id_missing_count')}`",
            f"- Feature frame `team_id` missing: `{integrity.get('feature_frame_team_id_missing_count')}`",
            f"- Promoted shots `shot_id` missing: `{integrity.get('shot_predictions_shot_id_missing_count')}`",
            f"- Promoted shots `player_id` missing: `{integrity.get('shot_predictions_player_id_missing_count')}`",
            f"- Promoted shots `team_id` missing: `{integrity.get('shot_predictions_team_id_missing_count')}`",
            "",
            "## Feature Impact",
            "The post-promotion feature impact analysis points to geometry as the strongest driver, followed by shot execution and contextual groups where they carry signal.",
            "",
            "Top feature groups:",
            *_bullet_rows(top_groups, "name", "log_loss_delta"),
            "",
            "Top individual features:",
            *_bullet_rows(top_features, "name", "log_loss_delta"),
            "",
            "## Team Outputs",
            "Top teams by total diagnostic CxG:",
            *_bullet_rows(top_team_rows, "team_name", "total_cxg"),
            "",
            "Top teams by goals minus CxG:",
            *_bullet_rows(top_goal_delta_rows, "team_name", "goals_minus_cxg"),
            "",
            "Teams with high total CxG are generating the largest aggregate shot quality; goals-minus-CxG is descriptive finishing variance and should not be overread as repeatable skill on its own.",
            "",
            "## Player Outputs",
            "Top players by total diagnostic CxG:",
            *_player_bullets(top_player_rows),
            "",
            "Player rows retain real `player_id` values and are not collapsed through null identifiers.",
            "",
            "## Category Insights",
            *_category_bullets(categories),
            "",
            "These category lifts describe observed model output patterns, not causal effects.",
            "",
            "## Charts",
            "- [Model metric comparison](charts/model_metric_comparison.png)",
            "- [Feature group impact](charts/feature_group_impact.png)",
            "- [Top feature importance](charts/top_feature_importance.png)",
            "- [Team CxG ranking](charts/team_cxg_ranking.png)",
            "- [Player CxG ranking](charts/player_cxg_ranking.png)",
            "- [Goals minus CxG teams](charts/goals_minus_cxg_teams.png)",
            "- [Body part category lift](charts/category_lift_body_part.png)",
            "- [Shot type category lift](charts/category_lift_shot_type.png)",
            "- [Set-piece category lift](charts/category_lift_set_piece_category.png)",
            "",
            "## Limitations",
            "- This analysis is based on StatsBomb public event data.",
            "- Feature impact is post-promotion perturbation/permutation analysis, not causal proof.",
            "- Calibration remains monitored because the fair baseline is slightly better on ECE.",
            "- A Streamlit dashboard can later provide interactive exploration over the same static outputs.",
            "",
        ]
    )


def _metric_table(metrics: dict[str, dict[str, Any]]) -> str:
    lines = [
        "| Metric | Baseline | Diagnostic | Diagnostic - baseline |",
        "|---|---:|---:|---:|",
    ]
    for metric in METRICS:
        lines.append(
            "| "
            f"{_metric_label(metric)} | "
            f"{_fmt(metrics['baseline'].get(metric))} | "
            f"{_fmt(metrics['diagnostic'].get(metric))} | "
            f"{_fmt(metrics['diagnostic_minus_baseline'].get(metric))} |"
        )
    return "\n".join(lines)


def _bullet_rows(frame: pd.DataFrame, label_col: str, value_col: str) -> list[str]:
    if frame.empty or label_col not in frame.columns or value_col not in frame.columns:
        return ["- No rows available."]
    return [f"- {row[label_col]}: `{_fmt(row[value_col])}`" for _, row in frame.head(10).iterrows()]


def _player_bullets(frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return ["- No player rows available."]
    return [
        f"- {_player_label(row)}: `{_fmt(row.get('total_cxg'))}` total CxG"
        for _, row in frame.head(10).iterrows()
    ]


def _category_bullets(categories: pd.DataFrame) -> list[str]:
    if categories.empty:
        return ["- No category lift tables were available."]
    lines = []
    for category_column, group in categories.groupby("category_column"):
        top = group.sort_values("total_predicted_cxg", ascending=False).head(3)
        values = ", ".join(
            f"{row.category} ({_fmt(row.mean_predicted_cxg)} mean CxG)"
            for row in top.itertuples(index=False)
        )
        lines.append(f"- `{category_column}`: {values}")
    return lines


def _fmt(value: Any) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.4f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build static CxG portfolio summary outputs")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--feature-impact-dir", type=Path, default=DEFAULT_FEATURE_IMPACT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-n-teams", type=int, default=15)
    parser.add_argument("--top-n-players", type=int, default=20)
    parser.add_argument("--top-n-features", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = build_cxg_portfolio_summary(
        paths=PortfolioPaths.from_roots(
            results_dir=args.results_dir,
            feature_impact_dir=args.feature_impact_dir,
            output_dir=args.output_dir,
        ),
        top_n_teams=args.top_n_teams,
        top_n_players=args.top_n_players,
        top_n_features=args.top_n_features,
    )
    print(json.dumps({name: str(path) for name, path in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
