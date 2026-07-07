#!/usr/bin/env python
"""Build static portfolio-ready CxA summary artifacts and charts."""

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

DEFAULT_RESULTS_DIR = Path("outputs/results/cxa/diagnostic_v1")
DEFAULT_VALIDATION_DIR = Path("outputs/validation/cxa/diagnostic_v1")
DEFAULT_FEATURE_IMPACT_DIR = Path("outputs/modeling/cxa/diagnostic_v1/feature_impact")
DEFAULT_OUTPUT_DIR = Path("outputs/portfolio/cxa")
MODEL_METRICS = (
    "log_loss",
    "brier",
    "roc_auc",
    "average_precision",
    "expected_calibration_error",
    "precision_at_top_1pct",
    "precision_at_top_5pct",
)
REQUIRED_CHARTS = {
    "top_players_by_cxa": "top_players_by_cxa.png",
    "top_teams_by_cxa": "top_teams_by_cxa.png",
    "feature_group_impact": "feature_group_impact.png",
    "baseline_vs_diagnostic_metrics": "baseline_vs_diagnostic_metrics.png",
    "prediction_distribution": "prediction_distribution.png",
}
PLAYER_COLUMNS = [
    "player_id",
    "team_id",
    "actions",
    "shot_creating_actions",
    "total_diagnostic_cxa",
    "mean_diagnostic_cxa",
    "max_diagnostic_cxa",
    "rank",
]
TEAM_COLUMNS = [
    "team_id",
    "actions",
    "shot_creating_actions",
    "total_diagnostic_cxa",
    "mean_diagnostic_cxa",
    "max_diagnostic_cxa",
    "rank",
]
SEQUENCE_COLUMNS = [
    "sequence_id",
    "match_id",
    "team_id",
    "possession",
    "actions",
    "shot_creating_actions",
    "total_diagnostic_cxa",
    "max_diagnostic_cxa",
    "mean_diagnostic_cxa",
    "sequence_led_to_shot",
    "rank",
]
FORBIDDEN_MODEL_FEATURES = {"created_shot_cxg", "cxa_value", "created_shot_id"}


@dataclass(frozen=True)
class CxAPortfolioPaths:
    results_dir: Path
    validation_dir: Path
    feature_impact_dir: Path
    output_dir: Path

    @property
    def charts_dir(self) -> Path:
        return self.output_dir / "charts"

    @property
    def action_predictions(self) -> Path:
        return self.results_dir / "action_predictions.parquet"

    @property
    def player_summary(self) -> Path:
        return self.results_dir / "player_cxa_summary.csv"

    @property
    def team_summary(self) -> Path:
        return self.results_dir / "team_cxa_summary.csv"

    @property
    def sequence_summary(self) -> Path:
        return self.results_dir / "sequence_cxa_summary.csv"

    @property
    def promotion_summary(self) -> Path:
        return self.results_dir / "model_promotion_summary.json"

    @property
    def quality_checks(self) -> Path:
        return self.results_dir / "prediction_quality_checks.csv"

    @property
    def baseline_metrics(self) -> Path:
        return self.validation_dir / "baseline_vs_diagnostic_metrics.csv"

    @property
    def feature_impact_summary_csv(self) -> Path:
        return self.feature_impact_dir / "feature_impact_summary.csv"

    @property
    def feature_group_impact(self) -> Path:
        return self.feature_impact_dir / "feature_group_impact.csv"

    @property
    def feature_impact_summary_json(self) -> Path:
        return self.feature_impact_dir / "feature_impact_summary.json"

    @classmethod
    def from_roots(
        cls,
        results_dir: Path = DEFAULT_RESULTS_DIR,
        validation_dir: Path = DEFAULT_VALIDATION_DIR,
        feature_impact_dir: Path = DEFAULT_FEATURE_IMPACT_DIR,
        output_dir: Path = DEFAULT_OUTPUT_DIR,
    ) -> "CxAPortfolioPaths":
        return cls(
            results_dir=results_dir,
            validation_dir=validation_dir,
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
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if np.isnan(value) else float(value)
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    if pd.isna(value) and not isinstance(value, bool | str):
        return None
    return value


def require_inputs(paths: CxAPortfolioPaths) -> None:
    required = {
        "model_promotion_summary": paths.promotion_summary,
        "action_predictions": paths.action_predictions,
        "player_cxa_summary": paths.player_summary,
        "team_cxa_summary": paths.team_summary,
        "sequence_cxa_summary": paths.sequence_summary,
        "baseline_vs_diagnostic_metrics": paths.baseline_metrics,
        "feature_impact_summary_csv": paths.feature_impact_summary_csv,
        "feature_group_impact": paths.feature_group_impact,
        "feature_impact_summary_json": paths.feature_impact_summary_json,
    }
    missing = [f"{name}: {path}" for name, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Cannot build CxA portfolio summary because required inputs are missing: "
            + "; ".join(missing)
        )


def validate_promotion(promotion: dict[str, Any]) -> None:
    if promotion.get("promotion_status") == "blocked" or not promotion.get("promotion_gate_passed"):
        raise ValueError("Cannot build CxA portfolio summary for a blocked promotion gate")
    forbidden = set(
        promotion.get("governance_summary", {}).get("selected_features", [])
    ).intersection(FORBIDDEN_MODEL_FEATURES)
    if forbidden:
        raise ValueError(
            f"Forbidden reference/output columns appear as model features: {forbidden}"
        )


def _ensure_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        if column not in frame.columns:
            frame[column] = np.nan
    return frame[columns]


def entity_rankings(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    ranked = _ensure_columns(frame.copy(), columns)
    return ranked.sort_values("total_diagnostic_cxa", ascending=False, na_position="last")


def metric_comparison(metrics: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if not {"metric", "baseline", "diagnostic", "diagnostic_minus_baseline"}.issubset(
        metrics.columns
    ):
        raise ValueError("baseline_vs_diagnostic_metrics.csv is missing required metric columns")
    indexed = metrics.set_index("metric")
    missing = [metric for metric in MODEL_METRICS if metric not in indexed.index]
    if missing:
        raise ValueError(f"Required baseline-vs-diagnostic metrics are unavailable: {missing}")
    return {
        "baseline": {
            metric: _maybe_float(indexed.loc[metric, "baseline"]) for metric in MODEL_METRICS
        },
        "diagnostic": {
            metric: _maybe_float(indexed.loc[metric, "diagnostic"]) for metric in MODEL_METRICS
        },
        "diagnostic_minus_baseline": {
            metric: _maybe_float(indexed.loc[metric, "diagnostic_minus_baseline"])
            for metric in MODEL_METRICS
        },
    }


def _maybe_float(value: Any) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def feature_driver_summary(
    feature_impact: pd.DataFrame,
    group_impact: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    if not group_impact.empty:
        groups = group_impact.copy()
        groups["driver_type"] = "feature_group"
        groups["name"] = groups["feature_group"]
        groups["rank"] = groups["impact"].rank(method="first", ascending=False).astype(int)
        rows.append(
            _ensure_columns(
                groups,
                [
                    "driver_type",
                    "name",
                    "feature_group",
                    "impact",
                    "mean_probability_shift",
                    "rank",
                ],
            )
        )
    if not feature_impact.empty:
        features = feature_impact.copy()
        features = features.loc[~features["feature_name"].isin(FORBIDDEN_MODEL_FEATURES)].copy()
        features["driver_type"] = "feature"
        features["name"] = features["feature_name"]
        rows.append(
            _ensure_columns(
                features,
                [
                    "driver_type",
                    "name",
                    "feature_group",
                    "impact",
                    "mean_probability_shift",
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
                "impact",
                "mean_probability_shift",
                "rank",
            ]
        )
    return pd.concat(rows, ignore_index=True).sort_values(
        ["driver_type", "rank"], ascending=[True, True]
    )


def build_headline_metrics(
    *,
    promotion: dict[str, Any],
    impact_summary: dict[str, Any],
    actions: pd.DataFrame,
    comparison: dict[str, dict[str, Any]],
    drivers: pd.DataFrame,
) -> dict[str, Any]:
    probability = pd.to_numeric(actions["predicted_shot_created_probability"], errors="coerce")
    top_feature = _top_driver(drivers, "feature")
    top_group = _top_driver(drivers, "feature_group")
    return {
        "action_row_count": int(len(actions)),
        "total_diagnostic_cxa": float(actions["diagnostic_cxa"].sum()),
        "mean_predicted_probability": float(probability.mean()),
        "probability_min": float(probability.min()),
        "probability_max": float(probability.max()),
        "selected_model": promotion.get("selected_model_candidate"),
        "promotion_status": promotion.get("promotion_status"),
        "promotion_gate_passed": bool(promotion.get("promotion_gate_passed")),
        "selected_feature_count": impact_summary.get(
            "selected_feature_count",
            promotion.get("governance_summary", {}).get("selected_feature_count"),
        ),
        "top_feature_driver": top_feature,
        "top_feature_group_driver": top_group,
        "baseline_vs_diagnostic": comparison,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def _top_driver(drivers: pd.DataFrame, driver_type: str) -> dict[str, Any] | None:
    subset = drivers.loc[drivers["driver_type"] == driver_type]
    if subset.empty:
        return None
    row = subset.sort_values("rank").iloc[0]
    return {
        "name": row.get("name"),
        "feature_group": row.get("feature_group"),
        "impact": _maybe_float(row.get("impact")),
        "mean_probability_shift": _maybe_float(row.get("mean_probability_shift")),
    }


def build_cxa_portfolio_summary(
    *,
    paths: CxAPortfolioPaths | None = None,
    top_n_players: int = 20,
    top_n_teams: int = 15,
    top_n_sequences: int = 20,
) -> dict[str, Path]:
    paths = paths or CxAPortfolioPaths.from_roots()
    require_inputs(paths)
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    paths.charts_dir.mkdir(parents=True, exist_ok=True)

    promotion = _read_json(paths.promotion_summary)
    validate_promotion(promotion)
    impact_json = _read_json(paths.feature_impact_summary_json)
    actions = pd.read_parquet(paths.action_predictions)
    players = entity_rankings(pd.read_csv(paths.player_summary), PLAYER_COLUMNS)
    teams = entity_rankings(pd.read_csv(paths.team_summary), TEAM_COLUMNS)
    sequences = entity_rankings(pd.read_csv(paths.sequence_summary), SEQUENCE_COLUMNS)
    baseline_metrics = pd.read_csv(paths.baseline_metrics)
    comparison = metric_comparison(baseline_metrics)
    feature_impact = pd.read_csv(paths.feature_impact_summary_csv)
    group_impact = pd.read_csv(paths.feature_group_impact)
    drivers = feature_driver_summary(feature_impact, group_impact)

    outputs = {
        "summary_md": paths.output_dir / "portfolio_summary.md",
        "headline_metrics_json": paths.output_dir / "headline_metrics.json",
        "top_players_csv": paths.output_dir / "top_players_by_cxa.csv",
        "top_teams_csv": paths.output_dir / "top_teams_by_cxa.csv",
        "top_sequences_csv": paths.output_dir / "top_sequences_by_cxa.csv",
        "feature_driver_summary_csv": paths.output_dir / "feature_driver_summary.csv",
    }
    charts = {name: paths.charts_dir / filename for name, filename in REQUIRED_CHARTS.items()}

    players.head(top_n_players).to_csv(outputs["top_players_csv"], index=False)
    teams.head(top_n_teams).to_csv(outputs["top_teams_csv"], index=False)
    sequences.head(top_n_sequences).to_csv(outputs["top_sequences_csv"], index=False)
    drivers.to_csv(outputs["feature_driver_summary_csv"], index=False)

    headline = build_headline_metrics(
        promotion=promotion,
        impact_summary=impact_json,
        actions=actions,
        comparison=comparison,
        drivers=drivers,
    )
    _write_json(outputs["headline_metrics_json"], headline)
    create_charts(
        actions=actions,
        players=players,
        teams=teams,
        drivers=drivers,
        comparison=comparison,
        charts=charts,
        top_n_players=top_n_players,
        top_n_teams=top_n_teams,
    )
    outputs["summary_md"].write_text(
        build_markdown_summary(
            headline=headline,
            promotion=promotion,
            players=players,
            teams=teams,
            sequences=sequences,
            drivers=drivers,
            top_n_players=top_n_players,
            top_n_teams=top_n_teams,
            top_n_sequences=top_n_sequences,
        ),
        encoding="utf-8",
    )
    return {**outputs, **{f"chart_{key}": value for key, value in charts.items()}}


def create_charts(
    *,
    actions: pd.DataFrame,
    players: pd.DataFrame,
    teams: pd.DataFrame,
    drivers: pd.DataFrame,
    comparison: dict[str, dict[str, Any]],
    charts: dict[str, Path],
    top_n_players: int,
    top_n_teams: int,
) -> None:
    _plot_horizontal_bars(
        players.head(top_n_players),
        label_col="player_id",
        value_col="total_diagnostic_cxa",
        path=charts["top_players_by_cxa"],
        title=f"Top {top_n_players} players by diagnostic CxA",
        xlabel="Total diagnostic CxA",
    )
    _plot_horizontal_bars(
        teams.head(top_n_teams),
        label_col="team_id",
        value_col="total_diagnostic_cxa",
        path=charts["top_teams_by_cxa"],
        title=f"Top {top_n_teams} teams by diagnostic CxA",
        xlabel="Total diagnostic CxA",
    )
    group_rows = drivers.loc[drivers["driver_type"] == "feature_group"].sort_values(
        "impact", ascending=False
    )
    _plot_horizontal_bars(
        group_rows,
        label_col="name",
        value_col="impact",
        path=charts["feature_group_impact"],
        title="Which feature groups move CxA log loss most?",
        xlabel="Permutation impact on log loss",
    )
    _plot_metric_comparison(comparison, charts["baseline_vs_diagnostic_metrics"])
    _plot_prediction_distribution(actions, charts["prediction_distribution"])


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
    labels = [_wrap_label(str(value)) for value in plot[label_col].fillna("unknown")]
    fig, ax = plt.subplots(figsize=(10, max(4.5, len(plot) * 0.42)))
    ax.barh(labels, plot[value_col].astype(float))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.tick_params(axis="y", labelsize=8)
    _save_figure(fig, path)


def _plot_metric_comparison(metrics: dict[str, dict[str, Any]], path: Path) -> None:
    labels = list(MODEL_METRICS)
    baseline = [metrics["baseline"].get(metric, np.nan) for metric in labels]
    diagnostic = [metrics["diagnostic"].get(metric, np.nan) for metric in labels]
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - width / 2, baseline, width, label="Baseline reference")
    ax.bar(x + width / 2, diagnostic, width, label="Diagnostic CxA")
    ax.set_xticks(x)
    ax.set_xticklabels([_metric_label(label) for label in labels], rotation=25, ha="right")
    ax.set_title("How does diagnostic CxA compare with the reference baseline?")
    ax.set_ylabel("Metric value")
    ax.text(
        0.01,
        0.98,
        "Lower is better: log loss, Brier, ECE\nHigher is better: ROC AUC, AP, top-k precision",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
    )
    ax.legend()
    _save_figure(fig, path)


def _plot_prediction_distribution(actions: pd.DataFrame, path: Path) -> None:
    probability = pd.to_numeric(
        actions["predicted_shot_created_probability"], errors="coerce"
    ).dropna()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(probability, bins=40)
    ax.set_title("Distribution of diagnostic CxA shot-creation probabilities")
    ax.set_xlabel("Predicted shot-created probability")
    ax.set_ylabel("Action count")
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
        "average_precision": "Average precision",
        "expected_calibration_error": "ECE",
        "precision_at_top_1pct": "Top 1% precision",
        "precision_at_top_5pct": "Top 5% precision",
    }.get(metric, metric)


def build_markdown_summary(
    *,
    headline: dict[str, Any],
    promotion: dict[str, Any],
    players: pd.DataFrame,
    teams: pd.DataFrame,
    sequences: pd.DataFrame,
    drivers: pd.DataFrame,
    top_n_players: int,
    top_n_teams: int,
    top_n_sequences: int,
) -> str:
    top_features = drivers.loc[drivers["driver_type"] == "feature"].nsmallest(10, "rank")
    top_groups = drivers.loc[drivers["driver_type"] == "feature_group"].nsmallest(5, "rank")
    return "\n".join(
        [
            "# Diagnostic CxA Portfolio Summary",
            "",
            "## Executive summary",
            "- Diagnostic CxA estimates the probability that an action creates a shot.",
            f"- Current status is `{headline['promotion_status']}`.",
            "- The baseline comparison is reference-only/in-sample, so promotion is provisional.",
            "- `created_shot_cxg` and `cxa_value` are not model features.",
            "- CxA+ and Advanced CxA come later.",
            "",
            "## What CxA measures",
            "`diagnostic_cxa` is the governed model probability that an action creates a shot. It is not yet a downstream shot-value attribution model.",
            "",
            "## Promotion status and governance",
            f"- Selected model: `{headline['selected_model']}`",
            f"- Promotion gate passed: `{headline['promotion_gate_passed']}`",
            f"- Baseline is fair comparator: `{promotion.get('baseline_is_fair_comparator')}`",
            f"- Strict promotion comparison enabled: `{promotion.get('strict_promotion_comparison_enabled')}`",
            "",
            "## Baseline vs diagnostic improvement",
            _metric_table(headline["baseline_vs_diagnostic"]),
            "",
            "## Headline output scale",
            f"- Action row count: `{headline['action_row_count']}`",
            f"- Total diagnostic CxA: `{headline['total_diagnostic_cxa']:.3f}`",
            f"- Mean predicted probability: `{headline['mean_predicted_probability']:.6f}`",
            f"- Probability range: `{headline['probability_min']:.6f}` to `{headline['probability_max']:.6f}`",
            f"- Selected feature count: `{headline.get('selected_feature_count')}`",
            "",
            "## Top players",
            *_bullet_rows(players.head(top_n_players), "player_id", "total_diagnostic_cxa"),
            "",
            "## Top teams",
            *_bullet_rows(teams.head(top_n_teams), "team_id", "total_diagnostic_cxa"),
            "",
            "## Top sequences",
            *_bullet_rows(sequences.head(top_n_sequences), "sequence_id", "total_diagnostic_cxa"),
            "",
            "## Feature impact interpretation",
            "Feature impact is computed from existing promoted artifacts and explains which inputs most change diagnostic CxA probability quality.",
            "",
            "Top feature groups:",
            *_bullet_rows(top_groups, "name", "impact"),
            "",
            "Top individual features:",
            *_bullet_rows(top_features, "name", "impact"),
            "",
            "## Caveats",
            "- The model is provisionally promoted because the current baseline predictions are full-data/in-sample.",
            "- Feature impact is permutation-style reporting, not causal proof.",
            "- `created_shot_cxg`, `created_shot_id`, and `cxa_value` remain reference/output fields, not model features.",
            "",
            "## Next recommended PR",
            "- Add dashboard/docs views over the static CxA portfolio pack before implementing CxA+ or Advanced CxA.",
            "",
            "## Charts",
            "- [Top players by CxA](charts/top_players_by_cxa.png)",
            "- [Top teams by CxA](charts/top_teams_by_cxa.png)",
            "- [Feature group impact](charts/feature_group_impact.png)",
            "- [Baseline vs diagnostic metrics](charts/baseline_vs_diagnostic_metrics.png)",
            "- [Prediction distribution](charts/prediction_distribution.png)",
            "",
        ]
    )


def _metric_table(metrics: dict[str, dict[str, Any]]) -> str:
    lines = [
        "| Metric | Baseline | Diagnostic | Diagnostic - baseline |",
        "|---|---:|---:|---:|",
    ]
    for metric in MODEL_METRICS:
        lines.append(
            "| "
            + " | ".join(
                [
                    _metric_label(metric),
                    _fmt(metrics["baseline"].get(metric)),
                    _fmt(metrics["diagnostic"].get(metric)),
                    _fmt(metrics["diagnostic_minus_baseline"].get(metric)),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.6f}"


def _bullet_rows(frame: pd.DataFrame, label_col: str, value_col: str) -> list[str]:
    if frame.empty or label_col not in frame.columns or value_col not in frame.columns:
        return ["- No rows available."]
    rows = []
    for _, row in frame.iterrows():
        rows.append(f"- `{row[label_col]}`: {_fmt(row[value_col])}")
    return rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--validation-dir", type=Path, default=DEFAULT_VALIDATION_DIR)
    parser.add_argument("--feature-impact-dir", type=Path, default=DEFAULT_FEATURE_IMPACT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-n-players", type=int, default=20)
    parser.add_argument("--top-n-teams", type=int, default=15)
    parser.add_argument("--top-n-sequences", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    outputs = build_cxa_portfolio_summary(
        paths=CxAPortfolioPaths.from_roots(
            results_dir=args.results_dir,
            validation_dir=args.validation_dir,
            feature_impact_dir=args.feature_impact_dir,
            output_dir=args.output_dir,
        ),
        top_n_players=args.top_n_players,
        top_n_teams=args.top_n_teams,
        top_n_sequences=args.top_n_sequences,
    )
    print("Generated CxA portfolio outputs:")
    for name, path in outputs.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
