"""Generate comparison charts for baseline vs contextual CxG models."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from opponent_adjusted.modeling.modeling_utilities import (
    get_modeling_output_dir,
    load_cxg_modeling_dataset,
    load_dataset_from_versions,
)
from opponent_adjusted.modeling.plot_utilities import (
    annotate_bars,
    configure_modeling_style,
    format_metric_name,
    load_metrics_json,
    metrics_map_to_frame,
)

MODEL_SPECS = [
    ("baseline", "baseline_metrics.json"),
    ("contextual_prev", "contextual_metrics_filtered.json"),
    ("contextual_enriched", "contextual_metrics_enriched.json"),
]
MODEL_DISPLAY = {
    "provider_xg": "StatsBomb Provider xG",
    "baseline": "Baseline Geometry",
    "contextual_prev": "Contextual (Filtered)",
    "contextual_enriched": "Contextual (Enriched Priors)",
}
MODEL_COLORS = {
    "provider_xg": "#6C757D",
    "baseline": "#219EBC",
    "contextual_prev": "#FFB703",
    "contextual_enriched": "#023047",
}
PROVIDER_DATASET_CANDIDATES = [
    ["cxg_dataset_enriched.parquet", "cxg_dataset_enriched.csv"],
    ["cxg_dataset_filtered.parquet", "cxg_dataset_filtered.csv"],
    ["cxg_dataset.parquet", "cxg_dataset.csv"],
]
METRIC_KEYS = ["brier_mean", "log_loss_mean", "auc_mean"]


def _load_all_metrics(base_dir: Path) -> dict[str, dict]:
    metrics: dict[str, dict] = {}
    missing: list[str] = []
    for label, filename in MODEL_SPECS:
        path = base_dir / filename
        if not path.exists():
            missing.append(str(path))
            continue
        metrics[label] = load_metrics_json(path)
    if missing:
        raise FileNotFoundError(
            "Missing metrics files: " + ", ".join(missing)
        )
    return metrics


def _load_provider_dataset() -> pd.DataFrame:
    for filenames in PROVIDER_DATASET_CANDIDATES:
        try:
            return load_dataset_from_versions("cxg", filenames)
        except FileNotFoundError:
            continue
    return load_cxg_modeling_dataset(prefer_filtered=True)


def _compute_provider_metrics() -> dict:
    df = _load_provider_dataset()
    subset = df.dropna(subset=["is_goal", "statsbomb_xg"]).copy()
    if subset.empty:
        raise ValueError("statsbomb_xg column is missing or empty in dataset")
    y_true = subset["is_goal"].astype(int).to_numpy()
    y_pred = subset["statsbomb_xg"].astype(float).clip(0.0, 1.0).to_numpy()
    return {
        "brier_mean": float(brier_score_loss(y_true, y_pred)),
        "log_loss_mean": float(log_loss(y_true, y_pred)),
        "auc_mean": float(roc_auc_score(y_true, y_pred)),
    }


def _plot_metric(df: pd.DataFrame, metric: str, out_dir: Path) -> None:
    subset = df[df["metric"] == metric].copy()
    if subset.empty:
        return
    ascending = metric != "auc_mean"
    subset = subset.sort_values("value", ascending=ascending)

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    colors = [MODEL_COLORS.get(model, "#888888") for model in subset["model"]]
    ax.bar(subset["model_display"], subset["value"], color=colors)
    ax.set_ylabel(format_metric_name(metric))
    ax.set_xlabel("Model", fontsize=12)
    ax.tick_params(axis="x", labelsize=7)
    title = f"CxG {format_metric_name(metric)} Comparison"
    ax.set_title(title)
    if metric != "auc_mean":
        ax.set_ylim(bottom=0)
    annotate_bars(ax)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"model_compare_{metric}.png", dpi=150)
    plt.close(fig)


def main() -> None:
    configure_modeling_style()
    base_dir = get_modeling_output_dir("cxg")
    metrics_map = _load_all_metrics(base_dir)
    metrics_map["provider_xg"] = _compute_provider_metrics()
    df = metrics_map_to_frame(metrics_map, METRIC_KEYS)
    df["model_display"] = df["model"].map(MODEL_DISPLAY).fillna(df["model"])

    charts_dir = get_modeling_output_dir("cxg", subdir="modeling_charts")
    for metric in METRIC_KEYS:
        _plot_metric(df, metric, charts_dir)

    table = df.pivot(index="model_display", columns="metric", values="value")
    table.to_csv(charts_dir / "model_metric_table.csv")
    print(f"Saved comparison charts to {charts_dir}")


if __name__ == "__main__":  # pragma: no cover
    main()
