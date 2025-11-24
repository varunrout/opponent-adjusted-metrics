"""Plot reliability curves for multiple CxG variants on one chart."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from opponent_adjusted.modeling.cxg import baseline_geometry, contextual_model
from opponent_adjusted.modeling.modeling_utilities import (
    get_modeling_output_dir,
    load_cxg_modeling_dataset,
    load_dataset_from_versions,
)
from opponent_adjusted.modeling.plot_utilities import configure_modeling_style


CONTEXTUAL_VARIANTS = {
    "filtered": contextual_model.DATASET_FILE_MAP["filtered"],
    "raw": contextual_model.DATASET_FILE_MAP["raw"],
    "enriched": contextual_model.DATASET_FILE_MAP["enriched"],
}

PROVIDER_PREFERENCE: list[list[str]] = [
    contextual_model.DATASET_FILE_MAP["enriched"],
    contextual_model.DATASET_FILE_MAP["filtered"],
    contextual_model.DATASET_FILE_MAP["raw"],
]

DISPLAY_NAMES = {
    "provider": "StatsBomb Provider xG",
    "baseline": "Baseline Geometry",
    "contextual_filtered": "Contextual (Filtered)",
    "contextual_raw": "Contextual (Unfiltered)",
    "contextual_enriched": "Contextual (Enriched Priors)",
}

LINE_COLORS = {
    "provider": "#6C757D",
    "baseline": "#219EBC",
    "contextual_filtered": "#FFB703",
    "contextual_raw": "#8ECAE6",
    "contextual_enriched": "#023047",
}


@dataclass
class ReliabilityCurve:
    label: str
    display_name: str
    color: str
    centers: np.ndarray
    observed: np.ndarray
    counts: np.ndarray

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "model": self.label,
                "model_display": self.display_name,
                "bin_center": self.centers,
                "empirical_goal_rate": self.observed,
                "sample_count": self.counts,
            }
        )


def _compute_reliability(y_true: np.ndarray, y_pred: np.ndarray, bins: int = 10) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    clipped = np.clip(y_pred, 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ids = np.digitize(clipped, edges) - 1
    centers = 0.5 * (edges[:-1] + edges[1:])

    observed: list[float] = []
    counts: list[int] = []
    for idx in range(bins):
        mask = ids == idx
        if not np.any(mask):
            observed.append(np.nan)
            counts.append(0)
            continue
        observed.append(float(y_true[mask].mean()))
        counts.append(int(mask.sum()))
    return centers, np.array(observed), np.array(counts)


def _run_baseline_curve() -> ReliabilityCurve:
    df = baseline_geometry._load_dataset()
    _, y_true, y_pred = baseline_geometry._train_and_evaluate(df)
    centers, observed, counts = _compute_reliability(y_true, y_pred)
    return ReliabilityCurve("baseline", DISPLAY_NAMES["baseline"], LINE_COLORS["baseline"], centers, observed, counts)


def _run_contextual_curve(dataset_key: str, label: str) -> ReliabilityCurve:
    files = CONTEXTUAL_VARIANTS[dataset_key]
    df_raw = load_dataset_from_versions("cxg", files)
    df = contextual_model._prepare_frame(df_raw)
    numeric, binary, categorical = contextual_model._filter_features(df)
    _, y_true, y_pred = contextual_model._train_and_eval(
        df,
        numeric,
        binary,
        categorical,
    )
    centers, observed, counts = _compute_reliability(y_true, y_pred)
    return ReliabilityCurve(label, DISPLAY_NAMES[label], LINE_COLORS[label], centers, observed, counts)


def _run_provider_curve() -> ReliabilityCurve:
    df: pd.DataFrame | None = None
    for filenames in PROVIDER_PREFERENCE:
        try:
            df = load_dataset_from_versions("cxg", filenames)
            break
        except FileNotFoundError:
            continue
    if df is None:
        df = load_cxg_modeling_dataset(prefer_filtered=True)
    subset = df.dropna(subset=["is_goal", "statsbomb_xg"]).copy()
    y_true = subset["is_goal"].astype(int).to_numpy()
    y_pred = subset["statsbomb_xg"].astype(float).to_numpy()
    centers, observed, counts = _compute_reliability(y_true, y_pred)
    return ReliabilityCurve("provider", DISPLAY_NAMES["provider"], LINE_COLORS["provider"], centers, observed, counts)


def _plot_curves(curves: Iterable[ReliabilityCurve], out_path: Path) -> None:
    configure_modeling_style()
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#888888", label="Ideal")

    for curve in curves:
        ax.plot(
            curve.centers,
            curve.observed,
            marker="o",
            label=curve.display_name,
            color=curve.color,
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted probability bin (center)")
    ax.set_ylabel("Empirical goal rate")
    ax.set_title("CxG Reliability Comparison")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    curves: list[ReliabilityCurve] = []
    curves.append(_run_provider_curve())
    curves.append(_run_baseline_curve())
    curves.append(_run_contextual_curve("filtered", "contextual_filtered"))
    curves.append(_run_contextual_curve("raw", "contextual_raw"))
    curves.append(_run_contextual_curve("enriched", "contextual_enriched"))

    charts_dir = get_modeling_output_dir("cxg", subdir="modeling_charts")
    plot_path = charts_dir / "cxg_reliability_overlay.png"
    _plot_curves(curves, plot_path)

    combined = pd.concat([curve.to_frame() for curve in curves], ignore_index=True)
    combined.to_csv(charts_dir / "cxg_reliability_overlay.csv", index=False)
    print(f"Saved reliability overlay chart to {plot_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
