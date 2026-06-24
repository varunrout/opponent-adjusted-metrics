#!/usr/bin/env python
"""cXA Phase 2: xA Baseline analysis (logistic regression on all passes).

Purpose
-------
Analyze the calibrated logistic regression model that estimates the probability
any pass becomes an assist (xA Baseline). Outputs include player leaderboards,
calibration curves, feature weights, and spatial heatmaps of xA density.

Outputs (outputs/analysis/cxa/phase2_xa_baseline/)
-------
data/
  - player_leaderboard.csv
  - calibration_curve.csv
  - feature_weights.csv
  - spatial_heatmap_bins.csv
  - summary_metrics.csv
plots/
  - calibration_curve.png
  - xa_vs_assists_scatter.png
  - xa_probability_histogram.png
  - xa_spatial_heatmap.png
phase2_xa_baseline_report.md

Usage
-----
    PYTHONPATH=src python -m opponent_adjusted.analysis.cxa.phase2_xa_baseline
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from opponent_adjusted.features.cxa.xa_baseline import (
    BASELINE_FEATURES,
    compute_xa_baseline,
)

logger = logging.getLogger(__name__)


def _get_repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _load_passes(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing parquet: {path}")
    df = pd.read_parquet(path)
    logger.info(f"Loaded passes: {len(df):,} rows from {path}")
    return df


def _ensure_assist_flag(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "is_assist" not in out.columns:
        out["is_assist"] = (out["is_key_pass"].fillna(False).astype(bool)) & (
            out["sequence_resulted_goal"].fillna(False).astype(bool)
        )
    return out


def _calc_calibration_curve(
    df: pd.DataFrame,
    prob_col: str = "xa_baseline",
    true_col: str = "is_assist",
    n_bins: int = 20,
) -> pd.DataFrame:
    edges = np.linspace(0, df[prob_col].max(), n_bins + 1)
    df = df.copy()
    df["prob_bin"] = pd.cut(df[prob_col], bins=edges, include_lowest=True)

    grouped = (
        df.groupby("prob_bin", observed=False)
        .agg(
            mean_pred=(prob_col, "mean"),
            actual_rate=(true_col, "mean"),
            count=(prob_col, "size"),
        )
        .reset_index()
    )

    # Midpoint of each bin for plotting
    grouped["bin_mid"] = grouped["prob_bin"].apply(
        lambda b: (b.left + b.right) / 2 if pd.notnull(b) else np.nan
    )
    return grouped


def _player_leaderboard(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["player_id", "passer_name", "team_name"])
        .agg(
            passes=("pass_id", "count"),
            key_passes=("is_key_pass", "sum"),
            assists=("is_assist", "sum"),
            xa_baseline=("xa_baseline", "sum"),
            xa_mean=("xa_baseline", "mean"),
        )
        .reset_index()
    )

    grouped["assist_rate_per_pass"] = grouped["assists"] / grouped["passes"].replace(0, np.nan)
    grouped["assist_rate_per_key_pass"] = grouped["assists"] / grouped["key_passes"].replace(
        0, np.nan
    )
    grouped["xa_per_key_pass"] = grouped["xa_baseline"] / grouped["key_passes"].replace(0, np.nan)

    grouped = grouped.sort_values("xa_baseline", ascending=False)
    return grouped


def _spatial_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    x_bins = np.linspace(0, 120, 13)
    y_bins = np.linspace(0, 80, 9)
    df = df.copy()
    df["x_bin"] = pd.cut(df["end_x"], bins=x_bins, labels=False)
    df["y_bin"] = pd.cut(df["end_y"], bins=y_bins, labels=False)

    heatmap = (
        df.groupby(["x_bin", "y_bin"])
        .agg(
            xa_sum=("xa_baseline", "sum"),
            count=("xa_baseline", "size"),
        )
        .reset_index()
    )

    # Bin centers for reference
    heatmap["x_center"] = heatmap["x_bin"].apply(
        lambda i: (x_bins[i] + x_bins[i + 1]) / 2 if pd.notnull(i) else np.nan
    )
    heatmap["y_center"] = heatmap["y_bin"].apply(
        lambda i: (y_bins[i] + y_bins[i + 1]) / 2 if pd.notnull(i) else np.nan
    )
    return heatmap


def _plot_calibration(df: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(8, 6))
    plt.plot(df["bin_mid"], df["actual_rate"], label="Actual", marker="o")
    plt.plot(df["bin_mid"], df["mean_pred"], label="Predicted", marker="o")
    plt.plot([0, df["bin_mid"].max()], [0, df["bin_mid"].max()], "k--", alpha=0.4, label="Perfect")
    plt.xlabel("Predicted assist probability")
    plt.ylabel("Actual assist rate")
    plt.title("xA Baseline calibration")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _plot_scatter_assists(df: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(8, 6))
    plt.scatter(df["xa_baseline"], df["assists"], alpha=0.7)
    max_val = max(df["xa_baseline"].max(), df["assists"].max())
    plt.plot([0, max_val], [0, max_val], "r--", alpha=0.5, label="xA = Assists")
    plt.xlabel("xA Baseline (sum)")
    plt.ylabel("Assists")
    plt.title("Player xA Baseline vs Assists")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _plot_probability_hist(df: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(8, 6))
    sns.histplot(df["xa_baseline"], bins=50, kde=False)
    plt.xlabel("xA Baseline per pass")
    plt.ylabel("Count")
    plt.title("Distribution of xA Baseline per pass")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _plot_spatial_heatmap(heatmap: pd.DataFrame, path: Path) -> None:
    pivot = heatmap.pivot(index="y_bin", columns="x_bin", values="xa_sum").fillna(0)
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot, cmap="magma", cbar_kws={"label": "xA sum"})
    plt.xlabel("End location X bin")
    plt.ylabel("End location Y bin")
    plt.title("Spatial distribution of xA Baseline (sum)")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _write_markdown_report(
    output_dir: Path,
    summary: Dict[str, float],
    top_players: pd.DataFrame,
    model_weights: pd.DataFrame,
) -> None:
    lines: List[str] = []
    lines.append("# cXA Phase 2 — xA Baseline analysis\n")
    lines.append(f"Total passes: {summary['total_passes']:,}")
    lines.append(f"Total assists: {summary['total_assists']}")
    lines.append(f"Sum(xA Baseline): {summary['total_xa']:.1f}")
    lines.append(f"Calibration factor: {summary['calibration_factor']:.4f}\n")

    lines.append("## Top players by xA Baseline (top 10)\n")
    lines.append(top_players.head(10).to_markdown(index=False))
    lines.append("\n")

    lines.append("## Feature weights (logistic regression)\n")
    lines.append(model_weights.to_markdown(index=False))
    lines.append("\n")

    report_path = output_dir / "phase2_xa_baseline_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")


def run_phase2_xa_baseline() -> Dict[str, object]:
    sns.set(style="whitegrid")

    repo_root = _get_repo_root()
    data_path = repo_root / "feature_store/cxa/pass_sequences.parquet"
    output_path = repo_root / "outputs/analysis/cxa/phase2_xa_baseline"
    data_dir = output_path / "data"
    plots_dir = output_path / "plots"
    data_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading passes and computing xA Baseline...")
    passes = _ensure_assist_flag(_load_passes(data_path))
    passes, model = compute_xa_baseline(passes)

    # Summaries
    summary = {
        "total_passes": len(passes),
        "total_assists": int(passes["is_assist"].sum()),
        "total_xa": float(passes["xa_baseline"].sum()),
        "mean_xa": float(passes["xa_baseline"].mean()),
        "calibration_factor": getattr(model, "calibration_factor", float("nan")),
    }

    # Feature weights
    feature_weights = pd.DataFrame(
        {
            "feature": BASELINE_FEATURES,
            "weight": model.feature_weights,
        }
    )
    feature_weights.to_csv(data_dir / "feature_weights.csv", index=False)

    # Calibration curve
    calibration_df = _calc_calibration_curve(passes)
    calibration_df.to_csv(data_dir / "calibration_curve.csv", index=False)

    # Player leaderboard
    leaderboard = _player_leaderboard(passes)
    leaderboard.to_csv(data_dir / "player_leaderboard.csv", index=False)

    # Spatial heatmap data
    heatmap_df = _spatial_heatmap(passes)
    heatmap_df.to_csv(data_dir / "spatial_heatmap_bins.csv", index=False)

    # Summary metrics CSV
    pd.DataFrame([summary]).to_csv(data_dir / "summary_metrics.csv", index=False)

    # Plots
    _plot_calibration(calibration_df, plots_dir / "calibration_curve.png")
    _plot_scatter_assists(leaderboard, plots_dir / "xa_vs_assists_scatter.png")
    _plot_probability_hist(passes, plots_dir / "xa_probability_histogram.png")
    _plot_spatial_heatmap(heatmap_df, plots_dir / "xa_spatial_heatmap.png")

    # Report
    _write_markdown_report(output_path, summary, leaderboard, feature_weights)

    logger.info(f"Phase 2 complete. Outputs: {output_path}")
    print("=" * 72)
    print("cXA Phase 2 — xA Baseline Summary")
    print("=" * 72)
    print(f"Total passes analyzed: {summary['total_passes']:,}")
    print(f"Total assists: {summary['total_assists']}")
    print(f"Sum(xA Baseline): {summary['total_xa']:.1f}")
    print(f"Mean xA per pass: {summary['mean_xa']:.6f}")
    print(f"Calibration factor: {summary['calibration_factor']:.4f}")
    print(f"Outputs: {output_path}")

    return {
        "output_path": str(output_path),
        "summary": summary,
        "leaderboard_rows": len(leaderboard),
        "calibration_bins": len(calibration_df),
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    run_phase2_xa_baseline()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
