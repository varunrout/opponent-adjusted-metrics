"""Shared plotting utilities for modeling outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DEFAULT_PALETTE = ["#023047", "#219EBC", "#FFB703", "#FB8500", "#8ECAE6"]


def configure_modeling_style() -> None:
	"""Apply a consistent Matplotlib/Seaborn style for modeling charts."""

	sns.set_theme(style="whitegrid", palette="deep")
	plt.rcParams.update(
		{
			"axes.titlesize": 12,
			"axes.labelsize": 11,
			"xtick.labelsize": 10,
			"ytick.labelsize": 10,
		}
	)


def load_metrics_json(path: Path) -> dict:
	"""Load a metrics JSON file and return its dictionary."""

	with path.open("r", encoding="utf-8") as fp:
		return json.load(fp)


def metrics_map_to_frame(metrics_map: Mapping[str, Mapping[str, float]], metric_keys: list[str]) -> pd.DataFrame:
	rows: list[dict[str, object]] = []
	for model_label, metrics in metrics_map.items():
		for metric in metric_keys:
			if metric not in metrics:
				continue
			rows.append(
				{
					"model": model_label,
					"metric": metric,
					"value": metrics[metric],
				}
			)
	return pd.DataFrame(rows)


def format_metric_name(metric: str) -> str:
    mapping = {
        "brier_mean": "Brier Score",
        "log_loss_mean": "Log Loss",
        "auc_mean": "ROC AUC",
    }
    return mapping.get(metric, metric)


def annotate_bars(ax: plt.Axes) -> None:
    for patch in ax.patches:
        height = patch.get_height()
        ax.annotate(
            f"{height:.4f}",
            (patch.get_x() + patch.get_width() / 2, height),
            ha="center",
            va="bottom",
            fontsize=9,
        )