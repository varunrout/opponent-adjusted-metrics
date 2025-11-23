"""Generate charts for CxG prediction runs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

from opponent_adjusted.db.models import Team
from opponent_adjusted.db.session import SessionLocal
from opponent_adjusted.utils.logging import get_logger

logger = get_logger(__name__)


def _ensure_team_names(df: pd.DataFrame) -> pd.DataFrame:
    if "team_name" in df.columns:
        return df

    with SessionLocal() as session:
        rows = session.query(Team.id, Team.name).all()
    lookup = {team_id: name for team_id, name in rows}
    df = df.copy()
    df["team_name"] = df["team_id"].map(lookup).fillna(df["team_id"].astype(str))
    return df


def plot_team_totals(df: pd.DataFrame, out_path: Path, top_n: int = 10) -> None:
    ranked = df.sort_values("cxg_for", ascending=False).head(top_n)
    x = range(len(ranked))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar([i - width for i in x], ranked["cxg_for"], width, label="CxG For")
    ax.bar(x, ranked["provider_xg_for"], width, label="Provider xG For")
    ax.bar([i + width for i in x], ranked["goals_for"], width, label="Goals For")

    ax.set_xticks(list(x))
    ax.set_xticklabels(ranked["team_name"], rotation=45, ha="right")
    ax.set_ylabel("Season Total")
    ax.set_title("CxG vs Provider xG vs Goals")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    logger.info("Saved team totals chart -> %s", out_path)


def plot_finishing_deltas(df: pd.DataFrame, out_path: Path, top_n: int = 10) -> None:
    df = df.copy()
    df["finishing_delta"] = df["goals_for"] - df["cxg_for"]
    df["provider_delta"] = df["goals_for"] - df["provider_xg_for"]
    ranked = df.sort_values("finishing_delta", key=abs, ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(ranked["team_name"], ranked["finishing_delta"], label="Goals - CxG")
    ax.bar(ranked["team_name"], ranked["provider_delta"], alpha=0.6, label="Goals - Provider xG")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Goals minus Expected Goals")
    ax.set_title("Finishing delta (CxG vs Provider xG)")
    ax.set_xticklabels(ranked["team_name"], rotation=45, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    logger.info("Saved finishing delta chart -> %s", out_path)


def plot_cxg_vs_provider_scatter(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(df["provider_xg_for"], df["cxg_for"], alpha=0.8)

    max_val = float(max(df["provider_xg_for"].max(), df["cxg_for"].max()))
    ax.plot([0, max_val], [0, max_val], linestyle="--", color="gray", label="y = x")

    for _, row in df.iterrows():
        ax.annotate(row["team_name"], (row["provider_xg_for"], row["cxg_for"]), fontsize=8)

    ax.set_xlabel("Provider xG For")
    ax.set_ylabel("CxG For")
    ax.set_title("CxG vs Provider xG (season totals)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    logger.info("Saved scatter chart -> %s", out_path)


def generate_prediction_charts(team_csv: Path, output_dir: Optional[Path] = None, top_n: int = 10) -> None:
    df = pd.read_csv(team_csv)
    df = _ensure_team_names(df)

    if output_dir is None:
        output_dir = team_csv.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_team_totals(df, output_dir / "team_totals.png", top_n)
    plot_finishing_deltas(df, output_dir / "finishing_delta.png", top_n)
    plot_cxg_vs_provider_scatter(df, output_dir / "cxg_vs_provider_scatter.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot prediction summaries")
    parser.add_argument("team_csv", type=Path, help="Path to team_aggregates.csv")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for charts")
    parser.add_argument("--top-n", type=int, default=10, help="Number of teams to show in bar charts")
    return parser.parse_args()


if __name__ == "__main__":  # pragma: no cover
    args = parse_args()
    generate_prediction_charts(args.team_csv, args.output_dir, args.top_n)
