#!/usr/bin/env python
"""cXA Phase 3: xA+ Passes attribution analysis.

Purpose
-------
Compute xA+ Passes (softmax credit across up to 3 passes in goal sequences)
then analyze how credit is distributed:
  - credit by pass position (1/2/3)
  - credit to the assist pass vs earlier passes
  - player leaderboards (xA+ credit vs assists)

Outputs (outputs/analysis/cxa/phase3_xap_passes/)
-------
full/
    data/
    plots/  (plots currently disabled for speed)
    phase3_xap_passes_report.md
overlap/  (only goals whose shot_id exists in action_sequences.parquet goal set)
    data/
    plots/  (plots disabled)
    phase3_xap_passes_report.md

Usage
-----
    PYTHONPATH=src python -m opponent_adjusted.analysis.cxa.phase3_xap_passes
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from opponent_adjusted.features.cxa.xa_plus_passes import (
    PASS_FEATURES,
    MAX_PASSES,
    compute_xa_plus_passes,
)

logger = logging.getLogger(__name__)


def _get_repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _load_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing parquet: {path}")
    df = pd.read_parquet(path)
    logger.info(f"Loaded {len(df):,} rows from {path}")
    return df


def _goal_shot_ids_sequences(sequences_df: pd.DataFrame) -> set[int]:
    df = sequences_df.copy()
    if "is_goal" in df.columns:
        df = df[df["is_goal"] == True]
    if "shot_id" not in df.columns:
        return set()
    return set(df["shot_id"].dropna().astype(int).tolist())


def _write_outputs(
    out_dir: Path,
    title_suffix: str,
    sequences: pd.DataFrame,
    passes_long: pd.DataFrame,
    model,
    temperature: float,
    generate_plots: bool = True,
) -> Dict[str, Any]:
    data_dir = out_dir / "data"
    plots_dir = out_dir / "plots"
    data_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    goal_sequences = sequences[sequences["is_goal"] == True]
    total_credit = float(goal_sequences["xa_plus_total"].sum())
    assist_credit = float(passes_long[passes_long["is_assist"] == True]["xa_plus"].sum())

    summary = {
        "num_goals": float(goal_sequences.shape[0]),
        "total_credit": total_credit,
        "assist_credit": assist_credit,
        "assist_credit_share": assist_credit / max(total_credit, 1e-9),
        "temperature": float(temperature),
    }

    # Feature weights
    weights = pd.DataFrame(
        {
            "feature": PASS_FEATURES,
            "weight": getattr(model, "feature_weights", np.array([np.nan] * len(PASS_FEATURES))),
        }
    )
    weights.to_csv(data_dir / "feature_weights.csv", index=False)

    # Credit by pass position
    credit_by_pos = _credit_by_pass_position(goal_sequences)
    credit_by_pos.to_csv(data_dir / "credit_by_pass_position.csv", index=False)

    # Player leaderboard and hidden creators
    players = _player_leaderboard(passes_long)
    players.to_csv(data_dir / "player_leaderboard.csv", index=False)

    hidden = players.sort_values("xa_plus_minus_assists", ascending=False).head(50)
    hidden.to_csv(data_dir / "hidden_creators.csv", index=False)

    # Assist credit distribution
    assist_dist = _assist_pass_credit_distribution(passes_long)
    assist_dist.to_csv(data_dir / "assist_pass_credit_distribution.csv", index=False)

    pd.DataFrame([summary]).to_csv(data_dir / "summary_metrics.csv", index=False)

    # Plots
    if generate_plots:
        _plot_credit_by_position(credit_by_pos, plots_dir / "credit_by_pass_position.png")
        _plot_scatter_xa_vs_assists(players, plots_dir / "xa_plus_vs_assists_scatter.png")
        _plot_top_players(players, plots_dir / "top_players_xa_plus.png")
        _plot_assist_credit_hist(assist_dist, plots_dir / "assist_pass_credit_hist.png")

    # Report
    _write_report(out_dir, summary, credit_by_pos, players)

    logger.info(f"Phase 3 outputs written: {out_dir} {title_suffix}")
    return {"summary": summary, "players": players, "hidden": hidden}


def _credit_by_pass_position(sequences_with_credit: pd.DataFrame) -> pd.DataFrame:
    totals = []
    total_credit = sequences_with_credit["xa_plus_total"].sum()
    for i in range(1, MAX_PASSES + 1):
        col = f"xa_plus_pass{i}"
        if col not in sequences_with_credit.columns:
            continue
        credit = sequences_with_credit[col].sum()
        totals.append(
            {
                "pass_num": i,
                "credit": float(credit),
                "share": float(credit / max(total_credit, 1e-9)),
            }
        )
    return pd.DataFrame(totals)


def _player_leaderboard(passes_long: pd.DataFrame) -> pd.DataFrame:
    df = passes_long.copy()
    df = df[df["is_goal"] == True]

    grouped = (
        df.groupby(["player_id", "player_name"])  # type: ignore[pd]
        .agg(
            passes_in_goal_sequences=("pass_id", "count"),
            assist_passes=("is_assist", "sum"),
            xa_plus=("xa_plus", "sum"),
            xa_plus_mean=("xa_plus", "mean"),
        )
        .reset_index()
    )

    grouped["xa_plus_minus_assists"] = grouped["xa_plus"] - grouped["assist_passes"]
    grouped = grouped.sort_values("xa_plus", ascending=False)
    return grouped


def _assist_pass_credit_distribution(passes_long: pd.DataFrame) -> pd.DataFrame:
    df = passes_long.copy()
    df = df[(df["is_goal"] == True) & (df["is_assist"] == True)]
    return df[["sequence_id", "pass_num", "xa_plus"]].rename(columns={"xa_plus": "assist_pass_credit"})


def _plot_credit_by_position(df: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x="pass_num", y="share")
    plt.gca().yaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")
    plt.xlabel("Pass position in sequence")
    plt.ylabel("Share of total xA+ credit")
    plt.title("xA+ Passes: credit share by pass position")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _plot_scatter_xa_vs_assists(players: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(8, 6))
    plt.scatter(players["xa_plus"], players["assist_passes"], alpha=0.7)
    max_val = max(players["xa_plus"].max(), players["assist_passes"].max())
    plt.plot([0, max_val], [0, max_val], "r--", alpha=0.5, label="xA+ = assists")
    plt.xlabel("xA+ Passes (sum)")
    plt.ylabel("Assists (count, last pass in goal sequences)")
    plt.title("Player xA+ Passes vs assists")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _plot_top_players(players: pd.DataFrame, path: Path, top_n: int = 20) -> None:
    top = players.head(top_n).copy()
    top = top.sort_values("xa_plus", ascending=True)
    plt.figure(figsize=(10, 8))
    plt.barh(top["player_name"], top["xa_plus"])
    plt.xlabel("xA+ Passes (sum)")
    plt.title(f"Top {top_n} players by xA+ Passes")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _plot_assist_credit_hist(df: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(8, 6))
    sns.histplot(data=df, x="assist_pass_credit", bins=30)
    plt.xlabel("Assist-pass credit (xA+)")
    plt.ylabel("Count")
    plt.title("Distribution of assist-pass credit per goal")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _write_report(
    output_dir: Path,
    summary: Dict[str, float],
    credit_by_pos: pd.DataFrame,
    players: pd.DataFrame,
) -> None:
    lines: List[str] = []
    lines.append("# cXA Phase 3 — xA+ Passes attribution\n")

    lines.append("## Summary\n")
    lines.append(f"Goal sequences: {int(summary['num_goals'])}")
    lines.append(f"Total xA+ credit assigned: {summary['total_credit']:.1f}")
    lines.append(f"Credit to assist passes: {summary['assist_credit']:.1f} ({summary['assist_credit_share']:.1%})\n")

    lines.append("## Credit by pass position\n")
    lines.append(credit_by_pos.to_markdown(index=False))
    lines.append("\n")

    lines.append("## Top players (xA+ Passes)\n")
    lines.append(players.head(15).to_markdown(index=False))

    report_path = output_dir / "phase3_xap_passes_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")


def run_phase3_xap_passes(temperature: float = 1.0) -> Dict[str, Any]:
    sns.set(style="whitegrid")

    repo_root = _get_repo_root()
    sequences_path = repo_root / "feature_store/cxa/sequences.parquet"
    action_sequences_path = repo_root / "feature_store/cxa/action_sequences.parquet"
    output_root = repo_root / "outputs/analysis/cxa/phase3_xap_passes"
    full_dir = output_root / "full"
    overlap_dir = output_root / "overlap"
    full_dir.mkdir(parents=True, exist_ok=True)
    overlap_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading sequences and computing xA+ Passes...")
    sequences = _load_parquet(sequences_path)
    seq_with_credit, passes_long, model = compute_xa_plus_passes(sequences, temperature=temperature)

    full_res: Dict[str, Any] = _write_outputs(
        full_dir, "(full)", seq_with_credit, passes_long, model, temperature, generate_plots=False
    )

    # True overlap with Phase 4 universe: intersection of goal shot_ids
    action_sequences = _load_parquet(action_sequences_path)
    goal_ids_passes = _goal_shot_ids_sequences(sequences)
    goal_ids_actions = _goal_shot_ids_sequences(action_sequences)
    goal_ids = goal_ids_passes & goal_ids_actions
    if "shot_id" in seq_with_credit.columns:
        seq_overlap = seq_with_credit[seq_with_credit["shot_id"].isin(goal_ids)].copy()
    else:
        seq_overlap = seq_with_credit.copy()
    if "shot_id" in passes_long.columns:
        passes_overlap = passes_long[passes_long["shot_id"].isin(goal_ids)].copy()
    else:
        passes_overlap = passes_long.copy()

    overlap_res: Dict[str, Any] = _write_outputs(
        overlap_dir, "(overlap)", seq_overlap, passes_overlap, model, temperature, generate_plots=False
    )

    logger.info(f"Phase 3 complete. Outputs: {output_root}")
    print("=" * 72)
    print("cXA Phase 3 — xA+ Passes Summary")
    print("=" * 72)
    print(f"Full goals: {int(full_res['summary']['num_goals'])} | credit: {full_res['summary']['total_credit']:.1f}")
    print(f"Overlap goals: {int(overlap_res['summary']['num_goals'])} | credit: {overlap_res['summary']['total_credit']:.1f}")
    print(f"Outputs: {output_root}")

    return {
        "output_path": str(output_root),
        "full": full_res["summary"],
        "overlap": overlap_res["summary"],
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    run_phase3_xap_passes()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
