#!/usr/bin/env python
"""cXA Phase 5: Joined comparison — xA+ Passes vs xA+ Actions (overlap goal set).

Purpose
-------
Create a joined, fair comparison between:
  - Phase 3 overlap (xA+ Passes on sequences.parquet goal set)
  - Phase 4 overlap (xA+ Actions on action_sequences.parquet filtered to same shot_ids)

This report answers:
  - Who are the biggest 'hidden creators' in passes-only attribution?
  - Who gains (or loses) credit when carries are included?
  - What fraction of goal-creation credit is carried vs passed (overlap set)?

Inputs
------
Reads the CSV artifacts produced by:
  - outputs/analysis/cxa/phase3_xap_passes/overlap/data/
  - outputs/analysis/cxa/phase4_xap_actions/overlap/data/

Outputs (outputs/analysis/cxa/phase5_joined_xap_comparison/)
-------
data/
  - joined_summary.csv
  - joined_player_comparison.csv
  - top_gainers_when_adding_carries.csv
  - top_losers_when_adding_carries.csv
  - top_pass_only_creators.csv
  - top_carry_creators.csv
plots/
  - delta_actions_minus_passes_top.png
  - passes_vs_actions_scatter.png
phase5_joined_xap_comparison_report.md

Usage
-----
    PYTHONPATH=src python -m opponent_adjusted.analysis.cxa.phase5_joined_xap_comparison
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


def _get_repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing expected artifact: {path}")
    return pd.read_csv(path)


def _plot_delta_bar(df: pd.DataFrame, path: Path, top_n: int = 25) -> None:
    show = df.head(top_n).copy()
    show = show.sort_values("delta_actions_minus_passes", ascending=True)

    plt.figure(figsize=(10, 9))
    plt.barh(show["player_name"], show["delta_actions_minus_passes"])
    plt.axvline(0, color="black", linewidth=1)
    plt.xlabel("Δ credit = xA+ Actions − xA+ Passes")
    plt.title(f"Top {top_n} gainers when adding carries")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _plot_scatter(df: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(8, 6))
    plt.scatter(df["xa_plus_passes"], df["xa_plus_actions"], alpha=0.6)
    max_val = float(max(df["xa_plus_passes"].max(), df["xa_plus_actions"].max(), 1.0))
    plt.plot([0, max_val], [0, max_val], "r--", alpha=0.5, label="Parity")
    plt.xlabel("xA+ Passes (overlap)")
    plt.ylabel("xA+ Actions (overlap)")
    plt.title("Player credit: passes-only vs passes+carries")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _write_report(
    output_dir: Path,
    summary: Dict[str, object],
    joined: pd.DataFrame,
    top_gainers: pd.DataFrame,
    top_losers: pd.DataFrame,
    top_carry: pd.DataFrame,
) -> None:
    lines: List[str] = []
    lines.append("# cXA Phase 5 — Joined xA+ comparison (overlap goal set)\n")

    lines.append("## Summary\n")
    lines.append(f"Overlap goals: {summary['overlap_goals']}")
    lines.append(f"Pass credit share (actions): {summary['action_pass_share']:.1%}")
    lines.append(f"Carry credit share (actions): {summary['action_carry_share']:.1%}")
    lines.append("\n")

    lines.append("## Biggest gainers when adding carries (top 15)\n")
    lines.append(top_gainers.head(15).to_markdown(index=False))
    lines.append("\n")

    lines.append("## Biggest losers when adding carries (top 15)\n")
    lines.append(top_losers.head(15).to_markdown(index=False))
    lines.append("\n")

    lines.append("## Top carry creators (by carry credit, top 15)\n")
    lines.append(top_carry.head(15).to_markdown(index=False))
    lines.append("\n")

    report_path = output_dir / "phase5_joined_xap_comparison_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")


def run_phase5_joined_xap_comparison() -> Dict[str, object]:
    sns.set(style="whitegrid")

    repo_root = _get_repo_root()

    p3 = repo_root / "outputs/analysis/cxa/phase3_xap_passes/overlap/data"
    p4 = repo_root / "outputs/analysis/cxa/phase4_xap_actions/overlap/data"

    p3_players = _read_csv(p3 / "player_leaderboard.csv")
    p4_players = _read_csv(p4 / "player_leaderboard.csv")
    p4_by_type = _read_csv(p4 / "player_credit_by_action_type.csv")
    p4_type_totals = _read_csv(p4 / "credit_by_action_type.csv")
    p4_summary = _read_csv(p4 / "summary_metrics.csv")

    # Build per-player pass/carry credit (from actions)
    by_type = p4_by_type.copy()
    by_type["action_type"] = by_type["action_type"].astype(str)

    carry = by_type[by_type["action_type"] == "Carry"].rename(columns={"xa_plus": "carry_credit"})
    pas = by_type[by_type["action_type"] == "Pass"].rename(columns={"xa_plus": "pass_credit"})

    carry = carry[["player_id", "player_name", "carry_credit"]]
    pas = pas[["player_id", "player_name", "pass_credit"]]

    # Player totals
    p3 = p3_players[["player_id", "player_name", "xa_plus", "assist_passes"]].rename(
        columns={"xa_plus": "xa_plus_passes"}
    )
    p4 = p4_players[["player_id", "player_name", "xa_plus"]].rename(
        columns={"xa_plus": "xa_plus_actions"}
    )

    joined = p3.merge(p4, on=["player_id", "player_name"], how="outer")
    joined = joined.merge(pas, on=["player_id", "player_name"], how="left")
    joined = joined.merge(carry, on=["player_id", "player_name"], how="left")

    for col in ["xa_plus_passes", "xa_plus_actions", "assist_passes", "pass_credit", "carry_credit"]:
        if col in joined.columns:
            joined[col] = joined[col].fillna(0.0)

    joined["delta_actions_minus_passes"] = joined["xa_plus_actions"] - joined["xa_plus_passes"]

    # Rankings
    top_gainers = joined.sort_values("delta_actions_minus_passes", ascending=False)
    top_losers = joined.sort_values("delta_actions_minus_passes", ascending=True)

    top_carry = (
        joined.sort_values("carry_credit", ascending=False)
        [["player_id", "player_name", "carry_credit", "xa_plus_actions", "xa_plus_passes", "delta_actions_minus_passes"]]
    )

    top_pass_only = (
        joined.sort_values("xa_plus_passes", ascending=False)
        [["player_id", "player_name", "xa_plus_passes", "assist_passes"]]
    )

    # Summary
    action_pass_share = float(p4_type_totals.loc[p4_type_totals["action_type"] == "Pass", "share"].iloc[0])
    action_carry_share = float(p4_type_totals.loc[p4_type_totals["action_type"] == "Carry", "share"].iloc[0])

    overlap_goals = int(float(p4_summary["num_goals"].iloc[0]))

    summary = {
        "overlap_goals": overlap_goals,
        "action_pass_share": action_pass_share,
        "action_carry_share": action_carry_share,
    }

    # Write outputs
    out_root = repo_root / "outputs/analysis/cxa/phase5_joined_xap_comparison"
    data_dir = out_root / "data"
    plots_dir = out_root / "plots"
    data_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([summary]).to_csv(data_dir / "joined_summary.csv", index=False)
    joined.to_csv(data_dir / "joined_player_comparison.csv", index=False)
    top_gainers.head(100).to_csv(data_dir / "top_gainers_when_adding_carries.csv", index=False)
    top_losers.head(100).to_csv(data_dir / "top_losers_when_adding_carries.csv", index=False)
    top_pass_only.head(100).to_csv(data_dir / "top_pass_only_creators.csv", index=False)
    top_carry.head(100).to_csv(data_dir / "top_carry_creators.csv", index=False)

    _plot_delta_bar(top_gainers, plots_dir / "delta_actions_minus_passes_top.png")
    _plot_scatter(joined, plots_dir / "passes_vs_actions_scatter.png")

    _write_report(out_root, summary, joined, top_gainers, top_losers, top_carry)

    logger.info(f"Phase 5 complete. Outputs: {out_root}")
    print("=" * 72)
    print("cXA Phase 5 — Joined xA+ comparison")
    print("=" * 72)
    print(f"Overlap goals: {overlap_goals}")
    print(f"Action credit split: Pass {action_pass_share:.1%} | Carry {action_carry_share:.1%}")
    print(f"Outputs: {out_root}")

    return {"output_path": str(out_root), "summary": summary}


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    run_phase5_joined_xap_comparison()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
