#!/usr/bin/env python
"""cXA Phase 4: xA+ Actions attribution analysis.

Purpose
-------
Compute xA+ Actions (softmax credit across actions: Pass/Carry/Dribble)
then analyze attribution patterns:
  - credit by action type (Pass vs Carry vs Dribble)
  - credit by action position (action1..action5)
  - player leaderboards (xA+ credit)
  - overlap comparison with Phase 3 goal set (common shot_ids)

Outputs (outputs/analysis/cxa/phase4_xap_actions/)
-------
full/  (all goals in action_sequences.parquet)
  data/
  plots/
  phase4_xap_actions_report.md
overlap/ (only goals whose shot_id exists in sequences.parquet goal set)
  data/
  plots/
  phase4_xap_actions_overlap_report.md

Usage
-----
    PYTHONPATH=src python -m opponent_adjusted.analysis.cxa.phase4_xap_actions
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Set

import matplotlib

matplotlib.use("Agg")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from opponent_adjusted.features.cxa.xa_plus_actions import compute_xa_plus_actions

logger = logging.getLogger(__name__)


def _get_repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _load_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing parquet: {path}")
    df = pd.read_parquet(path)
    logger.info(f"Loaded {len(df):,} rows from {path}")
    return df


def _goal_shot_ids_sequences(sequences_df: pd.DataFrame) -> Set[int]:
    df = sequences_df.copy()
    if "is_goal" in df.columns:
        df = df[df["is_goal"].fillna(False).astype(bool)]
    if "shot_id" not in df.columns:
        return set()
    ids = set(df["shot_id"].dropna().astype(int).tolist())
    return ids


def _credit_by_action_type(actions_long: pd.DataFrame) -> pd.DataFrame:
    df = actions_long.copy()
    df = df[df["is_goal"].fillna(False).astype(bool)]

    grouped = (
        df.groupby("action_type")
        .agg(
            credit=("xa_plus", "sum"),
            actions=("xa_plus", "size"),
        )
        .reset_index()
    )
    total = grouped["credit"].sum()
    grouped["share"] = grouped["credit"] / max(total, 1e-9)
    grouped = grouped.sort_values("credit", ascending=False)
    return grouped


def _credit_by_action_position(sequences_with_credit: pd.DataFrame) -> pd.DataFrame:
    df = sequences_with_credit.copy()
    df = df[df["is_goal"].fillna(False).astype(bool)]

    cols = [c for c in df.columns if c.startswith("xa_plus_action")]
    totals = []
    total_credit = float(df["xa_plus_total"].sum())
    for col in sorted(cols, key=lambda s: int(s.replace("xa_plus_action", ""))):
        idx = int(col.replace("xa_plus_action", ""))
        credit = float(df[col].sum())
        totals.append(
            {"action_num": idx, "credit": credit, "share": credit / max(total_credit, 1e-9)}
        )
    return pd.DataFrame(totals)


def _player_leaderboard(actions_long: pd.DataFrame) -> pd.DataFrame:
    df = actions_long.copy()
    df = df[df["is_goal"].fillna(False).astype(bool)]

    grouped = (
        df.groupby(["player_id", "player_name"])  # type: ignore[pd]
        .agg(
            actions_in_goal_sequences=("xa_plus", "size"),
            xa_plus=("xa_plus", "sum"),
            xa_plus_mean=("xa_plus", "mean"),
            pass_actions=("action_type", lambda s: (s == "Pass").sum()),
        )
        .reset_index()
    )

    grouped = grouped.sort_values("xa_plus", ascending=False)
    return grouped


def _plot_type_share(df: pd.DataFrame, path: Path, title: str) -> None:
    plt.figure(figsize=(7, 5))
    sns.barplot(data=df, x="action_type", y="share")
    plt.gca().yaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")
    plt.xlabel("Action type")
    plt.ylabel("Share of total xA+ credit")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _plot_position_share(df: pd.DataFrame, path: Path, title: str) -> None:
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x="action_num", y="share")
    plt.gca().yaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")
    plt.xlabel("Action position in sequence")
    plt.ylabel("Share of total xA+ credit")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _plot_top_players(players: pd.DataFrame, path: Path, title: str, top_n: int = 20) -> None:
    top = players.head(top_n).copy()
    top = top.sort_values("xa_plus", ascending=True)
    plt.figure(figsize=(10, 8))
    plt.barh(top["player_name"], top["xa_plus"])
    plt.xlabel("xA+ Actions (sum)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _write_report(
    report_path: Path,
    summary: Dict[str, float],
    type_df: pd.DataFrame,
    pos_df: pd.DataFrame,
    players: pd.DataFrame,
) -> None:
    lines: List[str] = []
    lines.append("# cXA Phase 4 — xA+ Actions attribution\n")
    lines.append("## Summary\n")
    lines.append(f"Goal sequences: {int(summary['num_goals'])}")
    lines.append(f"Total xA+ credit assigned: {summary['total_credit']:.1f}")
    lines.append(f"Temperature: {summary['temperature']:.2f}\n")

    lines.append("## Credit by action type\n")
    lines.append(type_df.to_markdown(index=False))
    lines.append("\n")

    lines.append("## Credit by action position\n")
    lines.append(pos_df.to_markdown(index=False))
    lines.append("\n")

    lines.append("## Top players (xA+ Actions)\n")
    lines.append(players.head(15).to_markdown(index=False))

    report_path.write_text("\n".join(lines), encoding="utf-8")


def _write_outputs(
    out_dir: Path,
    title_suffix: str,
    sequences_with_credit: pd.DataFrame,
    actions_long: pd.DataFrame,
    temperature: float,
    report_name: str,
    generate_plots: bool = True,
) -> Dict[str, Any]:
    data_dir = out_dir / "data"
    plots_dir = out_dir / "plots"
    data_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    goal_sequences = sequences_with_credit[
        sequences_with_credit["is_goal"].fillna(False).astype(bool)
    ]
    total_credit = float(goal_sequences["xa_plus_total"].sum())

    type_df = _credit_by_action_type(actions_long)
    pos_df = _credit_by_action_position(sequences_with_credit)
    players = _player_leaderboard(actions_long)

    # Player credit by action type
    goal_actions = actions_long[actions_long["is_goal"].fillna(False).astype(bool)].copy()
    if "player_id" in goal_actions.columns and "action_type" in goal_actions.columns:
        by_type = (
            goal_actions.groupby(["player_id", "player_name", "action_type"])  # type: ignore[pd]
            .agg(xa_plus=("xa_plus", "sum"), actions=("xa_plus", "size"))
            .reset_index()
        )
        by_type = by_type.sort_values("xa_plus", ascending=False)
    else:
        by_type = pd.DataFrame(
            columns=["player_id", "player_name", "action_type", "xa_plus", "actions"]
        )

    summary = {
        "num_goals": float(goal_sequences.shape[0]),
        "total_credit": total_credit,
        "temperature": float(temperature),
    }

    pd.DataFrame([summary]).to_csv(data_dir / "summary_metrics.csv", index=False)
    type_df.to_csv(data_dir / "credit_by_action_type.csv", index=False)
    pos_df.to_csv(data_dir / "credit_by_action_position.csv", index=False)
    players.to_csv(data_dir / "player_leaderboard.csv", index=False)
    by_type.to_csv(data_dir / "player_credit_by_action_type.csv", index=False)

    if generate_plots:
        _plot_type_share(
            type_df,
            plots_dir / "credit_by_action_type.png",
            f"xA+ Actions: credit by type {title_suffix}",
        )
        _plot_position_share(
            pos_df,
            plots_dir / "credit_by_action_position.png",
            f"xA+ Actions: credit by position {title_suffix}",
        )
        _plot_top_players(
            players,
            plots_dir / "top_players_xa_plus.png",
            f"Top players by xA+ Actions {title_suffix}",
        )

    _write_report(
        out_dir / report_name,
        summary,
        type_df,
        pos_df,
        players,
    )

    return {
        "summary": summary,
        "players": players,
        "type_df": type_df,
        "pos_df": pos_df,
    }


def run_phase4_xap_actions(temperature: float = 1.0) -> Dict[str, Any]:
    sns.set(style="whitegrid")

    repo_root = _get_repo_root()
    action_sequences_path = repo_root / "feature_store/cxa/action_sequences.parquet"
    sequences_path = repo_root / "feature_store/cxa/sequences.parquet"

    output_root = repo_root / "outputs/analysis/cxa/phase4_xap_actions"
    full_dir = output_root / "full"
    overlap_dir = output_root / "overlap"

    logger.info("Loading action sequences...")
    action_sequences = _load_parquet(action_sequences_path)

    logger.info("Computing xA+ Actions once (fit on full action_sequences)...")
    seq_with_credit, actions_long, _model = compute_xa_plus_actions(
        action_sequences, temperature=temperature
    )

    logger.info("Writing full outputs...")
    full_res: Dict[str, Any] = _write_outputs(
        full_dir,
        "(full)",
        seq_with_credit,
        actions_long,
        temperature,
        report_name="phase4_xap_actions_report.md",
        generate_plots=True,
    )

    # Overlap with Phase 3 goal set (shot_id intersection)
    sequences = _load_parquet(sequences_path)
    goal_ids = _goal_shot_ids_sequences(sequences)

    if "shot_id" in seq_with_credit.columns:
        seq_overlap = seq_with_credit[seq_with_credit["shot_id"].isin(goal_ids)].copy()
    else:
        seq_overlap = seq_with_credit.copy()

    if "shot_id" in actions_long.columns:
        actions_overlap = actions_long[actions_long["shot_id"].isin(goal_ids)].copy()
    else:
        actions_overlap = actions_long.copy()

    logger.info("Writing overlap outputs (filtered after scoring)...")
    overlap_res: Dict[str, Any] = _write_outputs(
        overlap_dir,
        "(overlap)",
        seq_overlap,
        actions_overlap,
        temperature,
        report_name="phase4_xap_actions_overlap_report.md",
        generate_plots=False,
    )

    logger.info(f"Phase 4 complete. Outputs: {output_root}")
    print("=" * 72)
    print("cXA Phase 4 — xA+ Actions Summary")
    print("=" * 72)
    print(
        f"Full goals: {int(full_res['summary']['num_goals'])} | credit: {full_res['summary']['total_credit']:.1f}"
    )
    print(
        f"Overlap goals: {int(overlap_res['summary']['num_goals'])} | credit: {overlap_res['summary']['total_credit']:.1f}"
    )
    print(f"Outputs: {output_root}")

    return {
        "output_path": str(output_root),
        "full": full_res["summary"],
        "overlap": overlap_res["summary"],
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    run_phase4_xap_actions()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
