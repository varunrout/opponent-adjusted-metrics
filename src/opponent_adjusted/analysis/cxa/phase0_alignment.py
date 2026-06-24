#!/usr/bin/env python
"""cXA Phase 0: Dataset alignment for assist / goal-sequence analyses.

Purpose
-------
Establish an apples-to-apples goal population before comparing:
  - xA+ Passes (built from `feature_store/cxa/sequences.parquet`)
  - xA+ Actions (built from `feature_store/cxa/action_sequences.parquet`)

Outputs (outputs/analysis/cxa/phase0_alignment/)
-------
data/
  - goal_counts.csv
  - overlap_summary.csv
  - shot_ids_missing_in_actions.csv
  - shot_ids_extra_in_actions.csv
plots/
  - goal_population_counts.png
phase0_alignment_report.md

Usage
-----
    python -m opponent_adjusted.analysis.cxa.phase0_alignment
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Set, List

import pandas as pd

logger = logging.getLogger(__name__)


def _get_repo_root() -> Path:
    """Return repo root (4 levels up from this file)."""
    return Path(__file__).resolve().parents[4]


def _load_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_parquet(path)


def _safe_goal_shot_ids(df: pd.DataFrame, label: str) -> Set[int]:
    required = {"shot_id", "is_goal"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{label} is missing required columns: {sorted(missing)}")

    goals = df[df["is_goal"].fillna(False).astype(bool)]
    shot_ids = goals["shot_id"].dropna().astype(int).tolist()
    return set(shot_ids)


def _write_markdown_report(
    out_path: Path,
    sequences_goals: int,
    actions_goals: int,
    overlap: int,
    missing_in_actions: List[int],
    extra_in_actions: List[int],
) -> None:
    lines: List[str] = []
    lines.append("# cXA Phase 0 — Goal Population Alignment")
    lines.append("")
    lines.append(
        "This report aligns goal/assist populations between pass-only sequences and action sequences."
    )
    lines.append("")
    lines.append("## Counts")
    lines.append("")
    lines.append("| Dataset | Goals |")
    lines.append("|---------|-------|")
    lines.append(f"| sequences.parquet | **{sequences_goals}** |")
    lines.append(f"| action_sequences.parquet | **{actions_goals}** |")
    lines.append(f"| Overlap (common `shot_id`) | **{overlap}** |")
    lines.append("")
    lines.append("## Alignment Status")
    lines.append("")
    if sequences_goals == actions_goals and overlap == sequences_goals:
        lines.append(
            "✅ **ALIGNED** — Goal populations match. You can compare xA+ Passes and xA+ Actions directly."
        )
    else:
        lines.append("⚠️ **NOT ALIGNED** — Goal populations differ.")
        lines.append("")
        lines.append("For fair comparisons, either:")
        lines.append("1. Run comparisons on the **overlap set only** (360 goals)")
        lines.append("2. Fix the action sequence builder to include missing goals (recommended)")
    lines.append("")

    lines.append("## Missing Goals (in sequences but NOT in actions)")
    lines.append("")
    lines.append(f"Count: **{len(missing_in_actions)}**")
    if missing_in_actions:
        lines.append("")
        lines.append("| shot_id |")
        lines.append("|---------|")
        for sid in missing_in_actions[:20]:
            lines.append(f"| {sid} |")
        if len(missing_in_actions) > 20:
            lines.append(f"| ... ({len(missing_in_actions) - 20} more) |")

    lines.append("")
    lines.append("## Extra Goals (in actions but NOT in sequences)")
    lines.append("")
    lines.append(f"Count: **{len(extra_in_actions)}**")
    lines.append("")
    lines.append(
        "These are likely goals with no passes in buildup (solo runs, direct shots after winning ball)."
    )

    out_path.write_text("\n".join(lines), encoding="utf-8")


def _plot_counts(out_png: Path, counts: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    bars = ax.bar(counts["dataset"], counts["goals"], color=colors[: len(counts)], alpha=0.85)
    ax.set_title("Goal Population Counts (Phase 0 Alignment)", fontsize=12, fontweight="bold")
    ax.set_ylabel("# Goals (is_goal=True)")
    ax.set_xlabel("")

    for bar, v in zip(bars, counts["goals"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + 5,
            str(int(v)),
            ha="center",
            va="bottom",
            fontsize=11,
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def run_phase0_alignment(
    feature_store_path: Path | None = None,
    output_path: Path | None = None,
) -> dict:
    """Run Phase 0 alignment analysis.

    Args:
        feature_store_path: Path to feature_store/cxa/ (default: auto-detect)
        output_path: Path to output folder (default: outputs/analysis/cxa/phase0_alignment/)

    Returns:
        dict with alignment summary
    """
    repo_root = _get_repo_root()

    if feature_store_path is None:
        feature_store_path = repo_root / "feature_store" / "cxa"
    if output_path is None:
        output_path = repo_root / "outputs" / "analysis" / "cxa" / "phase0_alignment"

    sequences_path = feature_store_path / "sequences.parquet"
    actions_path = feature_store_path / "action_sequences.parquet"

    out_data = output_path / "data"
    out_plots = output_path / "plots"
    out_data.mkdir(parents=True, exist_ok=True)
    out_plots.mkdir(parents=True, exist_ok=True)

    logger.info("Loading datasets...")
    sequences = _load_parquet(sequences_path)
    actions = _load_parquet(actions_path)

    seq_goal_ids = _safe_goal_shot_ids(sequences, label="sequences.parquet")
    act_goal_ids = _safe_goal_shot_ids(actions, label="action_sequences.parquet")

    overlap_ids = sorted(seq_goal_ids & act_goal_ids)
    missing_in_actions = sorted(seq_goal_ids - act_goal_ids)
    extra_in_actions = sorted(act_goal_ids - seq_goal_ids)

    # Save CSVs
    counts = pd.DataFrame(
        [
            {"dataset": "sequences.parquet", "goals": len(seq_goal_ids)},
            {"dataset": "action_sequences.parquet", "goals": len(act_goal_ids)},
            {"dataset": "overlap", "goals": len(overlap_ids)},
        ]
    )

    overlap_summary = pd.DataFrame(
        [
            {
                "sequences_goals": len(seq_goal_ids),
                "actions_goals": len(act_goal_ids),
                "overlap_goals": len(overlap_ids),
                "missing_in_actions": len(missing_in_actions),
                "extra_in_actions": len(extra_in_actions),
            }
        ]
    )

    counts.to_csv(out_data / "goal_counts.csv", index=False)
    overlap_summary.to_csv(out_data / "overlap_summary.csv", index=False)
    pd.DataFrame({"shot_id": missing_in_actions}).to_csv(
        out_data / "shot_ids_missing_in_actions.csv", index=False
    )
    pd.DataFrame({"shot_id": extra_in_actions}).to_csv(
        out_data / "shot_ids_extra_in_actions.csv", index=False
    )

    # Plot
    try:
        _plot_counts(out_plots / "goal_population_counts.png", counts)
        logger.info(f"Saved plot: {out_plots / 'goal_population_counts.png'}")
    except Exception as exc:
        logger.warning(f"Plot skipped: {exc}")

    # Markdown report
    _write_markdown_report(
        output_path / "phase0_alignment_report.md",
        sequences_goals=len(seq_goal_ids),
        actions_goals=len(act_goal_ids),
        overlap=len(overlap_ids),
        missing_in_actions=missing_in_actions,
        extra_in_actions=extra_in_actions,
    )

    result = {
        "sequences_goals": len(seq_goal_ids),
        "actions_goals": len(act_goal_ids),
        "overlap_goals": len(overlap_ids),
        "missing_in_actions": len(missing_in_actions),
        "extra_in_actions": len(extra_in_actions),
        "aligned": len(seq_goal_ids) == len(act_goal_ids) == len(overlap_ids),
        "output_path": str(output_path),
    }

    return result


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

    result = run_phase0_alignment()

    print("=" * 72)
    print("cXA Phase 0 — Alignment Summary")
    print("=" * 72)
    print(f"sequences.parquet goals:        {result['sequences_goals']}")
    print(f"action_sequences.parquet goals: {result['actions_goals']}")
    print(f"overlap goals:                  {result['overlap_goals']}")
    print(f"missing in actions:             {result['missing_in_actions']}")
    print(f"extra in actions:               {result['extra_in_actions']}")
    print()
    if result["aligned"]:
        print("✅ ALIGNED — Ready for fair comparison")
    else:
        print("⚠️  NOT ALIGNED — Use overlap set (360 goals) for fair comparison")
    print()
    print(f"Outputs: {result['output_path']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
