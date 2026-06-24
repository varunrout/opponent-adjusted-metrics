"""Phase 4 EDA: Action Sequences Analysis.

Comprehensive EDA for action_sequences (cXA-xG) data (wide format):
- Action type distributions by position
- Carry analysis
- Position-based patterns
- Action feature analysis
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def _get_repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def analyze_action_counts(df: pd.DataFrame, output_dir: Path):
    """Analyze number of actions per shot window."""

    num_actions = df["num_actions"].dropna()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Distribution
    axes[0, 0].hist(num_actions, bins=range(1, 12), edgecolor="black", alpha=0.7)
    axes[0, 0].axvline(
        num_actions.mean(), color="red", linestyle="--", label=f"Mean: {num_actions.mean():.2f}"
    )
    axes[0, 0].set_title("Actions per Shot Window")
    axes[0, 0].set_xlabel("Number of Actions")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].legend()

    # 2. Value counts
    value_counts = num_actions.value_counts().sort_index()
    axes[0, 1].bar(value_counts.index, value_counts.values, color="steelblue")
    axes[0, 1].set_title("Action Count Distribution")
    axes[0, 1].set_xlabel("Number of Actions")
    axes[0, 1].set_ylabel("Frequency")

    # 3. Cumulative
    sorted_vals = np.sort(num_actions)
    cumsum = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
    axes[1, 0].plot(sorted_vals, cumsum)
    axes[1, 0].axhline(0.5, color="red", linestyle="--", alpha=0.5)
    axes[1, 0].axhline(0.9, color="red", linestyle="--", alpha=0.5)
    axes[1, 0].set_title("Cumulative Distribution")
    axes[1, 0].set_xlabel("Number of Actions")
    axes[1, 0].set_ylabel("Cumulative %")

    # 4. Percentiles
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    pct_values = np.percentile(num_actions, percentiles)
    axes[1, 1].bar(range(len(percentiles)), pct_values, color="coral")
    axes[1, 1].set_xticks(range(len(percentiles)))
    axes[1, 1].set_xticklabels([f"P{p}" for p in percentiles])
    axes[1, 1].set_title("Action Count Percentiles")
    axes[1, 1].set_ylabel("Number of Actions")

    plt.tight_layout()
    plt.savefig(output_dir / "action_count_distribution.png", dpi=150)
    plt.close()

    # Stats
    stats_df = pd.DataFrame(
        {
            "statistic": ["count", "mean", "std", "min", "p25", "p50", "p75", "max"],
            "value": [
                len(num_actions),
                num_actions.mean(),
                num_actions.std(),
                num_actions.min(),
                num_actions.quantile(0.25),
                num_actions.median(),
                num_actions.quantile(0.75),
                num_actions.max(),
            ],
        }
    )
    stats_df.to_csv(output_dir / "action_count_stats.csv", index=False)

    return num_actions


def analyze_action_types_by_position(df: pd.DataFrame, output_dir: Path):
    """Analyze action types at each position (action1, action2, etc.)."""

    # Find max actions
    max_actions = 0
    for col in df.columns:
        if col.startswith("action") and "_type" in col:
            try:
                num = int(col.replace("action", "").replace("_type", ""))
                max_actions = max(max_actions, num)
            except ValueError:
                pass

    if max_actions == 0:
        logger.warning("No action type columns found")
        return None

    logger.info(f"Found action columns up to action{max_actions}")

    # Count action types by position
    position_stats = []
    for pos in range(1, max_actions + 1):
        type_col = f"action{pos}_type"
        if type_col not in df.columns:
            continue

        type_counts = df[type_col].value_counts()
        n_total = df[type_col].notna().sum()

        stats_row = {
            "position": pos,
            "n_total": n_total,
            "pct_of_shots": 100 * n_total / len(df),
        }

        for action_type in ["Pass", "Carry", "Dribble"]:
            count = type_counts.get(action_type, 0)
            stats_row[f"{action_type.lower()}_count"] = count
            stats_row[f"{action_type.lower()}_pct"] = 100 * count / n_total if n_total > 0 else 0

        position_stats.append(stats_row)

    position_df = pd.DataFrame(position_stats)
    position_df.to_csv(output_dir / "action_types_by_position.csv", index=False)

    logger.info("\nAction Types by Position:")
    print(position_df.to_string(index=False))

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Total actions at each position
    axes[0, 0].bar(position_df["position"], position_df["n_total"], color="steelblue")
    axes[0, 0].set_title("Actions at Each Position")
    axes[0, 0].set_xlabel("Position (1=last before shot)")
    axes[0, 0].set_ylabel("Count")

    # 2. Stacked bar - action type mix
    x = position_df["position"]
    width = 0.8

    bottom_carry = position_df["pass_pct"].values

    axes[0, 1].bar(x, position_df["pass_pct"], width, label="Pass", color="steelblue")
    axes[0, 1].bar(
        x, position_df["carry_pct"], width, bottom=bottom_carry, label="Carry", color="coral"
    )
    if "dribble_pct" in position_df.columns:
        bottom_dribble = bottom_carry + position_df["carry_pct"].values
        axes[0, 1].bar(
            x,
            position_df["dribble_pct"],
            width,
            bottom=bottom_dribble,
            label="Dribble",
            color="green",
        )
    axes[0, 1].set_title("Action Type Mix by Position")
    axes[0, 1].set_xlabel("Position")
    axes[0, 1].set_ylabel("Percentage")
    axes[0, 1].legend()

    # 3. Pass count by position
    axes[1, 0].bar(position_df["position"], position_df["pass_count"], color="steelblue", alpha=0.7)
    axes[1, 0].set_title("Passes by Position")
    axes[1, 0].set_xlabel("Position")
    axes[1, 0].set_ylabel("Count")

    # 4. Carry count by position
    axes[1, 1].bar(position_df["position"], position_df["carry_count"], color="coral", alpha=0.7)
    axes[1, 1].set_title("Carries by Position")
    axes[1, 1].set_xlabel("Position")
    axes[1, 1].set_ylabel("Count")

    plt.tight_layout()
    plt.savefig(output_dir / "action_types_by_position.png", dpi=150)
    plt.close()

    return position_df


def analyze_shot_features(df: pd.DataFrame, output_dir: Path):
    """Analyze shot-level features in action sequences."""

    shot_cols = ["shot_x", "shot_y", "shot_xg", "is_goal"]
    shot_cols = [c for c in shot_cols if c in df.columns]

    if len(shot_cols) == 0:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Shot x distribution
    if "shot_x" in df.columns:
        axes[0, 0].hist(df["shot_x"].dropna(), bins=30, edgecolor="black", alpha=0.7)
        axes[0, 0].set_title("Shot X Distribution")
        axes[0, 0].set_xlabel("Shot X")
        axes[0, 0].set_ylabel("Count")

    # 2. Shot y distribution
    if "shot_y" in df.columns:
        axes[0, 1].hist(df["shot_y"].dropna(), bins=30, edgecolor="black", alpha=0.7, color="coral")
        axes[0, 1].set_title("Shot Y Distribution")
        axes[0, 1].set_xlabel("Shot Y")
        axes[0, 1].set_ylabel("Count")

    # 3. Shot xG distribution
    if "shot_xg" in df.columns:
        axes[1, 0].hist(
            df["shot_xg"].dropna(), bins=50, edgecolor="black", alpha=0.7, color="green"
        )
        axes[1, 0].set_title("Shot xG Distribution")
        axes[1, 0].set_xlabel("xG")
        axes[1, 0].set_ylabel("Count")

    # 4. Shot location heatmap
    if "shot_x" in df.columns and "shot_y" in df.columns:
        h = axes[1, 1].hist2d(df["shot_x"], df["shot_y"], bins=20, cmap="YlOrRd")
        axes[1, 1].set_title("Shot Location Heatmap")
        axes[1, 1].set_xlabel("Shot X")
        axes[1, 1].set_ylabel("Shot Y")
        plt.colorbar(h[3], ax=axes[1, 1])

    plt.tight_layout()
    plt.savefig(output_dir / "shot_features.png", dpi=150)
    plt.close()

    # Stats
    shot_stats = []
    for col in shot_cols:
        if col == "is_goal":
            shot_stats.append(
                {
                    "feature": col,
                    "mean": df[col].mean(),
                    "sum": df[col].sum(),
                }
            )
        else:
            series = df[col].dropna()
            shot_stats.append(
                {
                    "feature": col,
                    "mean": series.mean(),
                    "std": series.std(),
                    "min": series.min(),
                    "max": series.max(),
                }
            )

    shot_stats_df = pd.DataFrame(shot_stats)
    shot_stats_df.to_csv(output_dir / "shot_feature_stats.csv", index=False)

    return shot_stats_df


def analyze_action_features_by_position(df: pd.DataFrame, output_dir: Path):
    """Analyze action features (end_x, end_y, etc.) by position."""

    # Find max actions
    max_actions = 0
    for col in df.columns:
        if col.startswith("action") and "_type" in col:
            try:
                num = int(col.replace("action", "").replace("_type", ""))
                max_actions = max(max_actions, num)
            except ValueError:
                pass

    if max_actions == 0:
        return None

    # Analyze key features by position
    features = ["end_x", "end_y", "start_x", "start_y"]

    position_features = []
    for pos in range(1, max_actions + 1):
        stats_row = {"position": pos}

        for feat in features:
            col = f"action{pos}_{feat}"
            if col in df.columns:
                series = df[col].dropna()
                stats_row[f"{feat}_mean"] = series.mean() if len(series) > 0 else np.nan
                stats_row[f"{feat}_std"] = series.std() if len(series) > 0 else np.nan

        position_features.append(stats_row)

    position_features_df = pd.DataFrame(position_features)
    position_features_df.to_csv(output_dir / "action_features_by_position.csv", index=False)

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    if "end_x_mean" in position_features_df.columns:
        axes[0, 0].plot(
            position_features_df["position"],
            position_features_df["end_x_mean"],
            marker="o",
            linewidth=2,
        )
        axes[0, 0].set_title("Mean End X by Position")
        axes[0, 0].set_xlabel("Position")
        axes[0, 0].set_ylabel("Mean End X")

    if "end_y_mean" in position_features_df.columns:
        axes[0, 1].plot(
            position_features_df["position"],
            position_features_df["end_y_mean"],
            marker="o",
            linewidth=2,
            color="coral",
        )
        axes[0, 1].set_title("Mean End Y by Position")
        axes[0, 1].set_xlabel("Position")
        axes[0, 1].set_ylabel("Mean End Y")

    if "start_x_mean" in position_features_df.columns:
        axes[1, 0].plot(
            position_features_df["position"],
            position_features_df["start_x_mean"],
            marker="o",
            linewidth=2,
            color="green",
        )
        axes[1, 0].set_title("Mean Start X by Position")
        axes[1, 0].set_xlabel("Position")
        axes[1, 0].set_ylabel("Mean Start X")

    if "start_y_mean" in position_features_df.columns:
        axes[1, 1].plot(
            position_features_df["position"],
            position_features_df["start_y_mean"],
            marker="o",
            linewidth=2,
            color="purple",
        )
        axes[1, 1].set_title("Mean Start Y by Position")
        axes[1, 1].set_xlabel("Position")
        axes[1, 1].set_ylabel("Mean Start Y")

    plt.tight_layout()
    plt.savefig(output_dir / "action_features_by_position.png", dpi=150)
    plt.close()

    return position_features_df


def analyze_goal_vs_nongoal(df: pd.DataFrame, output_dir: Path):
    """Compare action sequences leading to goals vs non-goals."""

    goals = df[df["is_goal"] == 1]
    non_goals = df[df["is_goal"] == 0]

    comparison_rows = []

    # Compare num_actions
    comparison_rows.append(
        {
            "feature": "num_actions",
            "goal_mean": goals["num_actions"].mean(),
            "nongoal_mean": non_goals["num_actions"].mean(),
            "diff": goals["num_actions"].mean() - non_goals["num_actions"].mean(),
        }
    )

    # Compare shot_xg
    if "shot_xg" in df.columns:
        comparison_rows.append(
            {
                "feature": "shot_xg",
                "goal_mean": goals["shot_xg"].mean(),
                "nongoal_mean": non_goals["shot_xg"].mean(),
                "diff": goals["shot_xg"].mean() - non_goals["shot_xg"].mean(),
            }
        )

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(output_dir / "goal_vs_nongoal_comparison.csv", index=False)

    logger.info("\nGoal vs Non-Goal Comparison:")
    print(comparison_df.to_string(index=False))

    return comparison_df


def run_phase4_eda(output_dir: Path = None) -> Dict[str, Any]:
    """Run Phase 4 EDA: Action Sequences Analysis."""

    repo_root = _get_repo_root()
    if output_dir is None:
        output_dir = repo_root / "outputs" / "analysis" / "cxa" / "eda" / "phase4_actions"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("PHASE 4 EDA: Action Sequences Analysis")
    logger.info("=" * 60)

    # Load data
    actions = pd.read_parquet(repo_root / "feature_store" / "cxa" / "action_sequences.parquet")

    logger.info(f"Loaded {len(actions):,} action sequences (wide format)")
    logger.info(f"Columns: {list(actions.columns[:15])}...")

    results = {}

    # 1. Action Count Analysis
    logger.info("\n--- 1. Action Count Analysis ---")
    num_actions = analyze_action_counts(actions, output_dir)
    results["num_actions"] = num_actions

    logger.info(f"Shot windows: {len(num_actions):,}")
    logger.info(f"Mean actions per window: {num_actions.mean():.2f}")

    # 2. Action Types by Position
    logger.info("\n--- 2. Action Types by Position ---")
    type_stats = analyze_action_types_by_position(actions, output_dir)
    results["type_stats"] = type_stats

    # 3. Shot Features
    logger.info("\n--- 3. Shot Features ---")
    shot_stats = analyze_shot_features(actions, output_dir)
    results["shot_stats"] = shot_stats

    # 4. Action Features by Position
    logger.info("\n--- 4. Action Features by Position ---")
    position_features = analyze_action_features_by_position(actions, output_dir)
    results["position_features"] = position_features

    # 5. Goal vs Non-Goal
    logger.info("\n--- 5. Goal vs Non-Goal Comparison ---")
    comparison = analyze_goal_vs_nongoal(actions, output_dir)
    results["comparison"] = comparison

    # 6. Summary
    logger.info("\n--- 6. Summary ---")
    summary = {
        "total_shot_windows": len(actions),
        "mean_actions_per_window": float(num_actions.mean()),
        "total_goals": int(actions["is_goal"].sum()),
        "conversion_rate": float(100 * actions["is_goal"].mean()),
        "total_xg": float(actions["shot_xg"].sum()) if "shot_xg" in actions.columns else None,
    }

    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(output_dir / "action_summary.csv", index=False)

    for k, v in summary.items():
        logger.info(f"  {k}: {v}")

    logger.info(f"\nPhase 4 EDA complete. Outputs saved to {output_dir}")

    return results


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    run_phase4_eda()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
