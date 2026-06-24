"""Phase 1 EDA: Passes Analysis (Extended).

Comprehensive EDA for passes data:
- Feature distributions (histograms, box plots)
- Correlation analysis
- Class imbalance (assist rate)
- Spatial analysis (pass destinations)
- Missing data patterns
- Outlier detection
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any, List

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

logger = logging.getLogger(__name__)


def _get_repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def plot_feature_distributions(df: pd.DataFrame, numeric_cols: List[str], output_dir: Path):
    """Plot histograms for numeric features."""
    n_cols = min(3, len(numeric_cols))
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = np.array(axes).flatten() if n_rows * n_cols > 1 else [axes]

    for i, col in enumerate(numeric_cols):
        if i < len(axes):
            ax = axes[i]
            data = df[col].dropna()

            ax.hist(data, bins=50, edgecolor="black", alpha=0.7)
            ax.axvline(data.mean(), color="red", linestyle="--", label=f"Mean: {data.mean():.2f}")
            ax.axvline(
                data.median(), color="green", linestyle="--", label=f"Median: {data.median():.2f}"
            )
            ax.set_title(f"{col}\n(n={len(data):,}, null={df[col].isna().sum():,})")
            ax.set_xlabel(col)
            ax.set_ylabel("Count")
            ax.legend(fontsize=8)

    # Hide empty subplots
    for i in range(len(numeric_cols), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / "feature_distributions.png", dpi=150)
    plt.close()


def plot_correlation_matrix(df: pd.DataFrame, numeric_cols: List[str], output_dir: Path):
    """Plot correlation heatmap."""
    corr = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        ax=ax,
        annot_kws={"size": 8},
    )
    ax.set_title("Feature Correlation Matrix")

    plt.tight_layout()
    plt.savefig(output_dir / "correlation_matrix.png", dpi=150)
    plt.close()

    # Save high correlations
    high_corr = []
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            if abs(corr.iloc[i, j]) > 0.5:
                high_corr.append(
                    {
                        "feature_1": corr.columns[i],
                        "feature_2": corr.columns[j],
                        "correlation": round(corr.iloc[i, j], 3),
                    }
                )

    if high_corr:
        pd.DataFrame(high_corr).sort_values("correlation", key=abs, ascending=False).to_csv(
            output_dir / "high_correlations.csv", index=False
        )

    return corr


def plot_class_imbalance(df: pd.DataFrame, target_col: str, output_dir: Path):
    """Analyze class imbalance."""
    if target_col not in df.columns:
        logger.warning(f"Target column {target_col} not found")
        return

    counts = df[target_col].value_counts()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Bar chart
    axes[0].bar(counts.index.astype(str), counts.values, color=["steelblue", "coral"])
    axes[0].set_title(f"Class Distribution: {target_col}")
    axes[0].set_xlabel(target_col)
    axes[0].set_ylabel("Count")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + counts.max() * 0.01, f"{v:,}", ha="center")

    # Pie chart
    axes[1].pie(
        counts.values,
        labels=[f"{k}\n({v:,}, {100*v/len(df):.2f}%)" for k, v in counts.items()],
        autopct="",
        colors=["steelblue", "coral"],
    )
    axes[1].set_title(f"Class Proportions: {target_col}")

    plt.tight_layout()
    plt.savefig(output_dir / "class_imbalance.png", dpi=150)
    plt.close()

    # Log stats
    logger.info(f"\nClass Imbalance for {target_col}:")
    for k, v in counts.items():
        logger.info(f"  {k}: {v:,} ({100*v/len(df):.4f}%)")

    imbalance_ratio = counts.max() / counts.min() if counts.min() > 0 else float("inf")
    logger.info(f"  Imbalance ratio: {imbalance_ratio:.1f}:1")


def plot_spatial_heatmap(
    df: pd.DataFrame, x_col: str, y_col: str, title: str, output_dir: Path, filename: str
):
    """Plot spatial heatmap on pitch coordinates."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Filter valid coordinates
    valid = df[[x_col, y_col]].dropna()

    # 2D histogram
    h = ax.hist2d(valid[x_col], valid[y_col], bins=30, cmap="YlOrRd")
    plt.colorbar(h[3], ax=ax, label="Count")

    # Pitch outline (StatsBomb coordinates: 120 x 80)
    ax.axvline(60, color="white", linestyle="--", alpha=0.5)  # Halfway
    ax.axvline(102, color="white", linestyle="--", alpha=0.5)  # Penalty area
    ax.axhline(18, color="white", linestyle="--", alpha=0.5)  # Box edge
    ax.axhline(62, color="white", linestyle="--", alpha=0.5)  # Box edge

    ax.set_xlim(0, 120)
    ax.set_ylim(0, 80)
    ax.set_xlabel("X (0=own goal, 120=opponent goal)")
    ax.set_ylabel("Y (0-80)")
    ax.set_title(title)
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=150)
    plt.close()


def plot_boxplots_by_category(
    df: pd.DataFrame, numeric_col: str, category_col: str, output_dir: Path, filename: str
):
    """Plot boxplots of numeric feature by category."""
    fig, ax = plt.subplots(figsize=(10, 6))

    df.boxplot(column=numeric_col, by=category_col, ax=ax)
    ax.set_title(f"{numeric_col} by {category_col}")
    ax.set_xlabel(category_col)
    ax.set_ylabel(numeric_col)
    plt.suptitle("")  # Remove automatic title

    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=150)
    plt.close()


def detect_outliers(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    """Detect outliers using IQR method."""
    outlier_info = []

    for col in numeric_cols:
        data = df[col].dropna()
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = data[(data < lower_bound) | (data > upper_bound)]

        outlier_info.append(
            {
                "column": col,
                "n_outliers": len(outliers),
                "pct_outliers": 100 * len(outliers) / len(data),
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "min_value": data.min(),
                "max_value": data.max(),
            }
        )

    return pd.DataFrame(outlier_info)


def analyze_by_target(df: pd.DataFrame, target_col: str, feature_cols: List[str], output_dir: Path):
    """Compare feature distributions for target=0 vs target=1."""
    if target_col not in df.columns:
        return

    comparison_rows = []

    for col in feature_cols:
        if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
            continue

        group_0 = df[df[target_col] == 0][col].dropna()
        group_1 = df[df[target_col] == 1][col].dropna()

        if len(group_0) > 0 and len(group_1) > 0:
            # T-test
            t_stat, p_value = stats.ttest_ind(group_0, group_1)

            comparison_rows.append(
                {
                    "feature": col,
                    "mean_non_assist": group_0.mean(),
                    "mean_assist": group_1.mean(),
                    "diff": group_1.mean() - group_0.mean(),
                    "t_statistic": t_stat,
                    "p_value": p_value,
                    "significant": p_value < 0.05,
                }
            )

    if comparison_rows:
        comparison_df = pd.DataFrame(comparison_rows).sort_values("p_value")
        comparison_df.to_csv(output_dir / "feature_comparison_by_target.csv", index=False)

        logger.info("\nFeature Comparison (assist vs non-assist):")
        print(comparison_df.head(15).to_string(index=False))


def run_phase1_eda(output_dir: Path = None) -> Dict[str, Any]:
    """Run Phase 1 EDA: Comprehensive Passes Analysis."""

    repo_root = _get_repo_root()
    if output_dir is None:
        output_dir = repo_root / "outputs" / "analysis" / "cxa" / "eda" / "phase1_passes"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("PHASE 1 EDA: Passes Analysis")
    logger.info("=" * 60)

    # Load passes
    passes = pd.read_parquet(repo_root / "feature_store" / "cxa" / "passes.parquet")
    logger.info(f"Loaded {len(passes):,} passes")

    # Define feature groups
    numeric_features = [
        "start_x",
        "start_y",
        "end_x",
        "end_y",
        "pass_length",
        "pass_angle",
        "start_xt",
        "end_xt",
        "xt_delta",
    ]
    boolean_features = [
        "is_cross",
        "is_through_ball",
        "is_into_box",
        "is_progressive",
        "is_complete",
        "under_pressure",
        "is_set_piece",
        "is_final_third",
    ]

    # Filter to columns that exist
    numeric_features = [c for c in numeric_features if c in passes.columns]
    boolean_features = [c for c in boolean_features if c in passes.columns]

    results = {}

    # 1. Basic Stats
    logger.info("\n--- 1. Basic Statistics ---")
    stats_df = passes[numeric_features].describe().T
    stats_df["null_count"] = passes[numeric_features].isna().sum()
    stats_df["null_pct"] = 100 * passes[numeric_features].isna().mean()
    stats_df.to_csv(output_dir / "numeric_stats.csv")
    print(stats_df.round(2).to_string())
    results["numeric_stats"] = stats_df

    # 2. Boolean Feature Frequencies
    logger.info("\n--- 2. Boolean Feature Frequencies ---")
    bool_stats = []
    for col in boolean_features:
        true_count = passes[col].sum()
        bool_stats.append(
            {
                "feature": col,
                "true_count": true_count,
                "true_pct": 100 * true_count / len(passes),
                "false_count": len(passes) - true_count,
            }
        )
    bool_df = pd.DataFrame(bool_stats)
    bool_df.to_csv(output_dir / "boolean_frequencies.csv", index=False)
    print(bool_df.to_string(index=False))
    results["boolean_stats"] = bool_df

    # 3. Feature Distributions
    logger.info("\n--- 3. Plotting Feature Distributions ---")
    plot_feature_distributions(passes, numeric_features, output_dir)

    # 4. Correlation Analysis
    logger.info("\n--- 4. Correlation Analysis ---")
    all_numeric = numeric_features + [c for c in boolean_features if c in passes.columns]
    corr = plot_correlation_matrix(passes, all_numeric, output_dir)
    results["correlation_matrix"] = corr

    # 5. Class Imbalance (create is_assist if not exists)
    logger.info("\n--- 5. Class Imbalance Analysis ---")
    if "is_assist" not in passes.columns:
        # Load sequences to get assist pass IDs
        sequences = pd.read_parquet(repo_root / "feature_store" / "cxa" / "sequences.parquet")
        assist_ids = set()
        for col in ["pass1_id", "pass2_id", "pass3_id"]:
            if col in sequences.columns:
                # Get assists (pass1 is the final pass, which is the assist for goals)
                if col == "pass1_id":
                    goal_seqs = (
                        sequences[sequences["is_goal"].fillna(False).astype(bool)]
                        if "is_goal" in sequences.columns
                        else sequences
                    )
                    assist_ids.update(goal_seqs[col].dropna().astype(int).tolist())

        passes["is_assist"] = passes["pass_id"].isin(assist_ids).astype(int)
        logger.info(f"Created is_assist column: {passes['is_assist'].sum()} assists")

    plot_class_imbalance(passes, "is_assist", output_dir)

    # 6. Spatial Analysis
    logger.info("\n--- 6. Spatial Analysis ---")
    plot_spatial_heatmap(
        passes, "end_x", "end_y", "All Pass Destinations", output_dir, "spatial_all_passes.png"
    )

    if "is_assist" in passes.columns:
        assists = passes[passes["is_assist"] == 1]
        if len(assists) > 0:
            plot_spatial_heatmap(
                assists,
                "end_x",
                "end_y",
                "Assist Pass Destinations",
                output_dir,
                "spatial_assists.png",
            )

    # 7. Outlier Detection
    logger.info("\n--- 7. Outlier Detection ---")
    outliers_df = detect_outliers(passes, numeric_features)
    outliers_df.to_csv(output_dir / "outliers.csv", index=False)
    print(outliers_df.to_string(index=False))
    results["outliers"] = outliers_df

    # 8. Feature Comparison by Target
    logger.info("\n--- 8. Feature Comparison (Assist vs Non-Assist) ---")
    analyze_by_target(passes, "is_assist", numeric_features + boolean_features, output_dir)

    # 9. Assist Rate by Categories
    logger.info("\n--- 9. Assist Rate by Categories ---")

    # By pass type
    if "pass_type" in passes.columns:
        by_type = (
            passes.groupby("pass_type")
            .agg(total=("pass_id", "count"), assists=("is_assist", "sum"))
            .reset_index()
        )
        by_type["assist_rate"] = 100 * by_type["assists"] / by_type["total"]
        by_type = by_type.sort_values("assist_rate", ascending=False)
        by_type.to_csv(output_dir / "assist_rate_by_type.csv", index=False)
        logger.info("\nAssist rate by pass type:")
        print(by_type.head(10).to_string(index=False))

    # By zone
    if "end_x" in passes.columns:
        passes["zone"] = pd.cut(
            passes["end_x"], bins=[0, 40, 80, 120], labels=["Defensive", "Middle", "Attacking"]
        )
        by_zone = (
            passes.groupby("zone")
            .agg(total=("pass_id", "count"), assists=("is_assist", "sum"))
            .reset_index()
        )
        by_zone["assist_rate"] = 100 * by_zone["assists"] / by_zone["total"]
        by_zone.to_csv(output_dir / "assist_rate_by_zone.csv", index=False)
        logger.info("\nAssist rate by zone:")
        print(by_zone.to_string(index=False))

    logger.info(f"\nPhase 1 EDA complete. Outputs saved to {output_dir}")

    return results


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    run_phase1_eda()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
