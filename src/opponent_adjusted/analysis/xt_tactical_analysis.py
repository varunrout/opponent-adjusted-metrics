"""
Enhanced xT spatial analysis by tactical and on-pitch positions.

Extends basic spatial analysis to include:
- Tactical position (assigned role from team lineup)
- On-pitch position (actual location during shot creation)
- Formation-specific analysis
- Tactical flexibility metrics (players playing out of assigned roles)
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle

from opponent_adjusted.analysis.xt_model import xTModel

logger = logging.getLogger(__name__)


class TacticalPositionAnalyzer:
    """Analyzes xT by tactical vs on-pitch positions."""

    def __init__(self, xt_model: xTModel):
        """Initialize analyzer.

        Args:
            xt_model: Trained xTModel instance
        """
        self.xt_model = xt_model

    def analyze_xt_by_tactical_position(
        self, enriched_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Analyze xT generation by tactical position.

        Args:
            enriched_df: Baseline data with tactical_position column

        Returns:
            Tuple of (analysis_df, summary_stats)
        """
        analysis = (
            enriched_df.groupby("tactical_position")
            .agg(
                {
                    "xa_plus": ["count", "sum", "mean", "std", "min", "max"],
                    "pass_end_x": "mean",
                    "pass_end_y": "mean",
                }
            )
            .round(4)
        )

        analysis.columns = ["_".join(col).strip() for col in analysis.columns.values]
        analysis = analysis.rename(
            columns={
                "xa_plus_count": "shots",
                "xa_plus_sum": "total_xt",
                "xa_plus_mean": "avg_xt",
                "xa_plus_std": "std_xt",
                "xa_plus_min": "min_xt",
                "xa_plus_max": "max_xt",
                "pass_end_x_mean": "avg_x",
                "pass_end_y_mean": "avg_y",
            }
        )

        analysis = analysis.reset_index()

        # Add threat level classification
        analysis["threat_level"] = analysis["avg_xt"].apply(
            self.xt_model.get_threat_level
        )
        analysis["percentile_rank"] = analysis["avg_xt"].apply(
            self.xt_model.get_xt_percentile
        )

        summary_stats = {
            "positions_analyzed": len(analysis),
            "position_details": analysis.to_dict("records"),
            "highest_xt_position": analysis.loc[analysis["total_xt"].idxmax()].to_dict(),
            "highest_avg_xt_position": analysis.loc[analysis["avg_xt"].idxmax()].to_dict(),
        }

        return analysis, summary_stats

    def analyze_xt_by_on_pitch_position(
        self, enriched_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Analyze xT generation by on-pitch position (actual location).

        Args:
            enriched_df: Baseline data with on_pitch_position column

        Returns:
            Tuple of (analysis_df, summary_stats)
        """
        analysis = (
            enriched_df.groupby("on_pitch_position")
            .agg(
                {
                    "xa_plus": ["count", "sum", "mean", "std", "min", "max"],
                    "pass_end_x": "mean",
                    "pass_end_y": "mean",
                }
            )
            .round(4)
        )

        analysis.columns = ["_".join(col).strip() for col in analysis.columns.values]
        analysis = analysis.rename(
            columns={
                "xa_plus_count": "shots",
                "xa_plus_sum": "total_xt",
                "xa_plus_mean": "avg_xt",
                "xa_plus_std": "std_xt",
                "xa_plus_min": "min_xt",
                "xa_plus_max": "max_xt",
                "pass_end_x_mean": "avg_x",
                "pass_end_y_mean": "avg_y",
            }
        )

        analysis = analysis.reset_index()

        # Add threat level classification
        analysis["threat_level"] = analysis["avg_xt"].apply(
            self.xt_model.get_threat_level
        )
        analysis["percentile_rank"] = analysis["avg_xt"].apply(
            self.xt_model.get_xt_percentile
        )

        summary_stats = {
            "positions_analyzed": len(analysis),
            "position_details": analysis.to_dict("records"),
            "highest_xt_position": analysis.loc[analysis["total_xt"].idxmax()].to_dict(),
            "highest_avg_xt_position": analysis.loc[analysis["avg_xt"].idxmax()].to_dict(),
        }

        return analysis, summary_stats

    def analyze_tactical_vs_onpitch_mismatch(
        self, enriched_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Analyze cases where players play out of assigned positions.

        Identifies "tactical flexibility" - players assigned to one position
        but creating shots from another part of the pitch.

        Args:
            enriched_df: Baseline data with both position columns

        Returns:
            Tuple of (mismatch_df, mismatch_stats)
        """
        # Filter rows with both position types known
        valid_df = enriched_df[
            (enriched_df["tactical_position"].notna())
            & (enriched_df["tactical_position"] != "Unknown")
        ].copy()

        # Identify mismatches
        valid_df["position_match"] = (
            valid_df["tactical_position"] == valid_df["on_pitch_position"]
        )
        valid_df["tactical_flexibility"] = ~valid_df["position_match"]

        # Analyze mismatch patterns
        mismatch_analysis = (
            valid_df[valid_df["tactical_flexibility"]]
            .groupby(["tactical_position", "on_pitch_position"])
            .agg(
                {
                    "xa_plus": ["count", "sum", "mean"],
                    "pass_end_x": "mean",
                }
            )
            .round(4)
        )

        mismatch_analysis.columns = ["_".join(col).strip() for col in mismatch_analysis.columns.values]
        mismatch_analysis = mismatch_analysis.rename(
            columns={
                "xa_plus_count": "shots",
                "xa_plus_sum": "total_xt",
                "xa_plus_mean": "avg_xt",
                "pass_end_x_mean": "avg_x",
            }
        )
        mismatch_analysis = mismatch_analysis.reset_index()

        stats = {
            "total_passes_analyzed": len(valid_df),
            "mismatches": len(valid_df[valid_df["tactical_flexibility"]]),
            "mismatch_pct": 100
            * len(valid_df[valid_df["tactical_flexibility"]])
            / len(valid_df),
            "mismatch_xt_total": valid_df[valid_df["tactical_flexibility"]]["xa_plus"].sum(),
            "mismatch_xt_avg": valid_df[valid_df["tactical_flexibility"]]["xa_plus"].mean(),
            "match_xt_total": valid_df[~valid_df["tactical_flexibility"]]["xa_plus"].sum(),
            "match_xt_avg": valid_df[~valid_df["tactical_flexibility"]]["xa_plus"].mean(),
            "flexibility_details": mismatch_analysis.to_dict("records"),
        }

        return mismatch_analysis, stats

    def analyze_xt_by_formation(
        self, enriched_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Analyze xT generation by formation.

        Args:
            enriched_df: Baseline data with formation column

        Returns:
            Tuple of (formation_analysis_df, summary_stats)
        """
        analysis = (
            enriched_df[enriched_df["formation"] != "Unknown"]
            .groupby("formation")
            .agg(
                {
                    "xa_plus": ["count", "sum", "mean", "std"],
                    "pass_end_x": "mean",
                }
            )
            .round(4)
        )

        if len(analysis) == 0:
            logger.warning("No formation data available")
            return pd.DataFrame(), {}

        analysis.columns = ["_".join(col).strip() for col in analysis.columns.values]
        analysis = analysis.rename(
            columns={
                "xa_plus_count": "shots",
                "xa_plus_sum": "total_xt",
                "xa_plus_mean": "avg_xt",
                "xa_plus_std": "std_xt",
                "pass_end_x_mean": "avg_x",
            }
        )
        analysis = analysis.sort_values("total_xt", ascending=False)
        analysis = analysis.reset_index()

        analysis["threat_level"] = analysis["avg_xt"].apply(
            self.xt_model.get_threat_level
        )

        summary_stats = {
            "formations_analyzed": len(analysis),
            "formation_details": analysis.to_dict("records"),
            "most_frequent_formation": analysis.iloc[0].to_dict(),
            "highest_threat_formation": analysis.loc[analysis["avg_xt"].idxmax()].to_dict(),
        }

        return analysis, summary_stats

    def plot_tactical_vs_onpitch_comparison(
        self,
        tactical_df: pd.DataFrame,
        onpitch_df: pd.DataFrame,
        output_path: Optional[Path] = None,
    ) -> None:
        """Create comparison visualization of tactical vs on-pitch positions.

        Args:
            tactical_df: Results from analyze_xt_by_tactical_position
            onpitch_df: Results from analyze_xt_by_on_pitch_position
            output_path: Path to save figure (optional)
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Tactical positions
        tactical_df_sorted = tactical_df.sort_values("avg_xt", ascending=True)
        colors_tactical = [
            "red" if level == "High" else "orange" if level == "Medium" else "green"
            for level in tactical_df_sorted["threat_level"]
        ]
        axes[0].barh(tactical_df_sorted["tactical_position"], tactical_df_sorted["avg_xt"],
                     color=colors_tactical)
        axes[0].set_xlabel("Average xT")
        axes[0].set_title("xT by Tactical Position\n(Assigned Role)")
        axes[0].grid(axis="x", alpha=0.3)

        # On-pitch positions
        onpitch_df_sorted = onpitch_df.sort_values("avg_xt", ascending=True)
        colors_onpitch = [
            "red" if level == "High" else "orange" if level == "Medium" else "green"
            for level in onpitch_df_sorted["threat_level"]
        ]
        axes[1].barh(onpitch_df_sorted["on_pitch_position"], onpitch_df_sorted["avg_xt"],
                     color=colors_onpitch)
        axes[1].set_xlabel("Average xT")
        axes[1].set_title("xT by On-Pitch Position\n(Actual Location)")
        axes[1].grid(axis="x", alpha=0.3)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved comparison plot to {output_path}")
        else:
            plt.show()

        plt.close()

    def plot_formation_comparison(
        self,
        formation_df: pd.DataFrame,
        output_path: Optional[Path] = None,
    ) -> None:
        """Create visualization of xT by formation.

        Args:
            formation_df: Results from analyze_xt_by_formation
            output_path: Path to save figure (optional)
        """
        if len(formation_df) == 0:
            logger.warning("No formation data to plot")
            return

        fig, ax = plt.subplots(figsize=(12, 6))

        formation_sorted = formation_df.sort_values("avg_xt", ascending=True)
        colors = [
            "red" if level == "High" else "orange" if level == "Medium" else "green"
            for level in formation_sorted["threat_level"]
        ]

        ax.barh(formation_sorted["formation"], formation_sorted["avg_xt"], color=colors)
        ax.set_xlabel("Average xT")
        ax.set_title("xT Generation by Formation")
        ax.grid(axis="x", alpha=0.3)

        # Add sample size labels
        for i, (idx, row) in enumerate(formation_sorted.iterrows()):
            ax.text(row["avg_xt"], i, f" n={int(row['shots'])}", va="center", fontsize=9)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved formation plot to {output_path}")
        else:
            plt.show()

        plt.close()

    def plot_tactical_flexibility(
        self,
        enriched_df: pd.DataFrame,
        output_path: Optional[Path] = None,
    ) -> None:
        """Create heatmap of tactical position changes.

        Shows flows from assigned position to actual position where xT is created.

        Args:
            enriched_df: Baseline data with both position columns
            output_path: Path to save figure (optional)
        """
        # Filter valid data
        valid_df = enriched_df[
            (enriched_df["tactical_position"].notna())
            & (enriched_df["tactical_position"] != "Unknown")
        ]

        # Create pivot for heatmap
        pivot = (
            valid_df.pivot_table(
                index="tactical_position",
                columns="on_pitch_position",
                values="xa_plus",
                aggfunc="sum",
            )
            .fillna(0)
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".2f",
            cmap="YlOrRd",
            ax=ax,
            cbar_kws={"label": "Total xT"},
        )
        ax.set_title("xT Flow: Tactical Position → On-Pitch Position\n(Tactical Flexibility)")
        ax.set_xlabel("On-Pitch Position (Actual Location)")
        ax.set_ylabel("Tactical Position (Assigned Role)")

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved tactical flexibility heatmap to {output_path}")
        else:
            plt.show()

        plt.close()


def run_tactical_position_analysis(
    enriched_baseline_path: Path,
    output_dir: Path,
) -> None:
    """Run comprehensive tactical position analysis.

    Args:
        enriched_baseline_path: Path to enriched baseline CSV
        output_dir: Path to save analysis outputs
    """
    logger.info("Loading enriched baseline data...")
    enriched_df = pd.read_csv(enriched_baseline_path)

    # Initialize xT model
    xt_model = xTModel()

    # Create analyzer
    analyzer = TacticalPositionAnalyzer(xt_model)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Analyze by tactical position
    logger.info("Analyzing by tactical position...")
    tactical_df, tactical_stats = analyzer.analyze_xt_by_tactical_position(enriched_df)
    tactical_df.to_csv(
        output_dir / "cxa_xt_by_tactical_position.csv", index=False
    )
    logger.info(f"  Tactical positions analyzed: {len(tactical_df)}")
    logger.info(f"  Highest xT position: {tactical_stats['highest_xt_position']}")

    # Analyze by on-pitch position
    logger.info("Analyzing by on-pitch position...")
    onpitch_df, onpitch_stats = analyzer.analyze_xt_by_on_pitch_position(enriched_df)
    onpitch_df.to_csv(
        output_dir / "cxa_xt_by_onpitch_position.csv", index=False
    )
    logger.info(f"  On-pitch positions analyzed: {len(onpitch_df)}")

    # Analyze tactical vs on-pitch mismatch
    logger.info("Analyzing tactical flexibility...")
    mismatch_df, mismatch_stats = analyzer.analyze_tactical_vs_onpitch_mismatch(
        enriched_df
    )
    mismatch_df.to_csv(
        output_dir / "cxa_xt_tactical_flexibility.csv", index=False
    )
    logger.info(f"  Mismatch percentage: {mismatch_stats['mismatch_pct']:.1f}%")
    logger.info(f"  Mismatch avg xT: {mismatch_stats['mismatch_xt_avg']:.4f}")
    logger.info(f"  Match avg xT: {mismatch_stats['match_xt_avg']:.4f}")

    # Analyze by formation
    logger.info("Analyzing by formation...")
    formation_df, formation_stats = analyzer.analyze_xt_by_formation(enriched_df)
    if not formation_df.empty:
        formation_df.to_csv(
            output_dir / "cxa_xt_by_formation.csv", index=False
        )
        logger.info(f"  Formations analyzed: {len(formation_df)}")

    # Create visualizations
    logger.info("Creating visualizations...")
    analyzer.plot_tactical_vs_onpitch_comparison(
        tactical_df, onpitch_df,
        output_dir / "cxa_tactical_vs_onpitch_comparison.png"
    )

    if not formation_df.empty:
        analyzer.plot_formation_comparison(
            formation_df,
            output_dir / "cxa_xt_by_formation.png"
        )

    analyzer.plot_tactical_flexibility(
        enriched_df,
        output_dir / "cxa_tactical_flexibility_heatmap.png"
    )

    logger.info(f"Analysis complete! Outputs saved to {output_dir}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from opponent_adjusted.config import get_config

    config = get_config()
    run_tactical_position_analysis(
        Path("outputs/analysis/cxa/csv/cxa_baselines_pass_level_enriched.csv"),
        Path("outputs/analysis/cxa/csv"),
    )
