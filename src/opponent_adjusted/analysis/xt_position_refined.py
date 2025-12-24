"""
Refined xT analysis by on-pitch position.

Analyzes xT generation by:
1. On-pitch position (location-based inference)
2. Comparison between position categories
3. Detailed breakdown by thirds of the pitch
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from opponent_adjusted.features.xt_model import xTModel

logger = logging.getLogger(__name__)


class RefinedPositionAnalyzer:
    """Analyzes xT by position using detailed location-based inference."""

    def __init__(self, xt_model: xTModel):
        """Initialize analyzer.

        Args:
            xt_model: Trained xTModel instance
        """
        self.xt_model = xt_model

    @staticmethod
    def infer_detailed_position(x: Optional[float], y: Optional[float]) -> str:
        """Infer detailed position from x, y coordinates.

        Divides field into 9 zones:
        - Defensive third (0-40x): Defensive back, Defensive mid, Defensive wing
        - Middle third (40-80x): Central mid, Attacking mid, Wide mid
        - Attacking third (80-120x): Attacking back, Striker, Wing

        Args:
            x: X-coordinate (0-120)
            y: Y-coordinate (0-80)

        Returns:
            Detailed position
        """
        if pd.isna(x) or pd.isna(y):
            return "Unknown"

        x = float(x)
        y = float(y)

        # Determine third
        if x < 40:
            third = "Defensive"
        elif x < 80:
            third = "Midfield"
        else:
            third = "Attacking"

        # Determine position in third (wing, central, or field specific)
        if y < 27:
            return f"{third} (Wing)"
        elif y > 53:
            return f"{third} (Wing)"
        else:
            return f"{third} (Central)"

    @staticmethod
    def infer_broad_position(x: Optional[float]) -> str:
        """Infer broad position from x-coordinate.

        Args:
            x: X-coordinate (0-120)

        Returns:
            Position: Defender, Midfielder, Forward, or Unknown
        """
        if pd.isna(x):
            return "Unknown"

        x = float(x)
        if x < 40:
            return "Defender"
        elif x < 80:
            return "Midfielder"
        else:
            return "Forward"

    def enrich_baseline_with_positions(self, baseline_df: pd.DataFrame) -> pd.DataFrame:
        """Enrich baseline with position data.

        Args:
            baseline_df: Baseline pass-level data

        Returns:
            Enriched DataFrame with position columns
        """
        enriched = baseline_df.copy()

        # Add positions
        enriched["on_pitch_position"] = enriched["pass_end_x"].apply(
            self.infer_broad_position
        )
        enriched["detailed_position"] = enriched.apply(
            lambda row: self.infer_detailed_position(row["pass_end_x"], row["pass_end_y"]),
            axis=1,
        )

        logger.info(
            f"Enriched {len(enriched)} passes with position data "
            f"({enriched['on_pitch_position'].nunique()} position types)"
        )
        return enriched

    def analyze_xt_by_position(
        self, enriched_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, Dict[str, Any]]:
        """Analyze xT by broad position (Defender/Midfielder/Forward).

        Args:
            enriched_df: Enriched baseline data

        Returns:
            Tuple of (analysis_df, summary_stats)
        """
        analysis = (
            enriched_df.groupby("on_pitch_position", observed=True)
            .agg(
                {
                    "xa_plus": ["count", "sum", "mean", "std", "min", "max"],
                    "pass_end_x": ["mean", "std"],
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
                "pass_end_x_std": "std_x",
                "pass_end_y_mean": "avg_y",
            }
        )
        analysis = analysis.reset_index()

        # Add threat classification
        analysis["threat_level"] = analysis["avg_xt"].apply(
            self.xt_model.get_threat_level
        )
        analysis["percentile"] = analysis["avg_xt"].apply(
            self.xt_model.get_xt_percentile
        )

        # Order by position
        position_order = ["Defender", "Midfielder", "Forward", "Unknown"]
        analysis["on_pitch_position"] = pd.Categorical(
            analysis["on_pitch_position"], categories=position_order, ordered=True
        )
        analysis = analysis.sort_values("on_pitch_position")

        summary_stats = {
            "positions_analyzed": len(analysis),
            "details": analysis.to_dict("records"),
        }

        return analysis, summary_stats

    def analyze_xt_by_detailed_position(
        self, enriched_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, Dict[str, Any]]:
        """Analyze xT by detailed position (9 zones).

        Args:
            enriched_df: Enriched baseline data

        Returns:
            Tuple of (analysis_df, summary_stats)
        """
        analysis = (
            enriched_df.groupby("detailed_position", observed=True)
            .agg(
                {
                    "xa_plus": ["count", "sum", "mean", "std"],
                    "pass_end_x": "mean",
                }
            )
            .round(4)
        )

        if len(analysis) == 0:
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
        analysis = analysis.sort_values("avg_xt", ascending=False)
        analysis = analysis.reset_index()

        analysis["threat_level"] = analysis["avg_xt"].apply(
            self.xt_model.get_threat_level
        )

        summary_stats = {
            "positions_analyzed": len(analysis),
            "details": analysis.to_dict("records"),
        }

        return analysis, summary_stats

    def analyze_pitch_heatmap(
        self, enriched_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Create heatmap data of xT distribution across pitch.

        Args:
            enriched_df: Enriched baseline data

        Returns:
            Pivot table of avg xT by location bins
        """
        # Create bins for visualization
        df_copy = enriched_df.copy()
        df_copy["x_bin"] = pd.cut(
            df_copy["pass_end_x"], bins=[0, 40, 80, 120], labels=["Def", "Mid", "Att"], ordered=False
        )
        df_copy["y_bin"] = pd.cut(
            df_copy["pass_end_y"], bins=[0, 27, 53, 80], labels=["Wing_Left", "Central", "Wing_Right"], ordered=False
        )

        pivot = (
            df_copy.pivot_table(
                index="y_bin", columns="x_bin", values="xa_plus", aggfunc="mean"
            )
            .round(4)
        )

        return pivot

    def plot_position_comparison(
        self, analysis_df: pd.DataFrame, output_path: Optional[Path] = None
    ) -> None:
        """Plot xT by position with stats.

        Args:
            analysis_df: Results from analyze_xt_by_position
            output_path: Optional path to save figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: Average xT by position
        sorted_df = analysis_df.sort_values("avg_xt", ascending=True)
        colors = [
            "red" if level == "High" else "orange" if level == "Medium" else "green"
            for level in sorted_df["threat_level"]
        ]

        axes[0].barh(sorted_df["on_pitch_position"], sorted_df["avg_xt"], color=colors)
        axes[0].set_xlabel("Average xT")
        axes[0].set_title("Average xT by Position")
        axes[0].grid(axis="x", alpha=0.3)

        # Right: Total xT by position
        axes[1].barh(sorted_df["on_pitch_position"], sorted_df["total_xt"], color=colors)
        axes[1].set_xlabel("Total xT")
        axes[1].set_title("Total xT Volume by Position")
        axes[1].grid(axis="x", alpha=0.3)

        # Add labels
        for i, (idx, row) in enumerate(sorted_df.iterrows()):
            axes[0].text(row["avg_xt"] * 1.02, i, f"{row['shots']:.0f}", va="center", fontsize=9)
            axes[1].text(row["total_xt"] * 1.02, i, f"{row['shots']:.0f}", va="center", fontsize=9)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved position comparison plot to {output_path}")
        plt.close()

    def plot_detailed_position_comparison(
        self, analysis_df: pd.DataFrame, output_path: Optional[Path] = None
    ) -> None:
        """Plot xT by detailed position.

        Args:
            analysis_df: Results from analyze_xt_by_detailed_position
            output_path: Optional path to save figure
        """
        if len(analysis_df) == 0:
            logger.warning("No detailed position data to plot")
            return

        fig, ax = plt.subplots(figsize=(12, 6))

        sorted_df = analysis_df.sort_values("avg_xt", ascending=True)
        colors = [
            "red" if level == "High" else "orange" if level == "Medium" else "green"
            for level in sorted_df["threat_level"]
        ]

        ax.barh(sorted_df["detailed_position"], sorted_df["avg_xt"], color=colors)
        ax.set_xlabel("Average xT")
        ax.set_title("xT Generation by Detailed Field Position")
        ax.grid(axis="x", alpha=0.3)

        for i, (idx, row) in enumerate(sorted_df.iterrows()):
            ax.text(row["avg_xt"] * 1.02, i, f"n={int(row['shots'])}", va="center", fontsize=8)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved detailed position plot to {output_path}")
        plt.close()

    def plot_pitch_heatmap(
        self, pivot_df: pd.DataFrame, output_path: Optional[Path] = None
    ) -> None:
        """Plot xT heatmap across pitch zones.

        Args:
            pivot_df: Results from analyze_pitch_heatmap
            output_path: Optional path to save figure
        """
        if pivot_df.empty:
            logger.warning("No heatmap data to plot")
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        sns.heatmap(
            pivot_df,
            annot=True,
            fmt=".4f",
            cmap="YlOrRd",
            ax=ax,
            cbar_kws={"label": "Average xT"},
        )
        ax.set_title("xT Generation Across Pitch Zones\n(Defensive-Midfield-Attacking)")
        ax.set_xlabel("Field Third (Defensive ← → Attacking)")
        ax.set_ylabel("Position in Third")

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved heatmap to {output_path}")
        plt.close()


def run_refined_analysis(baseline_path: Path, output_dir: Path) -> None:
    """Run refined position-based xT analysis.

    Args:
        baseline_path: Path to baseline CSV
        output_dir: Path to save outputs
    """
    logger.info("Loading baseline data...")
    baseline_df = pd.read_csv(baseline_path)
    logger.info(f"Loaded {len(baseline_df)} passes")

    # Initialize analyzer
    xt_model = xTModel()
    analyzer = RefinedPositionAnalyzer(xt_model)

    # Enrich with positions
    logger.info("Enriching with position data...")
    enriched_df = analyzer.enrich_baseline_with_positions(baseline_df)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Analyze by broad position
    logger.info("\nAnalyzing by broad position...")
    position_analysis, position_stats = analyzer.analyze_xt_by_position(enriched_df)
    position_analysis.to_csv(output_dir / "cxa_xt_by_position.csv", index=False)
    
    logger.info("Position-Based xT Analysis:")
    logger.info("-" * 70)
    for detail in position_stats["details"]:
        logger.info(
            f"  {detail['on_pitch_position']:12} | "
            f"Shots: {int(detail['shots']):4} | "
            f"Avg xT: {detail['avg_xt']:.4f} | "
            f"Total xT: {detail['total_xt']:.2f} | "
            f"Threat: {detail['threat_level']:6} | "
            f"Percentile: {detail['percentile']:.1f}%"
        )

    # Analyze by detailed position
    logger.info("\nAnalyzing by detailed position...")
    detailed_analysis, detailed_stats = analyzer.analyze_xt_by_detailed_position(enriched_df)
    if not detailed_analysis.empty:
        detailed_analysis.to_csv(output_dir / "cxa_xt_by_detailed_position.csv", index=False)
        
        logger.info("\nDetailed Position-Based xT Analysis:")
        logger.info("-" * 70)
        for detail in detailed_stats["details"]:
            logger.info(
                f"  {detail['detailed_position']:25} | "
                f"Shots: {int(detail['shots']):4} | "
                f"Avg xT: {detail['avg_xt']:.4f} | "
                f"Threat: {detail['threat_level']:6}"
            )

    # Create heatmap data
    logger.info("\nAnalyzing pitch heatmap...")
    heatmap_data = analyzer.analyze_pitch_heatmap(enriched_df)
    if not heatmap_data.empty:
        heatmap_data.to_csv(output_dir / "cxa_xt_pitch_heatmap_data.csv")

    # Create visualizations
    logger.info("\nCreating visualizations...")
    analyzer.plot_position_comparison(
        position_analysis, output_dir / "cxa_position_comparison.png"
    )

    if not detailed_analysis.empty:
        analyzer.plot_detailed_position_comparison(
            detailed_analysis, output_dir / "cxa_detailed_position_comparison.png"
        )

    if not heatmap_data.empty:
        analyzer.plot_pitch_heatmap(heatmap_data, output_dir / "cxa_pitch_heatmap.png")

    # Save enriched baseline
    enriched_df.to_csv(output_dir / "cxa_baseline_enriched.csv", index=False)

    logger.info(f"\nAnalysis complete! Outputs saved to {output_dir}")
    logger.info(
        f"  - Position breakdown CSV: cxa_xt_by_position.csv"
    )
    logger.info(
        f"  - Detailed positions CSV: cxa_xt_by_detailed_position.csv"
    )
    logger.info(
        f"  - Pitch heatmap CSV: cxa_xt_pitch_heatmap_data.csv"
    )
    logger.info(
        f"  - Visualizations: 3 PNG files"
    )
    logger.info(
        f"  - Enriched baseline: cxa_baseline_enriched.csv"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_refined_analysis(
        Path("outputs/analysis/cxa/csv/cxa_baselines_pass_level.csv"),
        Path("outputs/analysis/cxa"),
    )
