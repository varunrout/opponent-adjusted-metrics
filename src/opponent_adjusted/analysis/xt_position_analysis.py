"""
Simplified xT analysis by position using available data.

Since baseline player IDs don't directly map to StatsBomb IDs,
we use location-based position inference plus formation data from event JSON.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from opponent_adjusted.features.xt_model import xTModel

logger = logging.getLogger(__name__)


class PositionAnalyzer:
    """Analyzes xT by position using location-based inference and formation data."""

    def __init__(self, xt_model: xTModel):
        """Initialize analyzer.

        Args:
            xt_model: Trained xTModel instance
        """
        self.xt_model = xt_model
        self._formation_cache: Dict[int, str] = {}

    def extract_formation_from_json(self, events_file: Path) -> str:
        """Extract formation from event JSON.

        Args:
            events_file: Path to match events JSON

        Returns:
            Formation string like "4-3-3" or "Unknown"
        """
        try:
            with open(events_file, "r", encoding="utf-8") as f:
                events = json.load(f)

            # Get first event with tactics/formation
            for event in events:
                if "tactics" in event and "formation" in event.get("tactics", {}):
                    formation = event["tactics"]["formation"]
                    # Format as standard notation (4-2-3-1, etc)
                    formation_str = str(formation)
                    if len(formation_str) == 4:
                        return f"{formation_str[0]}-{formation_str[1]}-{formation_str[2]}-{formation_str[3]}"
                    return formation_str

            return "Unknown"

        except Exception as e:
            logger.warning(f"Could not extract formation from {events_file}: {e}")
            return "Unknown"

    def enrich_with_formation_and_position(
        self, baseline_df: pd.DataFrame, events_dir: Path
    ) -> pd.DataFrame:
        """Enrich baseline with formation and on-pitch position.

        Args:
            baseline_df: Baseline pass-level data
            events_dir: Path to events JSON directory

        Returns:
            Enriched DataFrame with position and formation columns
        """
        enriched = baseline_df.copy()

        # Add on-pitch position based on pass_end_x
        enriched["on_pitch_position"] = enriched["pass_end_x"].apply(
            self._infer_position_from_x
        )

        # Add formation based on match_id
        enriched["formation"] = enriched["match_id"].apply(
            lambda mid: self._get_formation_for_match(mid, events_dir)
        )

        logger.info(f"Enriched {len(enriched)} passes with positions and formations")
        return enriched

    def _get_formation_for_match(self, match_id: int, events_dir: Path) -> str:
        """Get formation for a specific match.

        Args:
            match_id: Match ID in StatsBomb
            events_dir: Path to events directory

        Returns:
            Formation string
        """
        if match_id in self._formation_cache:
            return self._formation_cache[match_id]

        events_file = events_dir / f"{match_id}.json"
        if not events_file.exists():
            logger.warning(f"Events file not found for match {match_id}")
            return "Unknown"

        formation = self.extract_formation_from_json(events_file)
        self._formation_cache[match_id] = formation
        return formation

    @staticmethod
    def _infer_position_from_x(x: Optional[float]) -> str:
        """Infer on-pitch position from x-coordinate.

        Args:
            x: X-coordinate (0-120 in StatsBomb coordinates)

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

    def analyze_xt_by_on_pitch_position(
        self, enriched_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, Dict[str, Any]]:
        """Analyze xT by on-pitch position.

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

        # Add threat classification
        analysis["threat_level"] = analysis["avg_xt"].apply(
            self.xt_model.get_threat_level
        )
        analysis["percentile"] = analysis["avg_xt"].apply(
            self.xt_model.get_xt_percentile
        )

        summary_stats = {
            "positions_analyzed": len(analysis),
            "details": analysis.to_dict("records"),
        }

        return analysis, summary_stats

    def analyze_xt_by_formation(
        self, enriched_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, Dict[str, Any]]:
        """Analyze xT by formation.

        Args:
            enriched_df: Enriched baseline data

        Returns:
            Tuple of (analysis_df, summary_stats)
        """
        analysis = (
            enriched_df[enriched_df["formation"] != "Unknown"]
            .groupby("formation", observed=True)
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
            "details": analysis.to_dict("records"),
        }

        return analysis, summary_stats

    def analyze_position_by_formation(
        self, enriched_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Cross-tabulate xT by position and formation.

        Args:
            enriched_df: Enriched baseline data

        Returns:
            Pivot table of avg xT by position and formation
        """
        pivot = (
            enriched_df[enriched_df["formation"] != "Unknown"]
            .pivot_table(
                index="on_pitch_position",
                columns="formation",
                values="xa_plus",
                aggfunc="mean",
            )
            .round(4)
        )

        return pivot

    def plot_position_comparison(
        self, analysis_df: pd.DataFrame, output_path: Optional[Path] = None
    ) -> None:
        """Plot xT by on-pitch position.

        Args:
            analysis_df: Results from analyze_xt_by_on_pitch_position
            output_path: Optional path to save figure
        """
        fig, ax = plt.subplots(figsize=(10, 5))

        sorted_df = analysis_df.sort_values("avg_xt", ascending=True)
        colors = [
            "red" if level == "High" else "orange" if level == "Medium" else "green"
            for level in sorted_df["threat_level"]
        ]

        ax.barh(sorted_df["on_pitch_position"], sorted_df["avg_xt"], color=colors)
        ax.set_xlabel("Average xT")
        ax.set_title("xT Generation by On-Pitch Position")
        ax.grid(axis="x", alpha=0.3)

        # Add shot counts
        for i, (idx, row) in enumerate(sorted_df.iterrows()):
            ax.text(row["avg_xt"] * 1.02, i, f"n={int(row['shots'])}", va="center", fontsize=9)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved position plot to {output_path}")
        plt.close()

    def plot_formation_comparison(
        self, analysis_df: pd.DataFrame, output_path: Optional[Path] = None
    ) -> None:
        """Plot xT by formation.

        Args:
            analysis_df: Results from analyze_xt_by_formation
            output_path: Optional path to save figure
        """
        if len(analysis_df) == 0:
            logger.warning("No formation data to plot")
            return

        fig, ax = plt.subplots(figsize=(12, 6))

        sorted_df = analysis_df.sort_values("avg_xt", ascending=True)
        colors = [
            "red" if level == "High" else "orange" if level == "Medium" else "green"
            for level in sorted_df["threat_level"]
        ]

        ax.barh(sorted_df["formation"], sorted_df["avg_xt"], color=colors)
        ax.set_xlabel("Average xT")
        ax.set_title("xT Generation by Formation")
        ax.grid(axis="x", alpha=0.3)

        # Add shot counts
        for i, (idx, row) in enumerate(sorted_df.iterrows()):
            ax.text(row["avg_xt"] * 1.02, i, f"n={int(row['shots'])}", va="center", fontsize=9)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved formation plot to {output_path}")
        plt.close()

    def plot_position_formation_heatmap(
        self, pivot_df: pd.DataFrame, output_path: Optional[Path] = None
    ) -> None:
        """Plot heatmap of xT by position and formation.

        Args:
            pivot_df: Results from analyze_position_by_formation
            output_path: Optional path to save figure
        """
        if pivot_df.empty:
            logger.warning("No position-formation data to plot")
            return

        fig, ax = plt.subplots(figsize=(12, 6))

        sns.heatmap(
            pivot_df,
            annot=True,
            fmt=".4f",
            cmap="YlOrRd",
            ax=ax,
            cbar_kws={"label": "Average xT"},
        )
        ax.set_title("xT Generation by Position and Formation")
        ax.set_xlabel("Formation")
        ax.set_ylabel("On-Pitch Position")

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved heatmap to {output_path}")
        plt.close()


def run_analysis(
    baseline_path: Path, events_dir: Path, output_dir: Path
) -> None:
    """Run complete position-based xT analysis.

    Args:
        baseline_path: Path to baseline CSV
        events_dir: Path to events JSON directory
        output_dir: Path to save outputs
    """
    logger.info("Loading baseline data...")
    baseline_df = pd.read_csv(baseline_path)
    logger.info(f"Loaded {len(baseline_df)} passes")

    # Initialize analyzer
    xt_model = xTModel()
    analyzer = PositionAnalyzer(xt_model)

    # Enrich with position and formation
    logger.info("Enriching with position and formation data...")
    enriched_df = analyzer.enrich_with_formation_and_position(baseline_df, events_dir)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Analyze by on-pitch position
    logger.info("Analyzing by on-pitch position...")
    position_analysis, position_stats = analyzer.analyze_xt_by_on_pitch_position(enriched_df)
    position_analysis.to_csv(output_dir / "cxa_xt_by_position_simplified.csv", index=False)
    logger.info(f"  Positions: {position_stats['positions_analyzed']}")
    for detail in position_stats["details"]:
        logger.info(
            f"    {detail['on_pitch_position']}: {detail['shots']:.0f} shots, "
            f"{detail['avg_xt']:.4f} avg xT ({detail['threat_level']})"
        )

    # Analyze by formation
    logger.info("Analyzing by formation...")
    formation_analysis, formation_stats = analyzer.analyze_xt_by_formation(enriched_df)
    if not formation_analysis.empty:
        formation_analysis.to_csv(output_dir / "cxa_xt_by_formation_simplified.csv", index=False)
        logger.info(f"  Formations: {formation_stats['formations_analyzed']}")
        for detail in formation_stats["details"]:
            logger.info(
                f"    {detail['formation']}: {detail['shots']:.0f} shots, "
                f"{detail['avg_xt']:.4f} avg xT"
            )

    # Cross-analysis
    logger.info("Cross-analyzing position by formation...")
    position_formation = analyzer.analyze_position_by_formation(enriched_df)
    if not position_formation.empty:
        position_formation.to_csv(output_dir / "cxa_position_by_formation.csv")

    # Create visualizations
    logger.info("Creating visualizations...")
    analyzer.plot_position_comparison(
        position_analysis, output_dir / "cxa_position_comparison.png"
    )

    if not formation_analysis.empty:
        analyzer.plot_formation_comparison(
            formation_analysis, output_dir / "cxa_formation_comparison.png"
        )

    if not position_formation.empty:
        analyzer.plot_position_formation_heatmap(
            position_formation, output_dir / "cxa_position_formation_heatmap.png"
        )

    # Save enriched baseline
    enriched_df.to_csv(output_dir / "cxa_baseline_enriched_simplified.csv", index=False)

    logger.info(f"\nAnalysis complete! Outputs saved to {output_dir}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_analysis(
        Path("outputs/analysis/cxa/csv/cxa_baselines_pass_level.csv"),
        Path("data/statsbomb/events"),
        Path("outputs/analysis/cxa/csv"),
    )
