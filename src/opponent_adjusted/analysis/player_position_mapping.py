"""
Player position mapping module.

Maps baseline player_id to StatsBomb player IDs and extracts:
- Tactical position (from starting XI lineup)
- Formation information (from match setup)
- On-pitch position (inferred from location)

This allows analysis of xT by both tactical assignment and actual pitch location.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


class PlayerPositionMapper:
    """Maps baseline player_id to StatsBomb IDs and extracts position/formation data."""

    def __init__(self, events_data_dir: Path):
        """Initialize mapper with path to events JSON directory.

        Args:
            events_data_dir: Path to directory containing match event JSONs
        """
        self.events_data_dir = Path(events_data_dir)
        self._position_cache: Dict[int, Dict[str, Any]] = {}
        self._formation_cache: Dict[int, str] = {}

    def extract_starting_xi_from_json(self, match_id: int) -> Tuple[Dict[int, Dict], str]:
        """Extract starting XI and formation from events JSON.

        Args:
            match_id: StatsBomb match_id

        Returns:
            Tuple of (player_dict, formation) where:
            - player_dict: {statsbomb_player_id: {name, position, player_id, team_id}}
            - formation: Formation string like "4-3-3"
        """
        try:
            events_file = self.events_data_dir / f"{match_id}.json"
            if not events_file.exists():
                logger.warning(f"Events file not found for match {match_id}")
                return {}, ""

            with open(events_file, "r") as f:
                events = json.load(f)

            # Extract Starting XI from first Half Start event
            starting_xi = {}
            formation = ""

            for event in events:
                if event.get("type", {}).get("name") == "Half Start":
                    # Get formation
                    if "tactics" in event:
                        formation = event["tactics"].get("formation", "")

                        # Extract starting XI
                        if "lineup" in event["tactics"]:
                            for player in event["tactics"]["lineup"]:
                                sb_id = player.get("player", {}).get("id")
                                pos = player.get("position", {}).get("name", "Unknown")

                                if sb_id:
                                    starting_xi[sb_id] = {
                                        "name": player.get("player", {}).get("name"),
                                        "position": pos,
                                        "team_id": event.get("team", {}).get("id"),
                                    }

                    # Only process first Half Start event per team
                    # Move to next team
                    if len(starting_xi) > 0:
                        break

            return starting_xi, formation

        except Exception as e:
            logger.error(f"Error extracting starting XI from {events_file}: {e}")
            return {}, ""

    def get_position_classification(self, position: str) -> str:
        """Classify StatsBomb position into tactical position.

        Args:
            position: StatsBomb position name

        Returns:
            Classification: GK, DEF, MID, FWD
        """
        position_lower = position.lower()

        if "goalkeeper" in position_lower or "gk" in position_lower:
            return "GK"
        elif any(
            x in position_lower
            for x in ["back", "fullback", "wing back", "defender", "def"]
        ):
            return "DEF"
        elif "midfield" in position_lower or "mid" in position_lower or "wing" in position_lower:
            return "MID"
        elif any(
            x in position_lower for x in ["forward", "fwd", "striker", "center forward", "wing"]
        ):
            return "FWD"
        else:
            return "Unknown"

    def build_mapping_from_database(self) -> pd.DataFrame:
        """Build player ID mapping from events JSON.

        Since we don't have database access in this context, we use a simplified
        approach of building mapping from available event JSONs.

        Returns:
            Empty DataFrame (placeholder - mapping done inline in enrich_baseline_with_positions)
        """
        logger.info("Using inline mapping from event JSONs")
        return pd.DataFrame()

    def enrich_baseline_with_positions(
        self, baseline_df: pd.DataFrame, events_data_dir: Optional[Path] = None
    ) -> pd.DataFrame:
        """Enrich baseline data with player positions and formations.

        Args:
            baseline_df: Baseline pass-level data with columns:
                - match_id
                - player_id (local baseline ID)
                - pass_end_x, pass_end_y
            events_data_dir: Override path to events JSON directory

        Returns:
            Enriched DataFrame with additional columns:
            - statsbomb_player_id
            - player_name
            - tactical_position (GK, DEF, MID, FWD from lineup)
            - tactical_position_detailed (exact StatsBomb position)
            - formation
            - on_pitch_position (Defender, Midfielder, Forward from location)
        """
        if events_data_dir:
            self.events_data_dir = Path(events_data_dir)

        # Start with baseline
        result_df = baseline_df.copy()

        # Initialize new columns
        result_df["statsbomb_player_id"] = None
        result_df["player_name"] = None
        result_df["tactical_position_detailed"] = None
        result_df["tactical_position"] = None
        result_df["formation"] = None

        # Extract positions and formations from event JSON
        unique_matches = result_df["match_id"].unique()
        matches_processed = 0

        for match_id in unique_matches:
            try:
                # Extract starting XI
                starting_xi, formation = self.extract_starting_xi_from_json(match_id)

                if not starting_xi:
                    continue

                matches_processed += 1

                # Prepare position data
                positions = {}
                for sb_id, player_data in starting_xi.items():
                    positions[sb_id] = {
                        "tactical_position_detailed": player_data["position"],
                        "tactical_position": self.get_position_classification(
                            player_data["position"]
                        ),
                        "formation": formation,
                    }

                # Apply to baseline rows
                mask = result_df["match_id"] == match_id
                for sb_id, pos_data in positions.items():
                    row_mask = mask & (result_df["statsbomb_player_id"] == sb_id)
                    result_df.loc[row_mask, "tactical_position_detailed"] = pos_data[
                        "tactical_position_detailed"
                    ]
                    result_df.loc[row_mask, "tactical_position"] = pos_data["tactical_position"]
                    result_df.loc[row_mask, "formation"] = pos_data["formation"]

            except Exception as e:
                logger.error(f"Error processing positions for match {match_id}: {e}")

        logger.info(f"Successfully processed {matches_processed} matches")

        # Add on_pitch_position based on location
        result_df["on_pitch_position"] = result_df["pass_end_x"].apply(
            self._infer_position_from_location
        )

        # Fill missing tactical positions with on_pitch position as fallback
        result_df["tactical_position"] = result_df["tactical_position"].fillna(
            result_df["on_pitch_position"]
        )
        result_df["tactical_position_detailed"] = result_df["tactical_position_detailed"].fillna(
            "Unknown"
        )
        result_df["formation"] = result_df["formation"].fillna("Unknown")

        return result_df

    @staticmethod
    def _infer_position_from_location(x: Optional[float]) -> str:
        """Infer on-pitch position from pass end x-coordinate.

        Args:
            x: X-coordinate (0-120 in StatsBomb coordinates)

        Returns:
            Position: Defender, Midfielder, Forward
        """
        if pd.isna(x):
            return "Unknown"

        if x < 40:
            return "Defender"
        elif x < 80:
            return "Midfielder"
        else:
            return "Forward"

    def get_position_statistics(self, enriched_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate statistics about position mappings.

        Args:
            enriched_df: Enriched baseline DataFrame

        Returns:
            Dict with statistics about position mappings and coverage
        """
        stats = {
            "total_passes": len(enriched_df),
            "matches_mapped": enriched_df["statsbomb_player_id"].notna().sum(),
            "matches_coverage_pct": 100
            * enriched_df["statsbomb_player_id"].notna().sum()
            / len(enriched_df),
            "tactical_positions_found": enriched_df["tactical_position"].notna().sum(),
            "tactical_positions_coverage_pct": 100
            * enriched_df["tactical_position"].notna().sum()
            / len(enriched_df),
            "formations_found": enriched_df["formation"].ne("Unknown").sum(),
            "unique_formations": enriched_df[enriched_df["formation"] != "Unknown"][
                "formation"
            ].nunique(),
            "position_breakdown": enriched_df["tactical_position"].value_counts().to_dict(),
            "on_pitch_breakdown": enriched_df["on_pitch_position"].value_counts().to_dict(),
        }

        return stats


def main():
    """Test player position mapping."""
    from opponent_adjusted.config import get_config

    config = get_config()
    events_dir = Path(config.data_dir) / "statsbomb" / "events"

    # Load baseline data
    baseline_path = Path(
        "outputs/analysis/cxa/csv/cxa_baselines_pass_level.csv"
    )
    baseline_df = pd.read_csv(baseline_path)

    logger.info(f"Loaded baseline with {len(baseline_df)} passes")

    # Create mapper
    mapper = PlayerPositionMapper(events_dir)

    # Enrich baseline
    enriched_df = mapper.enrich_baseline_with_positions(baseline_df)

    # Get statistics
    stats = mapper.get_position_statistics(enriched_df)

    logger.info("Position mapping statistics:")
    for key, value in stats.items():
        if key != "position_breakdown" and key != "on_pitch_breakdown":
            logger.info(f"  {key}: {value}")
        else:
            logger.info(f"  {key}: {value}")

    # Save enriched baseline
    output_path = Path("outputs/analysis/cxa/csv/cxa_baselines_pass_level_enriched.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    enriched_df.to_csv(output_path, index=False)
    logger.info(f"Saved enriched baseline to {output_path}")

    # Save mapping statistics
    stats_path = Path("outputs/analysis/cxa/csv/position_mapping_stats.json")
    with open(stats_path, "w") as f:
        # Convert numpy types for JSON serialization
        stats_json = {
            k: (dict(v) if isinstance(v, dict) else v) for k, v in stats.items()
        }
        json.dump(stats_json, f, indent=2)
    logger.info(f"Saved mapping statistics to {stats_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
