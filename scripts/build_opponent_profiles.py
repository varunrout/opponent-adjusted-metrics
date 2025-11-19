"""Build opponent defensive profiles (global and zone-based).

For each defending team (opponent_team_id in Shots), compute:
- Global rating: negative mean conceded xG (lower is better)
- Global block rate: fraction of shots blocked
- Zone ratings: negative mean conceded xG per geometry zone, with simple shrinkage

Writes rows into `opponent_def_profile` with:
- zone_id = NULL for global row; per-zone rows have zone_id in {A..F}
- version_tag = provided version
"""

import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from opponent_adjusted.db.session import session_scope
from opponent_adjusted.db.models import Shot, ShotFeature, OpponentDefProfile
from opponent_adjusted.features.geometry import assign_zone
from opponent_adjusted.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Agg:
    sxg: float = 0.0
    n: int = 0
    blocks: int = 0


def _shrink(mean_zone: float, n_zone: int, mean_global: float, prior: float = 50.0) -> float:
    if n_zone <= 0:
        return mean_global
    return (n_zone * mean_zone + prior * mean_global) / (n_zone + prior)


def build_profiles(version: str) -> None:
    with session_scope() as session:
        # Join shots with features for the version
        rows = (
            session.query(Shot, ShotFeature)
            .join(ShotFeature, ShotFeature.shot_id == Shot.id)
            .filter(ShotFeature.version_tag == version)
            .all()
        )

        if not rows:
            logger.warning("No shots with features for version %s", version)
            return

        # Aggregations per defending team and zone
        global_agg: Dict[int, Agg] = defaultdict(Agg)
        zone_agg: Dict[Tuple[int, str], Agg] = defaultdict(Agg)

        for shot, feat in rows:
            defend_id = shot.opponent_team_id
            if defend_id is None:
                continue
            xg = shot.statsbomb_xg
            if xg is None:
                continue
            # Global
            g = global_agg[defend_id]
            g.sxg += float(xg)
            g.n += 1
            if shot.is_blocked:
                g.blocks += 1

            # Zone
            if feat.shot_distance is None or feat.centrality is None:
                continue
            z = assign_zone(float(feat.shot_distance), float(feat.centrality))
            zg = zone_agg[(defend_id, z)]
            zg.sxg += float(xg)
            zg.n += 1
            if shot.is_blocked:
                zg.blocks += 1

        # Upsert profiles
        inserted = 0
        updated = 0

        for team_id, g in global_agg.items():
            mean_xg = g.sxg / g.n if g.n > 0 else 0.0
            block_rate = g.blocks / g.n if g.n > 0 else 0.0
            global_rating = -mean_xg

            row = (
                session.query(OpponentDefProfile)
                .filter_by(team_id=team_id, version_tag=version, zone_id=None)
                .first()
            )
            if row:
                row.global_rating = global_rating
                row.block_rate = block_rate
                row.zone_rating = None
                row.shots_sample = g.n
                updated += 1
            else:
                row = OpponentDefProfile(
                    team_id=team_id,
                    version_tag=version,
                    zone_id=None,
                    global_rating=global_rating,
                    block_rate=block_rate,
                    zone_rating=None,
                    shots_sample=g.n,
                )
                session.add(row)
                inserted += 1

            # Per-zone ratings with shrinkage
            for zone in list("ABCDEF"):
                zg = zone_agg.get((team_id, zone))
                n_zone = zg.n if zg else 0
                mean_zone_xg = (zg.sxg / n_zone) if (zg and n_zone > 0) else mean_xg
                shrunk = _shrink(mean_zone_xg, n_zone, mean_xg, prior=50.0)
                zone_rating = -shrunk
                zone_block_rate = (zg.blocks / n_zone) if (zg and n_zone > 0) else block_rate

                prow = (
                    session.query(OpponentDefProfile)
                    .filter_by(team_id=team_id, version_tag=version, zone_id=zone)
                    .first()
                )
                if prow:
                    prow.global_rating = None
                    prow.block_rate = zone_block_rate
                    prow.zone_rating = zone_rating
                    prow.shots_sample = n_zone
                    updated += 1
                else:
                    prow = OpponentDefProfile(
                        team_id=team_id,
                        version_tag=version,
                        zone_id=zone,
                        global_rating=None,
                        block_rate=zone_block_rate,
                        zone_rating=zone_rating,
                        shots_sample=n_zone,
                    )
                    session.add(prow)
                    inserted += 1

        logger.info(
            "Opponent profiles built for version %s. Inserted: %d, Updated: %d",
            version,
            inserted,
            updated,
        )


def main():
    parser = argparse.ArgumentParser(description="Build opponent profiles")
    parser.add_argument("--version", default="v1", help="Feature version tag")
    args = parser.parse_args()

    build_profiles(args.version)


if __name__ == "__main__":
    main()
