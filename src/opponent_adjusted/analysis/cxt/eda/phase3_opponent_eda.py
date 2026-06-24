"""Phase 3: Opponent Context EDA for CxT.

Analyzes opponent strength effects on ball progression.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def run_phase3_eda(
    df: pd.DataFrame,
    output_dir: Path,
    opponent_profiles: pd.DataFrame = None,
) -> Dict[str, Any]:
    """Run Phase 3: Opponent Context analysis.

    Args:
        df: Progressions DataFrame
        output_dir: Directory to save outputs
        opponent_profiles: Optional opponent profiles DataFrame

    Returns:
        Dictionary with analysis results
    """
    logger.info("=" * 60)
    logger.info("Phase 3: Opponent Context EDA")
    logger.info("=" * 60)

    results = {}
    plots_dir = output_dir / "plots"
    csv_dir = output_dir / "csv"
    plots_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. Actions by Opponent (Team-Level Analysis)
    # -------------------------------------------------------------------------
    logger.info("\n1. Actions by Opponent")

    # Aggregate by opponent
    by_opponent = (
        df.groupby("opponent_id")
        .agg(
            total_actions=("event_id", "count"),
            mean_xt_delta=("xt_delta", "mean"),
            progressive_rate=("is_progressive", "mean"),
            under_pressure_rate=("under_pressure", "mean"),
            matches=("match_id", "nunique"),
        )
        .round(4)
    )

    by_opponent["actions_per_match"] = (
        by_opponent["total_actions"] / by_opponent["matches"]
    ).round(1)
    by_opponent = by_opponent.sort_values("total_actions", ascending=False)

    by_opponent.to_csv(csv_dir / "actions_by_opponent.csv")
    results["opponents_analyzed"] = len(by_opponent)

    logger.info(f"  Unique opponents: {len(by_opponent)}")
    logger.info(
        f"  Mean actions per match vs opponent: {by_opponent['actions_per_match'].mean():.1f}"
    )

    # -------------------------------------------------------------------------
    # 2. Opponent Pressure Analysis
    # -------------------------------------------------------------------------
    logger.info("\n2. Opponent Pressure Analysis")

    # Find high-pressure vs low-pressure opponents
    opponent_pressure = df.groupby("opponent_id")["under_pressure"].mean()

    # Top quartile = high pressure opponents
    q75 = opponent_pressure.quantile(0.75)
    q25 = opponent_pressure.quantile(0.25)

    high_pressure_opponents = opponent_pressure[opponent_pressure >= q75].index
    low_pressure_opponents = opponent_pressure[opponent_pressure <= q25].index

    df["opponent_pressure_tier"] = "Medium"
    df.loc[df["opponent_id"].isin(high_pressure_opponents), "opponent_pressure_tier"] = "High"
    df.loc[df["opponent_id"].isin(low_pressure_opponents), "opponent_pressure_tier"] = "Low"

    tier_analysis = (
        df.groupby("opponent_pressure_tier")
        .agg(
            total_actions=("event_id", "count"),
            mean_xt_delta=("xt_delta", "mean"),
            progressive_rate=("is_progressive", "mean"),
            under_pressure_rate=("under_pressure", "mean"),
        )
        .round(4)
    )

    tier_analysis.to_csv(csv_dir / "opponent_pressure_tiers.csv")

    logger.info("\nOpponent Pressure Tiers:")
    for tier, row in tier_analysis.iterrows():
        logger.info(
            f"  {tier}: {row['total_actions']:,} actions, "
            f"xT={row['mean_xt_delta']:.4f}, "
            f"prog={row['progressive_rate']*100:.1f}%"
        )

    # Check signal: high pressure should reduce xT delta
    if "High" in tier_analysis.index and "Low" in tier_analysis.index:
        xt_signal = (
            tier_analysis.loc["Low", "mean_xt_delta"] - tier_analysis.loc["High", "mean_xt_delta"]
        )
        results["pressure_tier_xt_signal"] = xt_signal
        logger.info(f"\n  Signal Check: Low vs High pressure xT diff = {xt_signal:.6f}")
        if xt_signal > 0:
            logger.info("  ✓ Expected direction (more xT vs low-pressure opponents)")
        else:
            logger.info("  ⚠ Unexpected direction (investigate)")

    # -------------------------------------------------------------------------
    # 3. Opponent Defensive Quality (if profiles available)
    # -------------------------------------------------------------------------
    if opponent_profiles is not None and len(opponent_profiles) > 0:
        logger.info("\n3. Opponent Profiles Integration")

        # Merge profiles
        global_profiles = opponent_profiles[opponent_profiles["zone_id"].isna()].copy()

        if "global_rating" in global_profiles.columns:
            profile_map = global_profiles.set_index("team_id")["global_rating"].to_dict()
            df["opponent_rating"] = df["opponent_id"].map(profile_map)

            # Quartile analysis
            df["opponent_strength_tier"] = pd.qcut(
                df["opponent_rating"].fillna(50),
                q=4,
                labels=["Weak", "Below Avg", "Above Avg", "Strong"],
            )

            strength_analysis = (
                df.groupby("opponent_strength_tier")
                .agg(
                    total_actions=("event_id", "count"),
                    mean_xt_delta=("xt_delta", "mean"),
                    progressive_rate=("is_progressive", "mean"),
                )
                .round(4)
            )

            strength_analysis.to_csv(csv_dir / "opponent_strength_analysis.csv")

            logger.info("\nOpponent Strength Analysis:")
            for tier, row in strength_analysis.iterrows():
                logger.info(
                    f"  {tier}: xT={row['mean_xt_delta']:.4f}, prog={row['progressive_rate']*100:.1f}%"
                )

            results["opponent_strength_integrated"] = True
    else:
        logger.info("\n3. Opponent Profiles: Not available (will be added in Phase 3)")
        results["opponent_strength_integrated"] = False

    # -------------------------------------------------------------------------
    # 4. Home vs Away Analysis
    # -------------------------------------------------------------------------
    logger.info("\n4. Home vs Away Analysis")

    home_away = (
        df.groupby("is_home")
        .agg(
            total_actions=("event_id", "count"),
            mean_xt_delta=("xt_delta", "mean"),
            progressive_rate=("is_progressive", "mean"),
            under_pressure_rate=("under_pressure", "mean"),
        )
        .round(4)
    )

    home_away.index = home_away.index.map({True: "Home", False: "Away"})
    home_away.to_csv(csv_dir / "home_away_analysis.csv")

    logger.info("\nHome vs Away:")
    for loc, row in home_away.iterrows():
        logger.info(
            f"  {loc}: {row['total_actions']:,} actions, "
            f"xT={row['mean_xt_delta']:.4f}, "
            f"pressure={row['under_pressure_rate']*100:.1f}%"
        )

    if "Home" in home_away.index and "Away" in home_away.index:
        home_advantage = (
            home_away.loc["Home", "mean_xt_delta"] - home_away.loc["Away", "mean_xt_delta"]
        )
        results["home_xt_advantage"] = home_advantage
        logger.info(f"\n  Home xT advantage: {home_advantage:.6f}")

    # -------------------------------------------------------------------------
    # 5. Competition-Level Effects
    # -------------------------------------------------------------------------
    logger.info("\n5. Competition Analysis")

    by_competition = (
        df.groupby("competition_id")
        .agg(
            total_actions=("event_id", "count"),
            mean_xt_delta=("xt_delta", "mean"),
            progressive_rate=("is_progressive", "mean"),
            matches=("match_id", "nunique"),
        )
        .round(4)
    )

    by_competition.to_csv(csv_dir / "by_competition.csv")

    logger.info(f"  Competitions analyzed: {len(by_competition)}")
    for comp, row in by_competition.iterrows():
        logger.info(
            f"    Competition {comp}: {row['matches']} matches, xT={row['mean_xt_delta']:.4f}"
        )

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Opponent pressure tier comparison
    ax1 = axes[0]
    tier_order = ["Low", "Medium", "High"]
    tier_data = tier_analysis.reindex(tier_order)
    ax1.bar(tier_data.index, tier_data["mean_xt_delta"] * 1000, color=["green", "gray", "red"])
    ax1.set_xlabel("Opponent Pressure Tier")
    ax1.set_ylabel("Mean xT Delta (×1000)")
    ax1.set_title("xT Delta by Opponent Pressure Tier")
    ax1.axhline(y=0, color="black", linestyle="--", alpha=0.5)

    # Home vs Away
    ax2 = axes[1]
    ax2.bar(home_away.index, home_away["mean_xt_delta"] * 1000, color=["blue", "orange"])
    ax2.set_xlabel("Location")
    ax2.set_ylabel("Mean xT Delta (×1000)")
    ax2.set_title("xT Delta: Home vs Away")
    ax2.axhline(y=0, color="black", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(plots_dir / "opponent_context.png", dpi=150)
    plt.close()

    logger.info("\n✓ Phase 3 complete")
    return results
