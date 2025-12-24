"""Run comprehensive cxA analysis.

This script runs all cxA analysis modules:
- Baseline analysis (pass-level xA statistics)
- Spatial analysis (zones, corridors)
- Team analysis (profiles, clustering by style)
- Position-cluster analysis (position groups split by team style)

Example:
    python scripts/run_cxa_analysis.py \
      --database-url sqlite:///data/opponent_adjusted.db \
      --baseline-csv outputs/analysis/cxa/csv/cxa_baselines_pass_level.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Allow running directly without Poetry
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_PATH = _REPO_ROOT / "src"
if _SRC_PATH.exists() and str(_SRC_PATH) not in sys.path:
    sys.path.insert(0, str(_SRC_PATH))

from opponent_adjusted.analysis.cxa import (
    run_cxa_baseline_analysis,
    run_spatial_analysis,
    run_team_analysis,
)
from opponent_adjusted.analysis.cxa.position_cluster_analysis import (
    run_position_cluster_analysis,
)
from opponent_adjusted.analysis.xt_spatial_analysis import run_xt_spatial_analysis
from opponent_adjusted.analysis.assist_sequence_analysis import run_assist_sequence_analysis


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run comprehensive cxA analysis")
    p.add_argument("--database-url", required=True, type=str)
    p.add_argument(
        "--baseline-csv",
        type=Path,
        default=Path("outputs") / "analysis" / "cxa" / "csv" / "cxa_baselines_pass_level.csv",
    )
    p.add_argument(
        "--analyses",
        nargs="+",
        default=["baseline", "spatial", "team", "position", "xt", "assists"],
        choices=["baseline", "spatial", "team", "position", "xt", "assists", "all"],
        help="Which analyses to run",
    )
    p.add_argument("--n-team-clusters", type=int, default=4)
    p.add_argument("--min-team-passes", type=int, default=20)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    engine = create_engine(args.database_url)
    SessionLocal = sessionmaker(bind=engine)

    analyses = set(args.analyses)
    if "all" in analyses:
        analyses = {"baseline", "spatial", "team", "position", "xt", "assists"}

    with SessionLocal() as session:
        if "baseline" in analyses:
            print("\n" + "=" * 60)
            print("Running baseline analysis...")
            print("=" * 60)
            run_cxa_baseline_analysis(args.baseline_csv, session=session)

        if "spatial" in analyses:
            print("\n" + "=" * 60)
            print("Running spatial analysis...")
            print("=" * 60)
            run_spatial_analysis(args.baseline_csv, session)

        if "team" in analyses:
            print("\n" + "=" * 60)
            print("Running team analysis...")
            print("=" * 60)
            run_team_analysis(
                args.baseline_csv,
                session,
                min_passes=args.min_team_passes,
                n_clusters=args.n_team_clusters,
            )

        if "position" in analyses:
            print("\n" + "=" * 60)
            print("Running position-cluster analysis...")
            print("=" * 60)
            team_clusters_csv = (
                Path("outputs") / "analysis" / "cxa" / "csv" / "cxa_team_clusters.csv"
            )
            if not team_clusters_csv.exists():
                print(f"Error: {team_clusters_csv} not found. Run 'team' analysis first.")
                sys.exit(1)
            run_position_cluster_analysis(args.baseline_csv, team_clusters_csv)

        if "xt" in analyses:
            print("\n" + "=" * 60)
            print("Running xT spatial analysis...")
            print("=" * 60)
            team_clusters_csv = (
                Path("outputs") / "analysis" / "cxa" / "csv" / "cxa_team_clusters.csv"
            )
            if not team_clusters_csv.exists():
                print(f"Error: {team_clusters_csv} not found. Run 'team' analysis first.")
                sys.exit(1)
            
            df = pd.read_csv(args.baseline_csv)
            team_clusters = pd.read_csv(team_clusters_csv)
            output_dir = Path("outputs") / "analysis" / "cxa"
            
            run_xt_spatial_analysis(df, team_clusters, output_dir)

        if "assists" in analyses:
            print("\n" + "=" * 60)
            print("Running assist sequence analysis...")
            print("=" * 60)
            team_clusters_csv = (
                Path("outputs") / "analysis" / "cxa" / "csv" / "cxa_team_clusters.csv"
            )
            if not team_clusters_csv.exists():
                print(f"Error: {team_clusters_csv} not found. Run 'team' analysis first.")
                sys.exit(1)
            
            df = pd.read_csv(args.baseline_csv)
            team_clusters = pd.read_csv(team_clusters_csv)
            output_dir = Path("outputs") / "analysis" / "cxa"
            
            run_assist_sequence_analysis(df, team_clusters, output_dir)

    print("\n" + "=" * 60)
    print("All requested analyses complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
