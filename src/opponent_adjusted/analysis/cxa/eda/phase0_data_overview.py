"""Phase 0 EDA: Data Overview & Alignment.

Comprehensive overview of all cXA data sources:
- Row counts and schema
- Data quality (nulls, duplicates)
- Key joins and alignment
- Basic statistics
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any

import pandas as pd

logger = logging.getLogger(__name__)


def _get_repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _safe_nunique(series: pd.Series) -> int:
    """Safe nunique that handles nulls."""
    return series.dropna().nunique()


def analyze_dataframe(df: pd.DataFrame, name: str) -> Dict[str, Any]:
    """Comprehensive analysis of a DataFrame."""
    analysis = {
        "name": name,
        "rows": len(df),
        "columns": len(df.columns),
        "memory_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
        "duplicates": df.duplicated().sum(),
    }

    # Column-level analysis
    col_analysis = []
    for col in df.columns:
        col_info = {
            "column": col,
            "dtype": str(df[col].dtype),
            "null_count": df[col].isna().sum(),
            "null_pct": 100 * df[col].isna().mean(),
            "unique_count": _safe_nunique(df[col]),
        }

        # Numeric stats
        if pd.api.types.is_numeric_dtype(df[col]):
            col_info.update(
                {
                    "min": df[col].min(),
                    "max": df[col].max(),
                    "mean": df[col].mean(),
                    "std": df[col].std(),
                    "median": df[col].median(),
                }
            )

        col_analysis.append(col_info)

    analysis["columns_detail"] = col_analysis
    return analysis


def run_phase0_eda(output_dir: Path = None) -> Dict[str, Any]:
    """Run Phase 0 EDA: Data Overview."""

    repo_root = _get_repo_root()
    if output_dir is None:
        output_dir = repo_root / "outputs" / "analysis" / "cxa" / "eda" / "phase0_data_overview"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("PHASE 0 EDA: Data Overview & Alignment")
    logger.info("=" * 60)

    results = {}

    # Load all datasets
    feature_store = repo_root / "feature_store" / "cxa"

    datasets = {
        "shots": feature_store / "shots.parquet",
        "passes": feature_store / "passes.parquet",
        "possessions": feature_store / "possessions.parquet",
        "sequences": feature_store / "sequences.parquet",
        "action_sequences": feature_store / "action_sequences.parquet",
    }

    dfs = {}
    for name, path in datasets.items():
        if path.exists():
            dfs[name] = pd.read_parquet(path)
            logger.info(f"Loaded {name}: {len(dfs[name]):,} rows, {len(dfs[name].columns)} cols")
        else:
            logger.warning(f"Missing: {path}")

    # 1. Basic Overview
    logger.info("\n--- 1. Dataset Overview ---")
    overview_rows = []
    for name, df in dfs.items():
        analysis = analyze_dataframe(df, name)
        results[name] = analysis
        overview_rows.append(
            {
                "dataset": name,
                "rows": analysis["rows"],
                "columns": analysis["columns"],
                "memory_mb": round(analysis["memory_mb"], 2),
                "duplicates": analysis["duplicates"],
            }
        )

    overview_df = pd.DataFrame(overview_rows)
    overview_df.to_csv(output_dir / "dataset_overview.csv", index=False)
    logger.info(f"\n{overview_df.to_string(index=False)}")

    # 2. Null Analysis
    logger.info("\n--- 2. Null Analysis (columns with >0% nulls) ---")
    null_rows = []
    for name, df in dfs.items():
        for col in df.columns:
            null_pct = 100 * df[col].isna().mean()
            if null_pct > 0:
                null_rows.append(
                    {
                        "dataset": name,
                        "column": col,
                        "null_count": df[col].isna().sum(),
                        "null_pct": round(null_pct, 2),
                    }
                )

    if null_rows:
        null_df = pd.DataFrame(null_rows).sort_values("null_pct", ascending=False)
        null_df.to_csv(output_dir / "null_analysis.csv", index=False)
        logger.info("\nTop 20 columns with nulls:")
        print(null_df.head(20).to_string(index=False))
    else:
        logger.info("No null values found!")

    # 3. Key Alignment Check
    logger.info("\n--- 3. Key Alignment Check ---")

    # Shots vs Sequences alignment
    if "shots" in dfs and "sequences" in dfs:
        shots_ids = set(dfs["shots"]["shot_id"].dropna())
        seq_ids = set(dfs["sequences"]["shot_id"].dropna())

        alignment = {
            "shots_total": len(shots_ids),
            "sequences_total": len(seq_ids),
            "overlap": len(shots_ids & seq_ids),
            "in_shots_not_seq": len(shots_ids - seq_ids),
            "in_seq_not_shots": len(seq_ids - shots_ids),
        }
        logger.info(f"Shots vs Sequences: {alignment}")

    # Shots vs Action Sequences alignment
    if "shots" in dfs and "action_sequences" in dfs:
        action_seq_ids = set(dfs["action_sequences"]["shot_id"].dropna())

        alignment2 = {
            "shots_total": len(shots_ids),
            "action_seq_total": len(action_seq_ids),
            "overlap": len(shots_ids & action_seq_ids),
            "in_shots_not_action": len(shots_ids - action_seq_ids),
            "in_action_not_shots": len(action_seq_ids - shots_ids),
        }
        logger.info(f"Shots vs Action Sequences: {alignment2}")

    # Goals alignment
    if "shots" in dfs:
        shots_goals = dfs["shots"]["is_goal"].sum()
        logger.info(f"\nGoals in shots.parquet: {shots_goals}")

    if "sequences" in dfs:
        seq_goals = (
            dfs["sequences"]["is_goal"].sum()
            if "is_goal" in dfs["sequences"].columns
            else len(dfs["sequences"])
        )
        logger.info(f"Goals in sequences.parquet: {seq_goals}")

    if "action_sequences" in dfs:
        action_goals = dfs["action_sequences"]["is_goal"].sum()
        logger.info(f"Goals in action_sequences.parquet: {action_goals}")

    # 4. Match/Team/Player Coverage
    logger.info("\n--- 4. Coverage Analysis ---")

    if "shots" in dfs:
        coverage = {
            "unique_matches": dfs["shots"]["match_id"].nunique(),
            "unique_teams": dfs["shots"]["team_id"].nunique(),
            "unique_players": (
                dfs["shots"]["player_id"].nunique()
                if "player_id" in dfs["shots"].columns
                else "N/A"
            ),
        }
        logger.info(f"Coverage (from shots): {coverage}")

    if "passes" in dfs:
        pass_coverage = {
            "unique_matches": dfs["passes"]["match_id"].nunique(),
            "unique_teams": dfs["passes"]["team_id"].nunique(),
            "unique_players": dfs["passes"]["player_id"].nunique(),
        }
        logger.info(f"Coverage (from passes): {pass_coverage}")

    # 5. Save detailed column analysis
    logger.info("\n--- 5. Saving Column Details ---")
    for name, analysis in results.items():
        col_df = pd.DataFrame(analysis["columns_detail"])
        col_df.to_csv(output_dir / f"{name}_columns.csv", index=False)

    logger.info(f"\nPhase 0 EDA complete. Outputs saved to {output_dir}")

    return results


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    run_phase0_eda()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
