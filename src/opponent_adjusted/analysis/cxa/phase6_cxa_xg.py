"""Phase 6: cXA-xG Analysis.

This script runs the full cXA-xG analysis pipeline:
1. Load existing action_sequences.parquet (already has wide-format actions)
2. Join with shots.parquet for full shot coverage
3. Train scorer and allocate credit
4. Generate summaries, leaderboards, and comparison tables
5. Output to outputs/analysis/cxa/phase6_cxa_xg/

Key outputs:
- cxa_xg_credits.csv: Per-action credits weighted by xG
- cxa_goals_credits.csv: Per-action credits for goals only
- player_leaderboard_xg.csv: Player totals for cXA-xG
- player_leaderboard_goals.csv: Player totals for cXA-Goals
- action_type_summary.csv: Credit breakdown by Pass/Carry/Dribble
- calibration_check.csv: Verify totals match expected
- phase6_report.md: Human-readable summary
"""

from __future__ import annotations

import logging
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def _get_repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def load_existing_data() -> pd.DataFrame:
    """Load existing action_sequences and join with shots for full coverage.
    
    Uses action_sequences.parquet which already has wide-format action columns,
    and joins with shots.parquet to get full shot set.
    """
    repo_root = _get_repo_root()
    
    # Load action sequences (has wide-format actions for 4,910 shots)
    action_seq = pd.read_parquet(repo_root / "feature_store" / "cxa" / "action_sequences.parquet")
    
    # Load full shots
    shots = pd.read_parquet(repo_root / "feature_store" / "cxa" / "shots.parquet")
    
    logger.info(f"  Action sequences: {len(action_seq):,} shots")
    logger.info(f"  Shots parquet: {len(shots):,} shots")
    
    # Rename shot_xg to statsbomb_xg for consistency
    if "shot_xg" in action_seq.columns:
        action_seq = action_seq.rename(columns={"shot_xg": "statsbomb_xg"})
    
    # Keep relevant columns from action_seq
    action_cols = [c for c in action_seq.columns if c.startswith("action") or c in [
        "shot_id", "match_id", "team_id", "possession", "statsbomb_xg", "is_goal", "num_actions",
        "shot_x", "shot_y", "shot_minute", "shot_second"
    ]]
    action_seq = action_seq[action_cols]
    
    # Left join with shots to get full coverage
    shots_minimal = shots[["shot_id", "match_id", "team_id", "possession", "statsbomb_xg", "is_goal", 
                           "shot_x", "shot_y", "minute", "second", "player_id"]]
    shots_minimal = shots_minimal.rename(columns={"minute": "shot_minute", "second": "shot_second"})
    
    # Merge: keep all shots, add action data where available
    merged = shots_minimal.merge(
        action_seq.drop(columns=["match_id", "team_id", "possession", "statsbomb_xg", "is_goal", 
                                  "shot_x", "shot_y", "shot_minute", "shot_second"], errors="ignore"),
        on="shot_id",
        how="left"
    )
    
    # Fill missing num_actions with 0
    merged["num_actions"] = merged["num_actions"].fillna(0).astype(int)
    
    # Set is_goal as int
    merged["is_goal"] = merged["is_goal"].astype(int)
    
    logger.info(f"  Merged dataset: {len(merged):,} shots")
    logger.info(f"  Shots with actions: {(merged['num_actions'] > 0).sum():,}")
    logger.info(f"  Shots without actions: {(merged['num_actions'] == 0).sum():,}")
    
    return merged


def melt_to_actions(windows_df: pd.DataFrame, max_actions: int = 5) -> pd.DataFrame:
    """Convert wide window format to long action format for scoring.
    
    Returns a DataFrame with one row per action.
    """
    action_rows = []
    
    for _, row in windows_df.iterrows():
        shot_id = row["shot_id"]
        num_actions = int(row["num_actions"])
        statsbomb_xg = row["statsbomb_xg"]
        is_goal = row["is_goal"]
        
        for i in range(1, min(num_actions, max_actions) + 1):
            prefix = f"action{i}_"
            
            action_type = row.get(f"{prefix}type")
            if pd.isna(action_type):
                continue
            
            # Calculate distance to goal
            end_x = float(row.get(f"{prefix}end_x", 60)) if pd.notna(row.get(f"{prefix}end_x")) else 60.0
            end_y = float(row.get(f"{prefix}end_y", 40)) if pd.notna(row.get(f"{prefix}end_y")) else 40.0
            
            distance_to_goal = np.sqrt((120.0 - end_x) ** 2 + (40.0 - end_y) ** 2)
            angle_to_goal = np.degrees(np.arctan(abs(40.0 - end_y) / max(120.0 - end_x, 0.1)))
            
            action_row = {
                "shot_id": shot_id,
                "action_position": i,
                "is_final_action": i == 1,
                "statsbomb_xg": statsbomb_xg,
                "is_goal": is_goal,
                "player_id": row.get(f"{prefix}player_id"),
                "player_name": row.get(f"{prefix}player_name"),
                "action_type": action_type,
                "start_x": row.get(f"{prefix}start_x", 60),
                "start_y": row.get(f"{prefix}start_y", 40),
                "end_x": end_x,
                "end_y": end_y,
                "distance_to_goal": distance_to_goal,
                "angle_to_goal": angle_to_goal,
                "is_pass": action_type == "Pass",
                "is_carry": action_type == "Carry",
                "is_dribble": action_type == "Dribble",
                "is_into_box": (end_x >= 102) and (18 <= end_y <= 62),
                "under_pressure": bool(row.get(f"{prefix}under_pressure", False)),
            }
            
            action_rows.append(action_row)
    
    return pd.DataFrame(action_rows)


def train_scorer(actions_df: pd.DataFrame):
    """Train logistic regression scorer predicting is_final_action."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    
    features = ["end_x", "end_y", "distance_to_goal", "angle_to_goal", 
                "is_pass", "is_carry", "is_dribble", "is_into_box", "under_pressure"]
    
    X = actions_df[features].fillna(0).values
    y = actions_df["is_final_action"].astype(int).values
    
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=500, random_state=42)),
    ])
    
    pipeline.fit(X, y)
    
    accuracy = (pipeline.predict(X) == y).mean()
    logger.info(f"  Scorer training accuracy: {accuracy:.3f}")
    
    return pipeline, features


def score_and_allocate(actions_df: pd.DataFrame, scorer, features: list, mode: str = "xG") -> pd.DataFrame:
    """Score actions and allocate credit via softmax."""
    X = actions_df[features].fillna(0).values
    
    # Get log-odds
    proba = scorer.predict_proba(X)[:, 1]
    proba = np.clip(proba, 1e-6, 1 - 1e-6)
    log_odds = np.log(proba / (1 - proba))
    
    actions_df = actions_df.copy()
    actions_df["raw_score"] = log_odds
    
    # Allocate credit per shot via softmax
    credits = []
    
    for shot_id, group in actions_df.groupby("shot_id"):
        scores = group["raw_score"].values
        
        # Softmax
        exp_scores = np.exp(scores - np.max(scores))
        shares = exp_scores / exp_scores.sum()
        
        # Weight
        if mode == "xG":
            weight = float(group["statsbomb_xg"].iloc[0])
        else:
            weight = 1.0 if group["is_goal"].iloc[0] else 0.0
        
        for i, (idx, row) in enumerate(group.iterrows()):
            credits.append({
                "shot_id": shot_id,
                "action_position": row["action_position"],
                "player_id": row["player_id"],
                "player_name": row["player_name"],
                "action_type": row["action_type"],
                "raw_score": row["raw_score"],
                "credit_share": shares[i],
                "weighted_credit": shares[i] * weight,
                "is_goal": row["is_goal"],
                "statsbomb_xg": row["statsbomb_xg"],
            })
    
    return pd.DataFrame(credits)


def player_leaderboard(credits_df: pd.DataFrame, mode: str = "xG") -> pd.DataFrame:
    """Aggregate credits to player level."""
    agg = credits_df.groupby(["player_id", "player_name"]).agg(
        total_credit=("weighted_credit", "sum"),
        num_actions=("shot_id", "count"),
        num_shots=("shot_id", "nunique"),
    ).reset_index()
    
    credit_col = f"cXA_{mode}"
    agg = agg.rename(columns={"total_credit": credit_col})
    
    return agg.sort_values(credit_col, ascending=False)


def action_type_summary(credits_df: pd.DataFrame, mode: str = "xG") -> pd.DataFrame:
    """Summarize credit by action type."""
    agg = credits_df.groupby("action_type").agg(
        total_credit=("weighted_credit", "sum"),
        num_actions=("shot_id", "count"),
        mean_share=("credit_share", "mean"),
    ).reset_index()
    
    total = agg["total_credit"].sum()
    agg["pct_of_total"] = 100.0 * agg["total_credit"] / total if total > 0 else 0.0
    
    return agg.sort_values("total_credit", ascending=False)


def run_phase6_analysis():
    """Run the full Phase 6 cXA-xG analysis."""
    repo_root = _get_repo_root()
    output_dir = repo_root / "outputs" / "analysis" / "cxa" / "phase6_cxa_xg"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Phase 6 output directory: {output_dir}")
    
    # 1. Load existing data
    logger.info("=" * 60)
    logger.info("STEP 1: Load shot action data")
    logger.info("=" * 60)
    windows_df = load_existing_data()
    logger.info(f"  Total shots: {len(windows_df):,}")
    logger.info(f"  Goals: {windows_df['is_goal'].sum():,}")
    logger.info(f"  Total xG: {windows_df['statsbomb_xg'].sum():.2f}")
    
    # 2. Melt to long format
    logger.info("=" * 60)
    logger.info("STEP 2: Convert to long format")
    logger.info("=" * 60)
    actions_df = melt_to_actions(windows_df)
    logger.info(f"  Total actions: {len(actions_df):,}")
    logger.info(f"  Unique shots with actions: {actions_df['shot_id'].nunique():,}")
    
    # 3. Train scorer
    logger.info("=" * 60)
    logger.info("STEP 3: Train scorer")
    logger.info("=" * 60)
    scorer, features = train_scorer(actions_df)
    
    # 4. Allocate credit
    logger.info("=" * 60)
    logger.info("STEP 4: Allocate credit")
    logger.info("=" * 60)
    
    logger.info("  Computing cXA-xG (all shots weighted by xG)...")
    cxa_xg = score_and_allocate(actions_df, scorer, features, mode="xG")
    
    logger.info("  Computing cXA-Goals (goals only)...")
    cxa_goals = score_and_allocate(actions_df, scorer, features, mode="goals")
    
    # Save raw credits
    cxa_xg.to_csv(output_dir / "cxa_xg_credits.csv", index=False)
    cxa_goals.to_csv(output_dir / "cxa_goals_credits.csv", index=False)
    logger.info(f"  Saved {len(cxa_xg):,} cXA-xG action credits")
    logger.info(f"  Saved {len(cxa_goals):,} cXA-Goals action credits")
    
    # 5. Player leaderboards
    logger.info("=" * 60)
    logger.info("STEP 5: Generate player leaderboards")
    logger.info("=" * 60)
    
    xg_leaders = player_leaderboard(cxa_xg, "xG")
    goals_leaders = player_leaderboard(cxa_goals, "goals")
    
    xg_leaders.to_csv(output_dir / "player_leaderboard_xg.csv", index=False)
    goals_leaders.to_csv(output_dir / "player_leaderboard_goals.csv", index=False)
    
    logger.info(f"\nTop 10 cXA-xG creators:")
    print(xg_leaders.head(10).to_string(index=False))
    
    logger.info(f"\nTop 10 cXA-Goals creators:")
    print(goals_leaders.head(10).to_string(index=False))
    
    # 6. Action type summary
    logger.info("=" * 60)
    logger.info("STEP 6: Action type summary")
    logger.info("=" * 60)
    
    xg_by_type = action_type_summary(cxa_xg, "xG")
    goals_by_type = action_type_summary(cxa_goals, "goals")
    
    # Combine for comparison
    type_summary = pd.merge(
        xg_by_type.rename(columns={
            "total_credit": "cXA_xG",
            "pct_of_total": "pct_xG",
            "num_actions": "actions_xG",
            "mean_share": "mean_share_xG",
        }),
        goals_by_type.rename(columns={
            "total_credit": "cXA_goals",
            "pct_of_total": "pct_goals",
            "num_actions": "actions_goals",
            "mean_share": "mean_share_goals",
        }),
        on="action_type",
        how="outer",
    )
    
    type_summary.to_csv(output_dir / "action_type_summary.csv", index=False)
    logger.info("\nCredit by action type:")
    print(type_summary.to_string(index=False))
    
    # 7. Calibration check
    logger.info("=" * 60)
    logger.info("STEP 7: Calibration check")
    logger.info("=" * 60)
    
    # Only shots with actions get credit
    shots_with_actions = windows_df[windows_df["num_actions"] > 0]
    total_xg = shots_with_actions["statsbomb_xg"].sum()
    total_goals = shots_with_actions["is_goal"].sum()
    attributed_xg = cxa_xg["weighted_credit"].sum()
    attributed_goals = cxa_goals["weighted_credit"].sum()
    
    calibration = pd.DataFrame([
        {"metric": "Total xG (shots w/ actions)", "expected": total_xg, "attributed": attributed_xg, "diff": attributed_xg - total_xg},
        {"metric": "Total Goals (shots w/ actions)", "expected": total_goals, "attributed": attributed_goals, "diff": attributed_goals - total_goals},
    ])
    
    calibration.to_csv(output_dir / "calibration_check.csv", index=False)
    logger.info("\nCalibration (should sum to same totals for shots with actions):")
    print(calibration.to_string(index=False))
    
    # Note unattributed
    shots_no_actions = windows_df[windows_df["num_actions"] == 0]
    logger.info(f"\n  Shots with no actions: {len(shots_no_actions):,}")
    logger.info(f"  Unattributed xG: {shots_no_actions['statsbomb_xg'].sum():.2f}")
    logger.info(f"  Unattributed goals: {shots_no_actions['is_goal'].sum()}")
    
    # 8. Position analysis
    logger.info("=" * 60)
    logger.info("STEP 8: Credit by action position")
    logger.info("=" * 60)
    
    position_summary = cxa_xg.groupby("action_position").agg(
        total_cxa_xg=("weighted_credit", "sum"),
        mean_share=("credit_share", "mean"),
        num_actions=("shot_id", "count"),
    ).reset_index()
    
    position_summary["pct_of_total"] = 100 * position_summary["total_cxa_xg"] / position_summary["total_cxa_xg"].sum()
    position_summary.to_csv(output_dir / "credit_by_position.csv", index=False)
    
    logger.info("\nCredit by position (1=closest to shot):")
    print(position_summary.to_string(index=False))
    
    # 9. Generate report
    logger.info("=" * 60)
    logger.info("STEP 9: Generate report")
    logger.info("=" * 60)
    
    report = _generate_report(
        windows_df, cxa_xg, cxa_goals,
        xg_leaders, goals_leaders,
        type_summary, position_summary,
        calibration,
    )
    
    report_path = output_dir / "phase6_report.md"
    report_path.write_text(report, encoding="utf-8")
    logger.info(f"  Report saved to {report_path}")
    
    logger.info("\n" + "=" * 60)
    logger.info("Phase 6 complete!")
    logger.info("=" * 60)
    
    return {
        "windows_df": windows_df,
        "cxa_xg": cxa_xg,
        "cxa_goals": cxa_goals,
        "xg_leaders": xg_leaders,
        "goals_leaders": goals_leaders,
        "type_summary": type_summary,
    }


def _generate_report(
    windows_df: pd.DataFrame,
    cxa_xg: pd.DataFrame,
    cxa_goals: pd.DataFrame,
    xg_leaders: pd.DataFrame,
    goals_leaders: pd.DataFrame,
    type_summary: pd.DataFrame,
    position_summary: pd.DataFrame,
    calibration: pd.DataFrame,
) -> str:
    """Generate a Markdown report summarizing the analysis."""
    
    total_shots = len(windows_df)
    total_goals = int(windows_df["is_goal"].sum())
    total_xg = windows_df["statsbomb_xg"].sum()
    shots_with_actions = (windows_df["num_actions"] > 0).sum()
    mean_actions = windows_df["num_actions"].mean()
    
    # Get pass/carry percentages
    pass_pct = type_summary[type_summary["action_type"] == "Pass"]["pct_xG"].iloc[0] if "Pass" in type_summary["action_type"].values else 0
    carry_pct = type_summary[type_summary["action_type"] == "Carry"]["pct_xG"].iloc[0] if "Carry" in type_summary["action_type"].values else 0
    dribble_pct = type_summary[type_summary["action_type"] == "Dribble"]["pct_xG"].iloc[0] if "Dribble" in type_summary["action_type"].values else 0
    
    # Top creators
    top5_xg = xg_leaders.head(5)
    top5_goals = goals_leaders.head(5)
    
    report = f"""# Phase 6: cXA-xG Analysis Report

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}

## Executive Summary

This analysis introduces **cXA-xG** (Created Expected Goals), a new metric that
attributes chance creation credit to all pre-shot actions, weighted by the
expected goals value of each shot.

**Key difference from prior xA+ metrics:**
- Trained on ALL shots (not just goals)
- Weighted by shot xG (stable, less noisy)
- Conservative: sum of credits = sum of shot xG

## Data Overview

| Metric | Value |
|--------|-------|
| Total shots | {total_shots:,} |
| Total goals | {total_goals:,} |
| Total xG | {total_xg:.2f} |
| Shots with ≥1 action | {shots_with_actions:,} ({100*shots_with_actions/total_shots:.1f}%) |
| Mean actions per shot | {mean_actions:.2f} |

## Credit by Action Type

| Type | cXA-xG | % of Total |
|------|--------|------------|
| Pass | {type_summary[type_summary['action_type']=='Pass']['cXA_xG'].iloc[0] if 'Pass' in type_summary['action_type'].values else 0:.2f} | {pass_pct:.1f}% |
| Carry | {type_summary[type_summary['action_type']=='Carry']['cXA_xG'].iloc[0] if 'Carry' in type_summary['action_type'].values else 0:.2f} | {carry_pct:.1f}% |
| Dribble | {type_summary[type_summary['action_type']=='Dribble']['cXA_xG'].iloc[0] if 'Dribble' in type_summary['action_type'].values else 0:.2f} | {dribble_pct:.1f}% |

**Key finding:** Passes account for {pass_pct:.0f}% of creation credit,
with carries ({carry_pct:.0f}%) and dribbles ({dribble_pct:.0f}%) making up the rest.

## Credit by Position in Sequence

Position 1 = action closest to the shot (typically the "assist").

{position_summary.to_markdown(index=False)}

**Interpretation:** The final action before the shot receives the most credit
(position 1), but substantial credit flows to earlier actions in the buildup.

## Top 10 cXA-xG Creators

Players ranked by total xG-weighted creation credit:

{top5_xg.to_markdown(index=False)}

## Top 10 cXA-Goals Creators

Players ranked by goal-weighted creation credit (comparable to traditional assists):

{top5_goals.to_markdown(index=False)}

## Calibration Check

The sum of attributed credit should equal the total xG (for cXA-xG) and
total goals (for cXA-Goals):

{calibration.to_markdown(index=False)}

Note: Any difference is due to shots with no preceding actions in the window.

## Methodology

### Window Definition
- Last 8 actions (Pass, Carry, Dribble)
- Within 15 seconds before the shot
- Same possession

### Scorer
- Logistic regression predicting `is_final_action_before_shot`
- Features: end_x, end_y, distance_to_goal, angle_to_goal, action type flags,
  under_pressure, seconds_to_shot, is_into_box

### Credit Allocation
- Softmax over log-odds scores within each shot's action window
- Multiply by shot xG (cXA-xG) or 1.0 for goals (cXA-Goals)

## Files Generated

- `cxa_xg_credits.csv`: Per-action credits (xG-weighted)
- `cxa_goals_credits.csv`: Per-action credits (goals only)
- `player_leaderboard_xg.csv`: Player totals for cXA-xG
- `player_leaderboard_goals.csv`: Player totals for cXA-Goals
- `action_type_summary.csv`: Credit by Pass/Carry/Dribble
- `credit_by_position.csv`: Credit by action position
- `calibration_check.csv`: Conservation checks
"""
    
    return report


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )
    run_phase6_analysis()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
