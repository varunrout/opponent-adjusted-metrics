"""Phase 2 EDA: Shots Analysis.

Comprehensive EDA for shots data:
- Shot location patterns
- xG distribution and calibration
- Goal vs non-goal comparison
- Feature relationships
- Shot type analysis
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any, List

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

logger = logging.getLogger(__name__)


def _get_repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def plot_xg_distribution(df: pd.DataFrame, output_dir: Path):
    """Analyze xG distribution."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    xg = df['statsbomb_xg'].dropna()
    
    # 1. Histogram
    axes[0, 0].hist(xg, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(xg.mean(), color='red', linestyle='--', label=f'Mean: {xg.mean():.3f}')
    axes[0, 0].axvline(xg.median(), color='green', linestyle='--', label=f'Median: {xg.median():.3f}')
    axes[0, 0].set_title('xG Distribution')
    axes[0, 0].set_xlabel('xG')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].legend()
    
    # 2. Log-transformed histogram
    xg_log = np.log10(xg + 0.001)
    axes[0, 1].hist(xg_log, bins=50, edgecolor='black', alpha=0.7, color='coral')
    axes[0, 1].set_title('xG Distribution (log10 scale)')
    axes[0, 1].set_xlabel('log10(xG + 0.001)')
    axes[0, 1].set_ylabel('Count')
    
    # 3. xG by outcome
    goals = df[df['is_goal'] == 1]['statsbomb_xg']
    non_goals = df[df['is_goal'] == 0]['statsbomb_xg']
    
    axes[1, 0].hist(non_goals, bins=30, alpha=0.6, label=f'Non-goal (n={len(non_goals):,})', color='steelblue')
    axes[1, 0].hist(goals, bins=30, alpha=0.6, label=f'Goal (n={len(goals):,})', color='coral')
    axes[1, 0].set_title('xG by Outcome')
    axes[1, 0].set_xlabel('xG')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].legend()
    
    # 4. Cumulative xG
    xg_sorted = np.sort(xg)
    cumsum = np.cumsum(xg_sorted) / xg_sorted.sum()
    axes[1, 1].plot(range(len(xg_sorted)), cumsum)
    axes[1, 1].axhline(0.5, color='red', linestyle='--', alpha=0.5)
    axes[1, 1].axhline(0.8, color='red', linestyle='--', alpha=0.5)
    axes[1, 1].set_title('Cumulative xG (sorted)')
    axes[1, 1].set_xlabel('Shot Index (sorted by xG)')
    axes[1, 1].set_ylabel('Cumulative % of Total xG')
    
    plt.tight_layout()
    plt.savefig(output_dir / "xg_distribution.png", dpi=150)
    plt.close()


def plot_xg_calibration(df: pd.DataFrame, output_dir: Path):
    """Check xG calibration - predicted vs actual."""
    # Bin shots by xG
    df = df.copy()
    df['xg_bin'] = pd.cut(df['statsbomb_xg'], bins=10)
    
    calibration = df.groupby('xg_bin', observed=True).agg(
        n_shots=('shot_id', 'count'),
        mean_xg=('statsbomb_xg', 'mean'),
        actual_goals=('is_goal', 'sum'),
        actual_rate=('is_goal', 'mean'),
    ).reset_index()
    
    calibration['expected_goals'] = calibration['mean_xg'] * calibration['n_shots']
    calibration.to_csv(output_dir / "xg_calibration.csv", index=False)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    
    # Actual calibration
    ax.scatter(calibration['mean_xg'], calibration['actual_rate'], 
               s=calibration['n_shots'] / 10, alpha=0.7, label='Observed')
    
    ax.set_xlabel('Mean xG in Bin')
    ax.set_ylabel('Actual Goal Rate')
    ax.set_title('xG Calibration: Predicted vs Actual')
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output_dir / "xg_calibration_curve.png", dpi=150)
    plt.close()
    
    # Summary stats
    total_xg = df['statsbomb_xg'].sum()
    total_goals = df['is_goal'].sum()
    logger.info(f"\nxG Calibration Summary:")
    logger.info(f"  Total xG: {total_xg:.2f}")
    logger.info(f"  Total Goals: {total_goals}")
    logger.info(f"  Ratio (Goals/xG): {total_goals/total_xg:.3f}")
    
    return calibration


def plot_shot_locations(df: pd.DataFrame, output_dir: Path):
    """Plot shot location heatmaps."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # All shots
    h1 = axes[0].hist2d(df['shot_x'], df['shot_y'], bins=20, cmap='YlOrRd')
    axes[0].set_title(f'All Shots (n={len(df):,})')
    axes[0].set_xlim(80, 120)
    axes[0].set_ylim(0, 80)
    plt.colorbar(h1[3], ax=axes[0])
    
    # Goals only
    goals = df[df['is_goal'] == 1]
    h2 = axes[1].hist2d(goals['shot_x'], goals['shot_y'], bins=20, cmap='YlOrRd')
    axes[1].set_title(f'Goals Only (n={len(goals):,})')
    axes[1].set_xlim(80, 120)
    axes[1].set_ylim(0, 80)
    plt.colorbar(h2[3], ax=axes[1])
    
    # Conversion rate heatmap
    # Create grid
    x_bins = np.linspace(80, 120, 11)
    y_bins = np.linspace(0, 80, 11)
    
    df_temp = df.copy()
    df_temp['x_bin'] = pd.cut(df_temp['shot_x'], bins=x_bins, labels=range(10))
    df_temp['y_bin'] = pd.cut(df_temp['shot_y'], bins=y_bins, labels=range(10))
    
    conversion = df_temp.groupby(['x_bin', 'y_bin'], observed=True).agg(
        shots=('shot_id', 'count'),
        goals=('is_goal', 'sum')
    ).reset_index()
    conversion['rate'] = conversion['goals'] / conversion['shots']
    
    # Pivot for heatmap
    rate_matrix = conversion.pivot(index='y_bin', columns='x_bin', values='rate').fillna(0)
    
    im = axes[2].imshow(rate_matrix.values, cmap='RdYlGn', aspect='auto', 
                        origin='lower', vmin=0, vmax=0.5)
    axes[2].set_title('Conversion Rate by Location')
    axes[2].set_xlabel('X bins (80-120)')
    axes[2].set_ylabel('Y bins (0-80)')
    plt.colorbar(im, ax=axes[2], label='Goal Rate')
    
    plt.tight_layout()
    plt.savefig(output_dir / "shot_locations.png", dpi=150)
    plt.close()


def analyze_shot_types(df: pd.DataFrame, output_dir: Path):
    """Analyze shots by type, body part, etc."""
    analyses = []
    
    # By shot type
    if 'shot_type' in df.columns:
        by_type = df.groupby('shot_type').agg(
            shots=('shot_id', 'count'),
            goals=('is_goal', 'sum'),
            total_xg=('statsbomb_xg', 'sum'),
            mean_xg=('statsbomb_xg', 'mean'),
        ).reset_index()
        by_type['conversion_rate'] = 100 * by_type['goals'] / by_type['shots']
        by_type['xg_overperformance'] = by_type['goals'] - by_type['total_xg']
        by_type.to_csv(output_dir / "shots_by_type.csv", index=False)
        analyses.append(("Shot Type", by_type))
    
    # By body part
    if 'body_part' in df.columns:
        by_body = df.groupby('body_part').agg(
            shots=('shot_id', 'count'),
            goals=('is_goal', 'sum'),
            total_xg=('statsbomb_xg', 'sum'),
            mean_xg=('statsbomb_xg', 'mean'),
        ).reset_index()
        by_body['conversion_rate'] = 100 * by_body['goals'] / by_body['shots']
        by_body.to_csv(output_dir / "shots_by_body_part.csv", index=False)
        analyses.append(("Body Part", by_body))
    
    # By play pattern
    if 'play_pattern' in df.columns:
        by_pattern = df.groupby('play_pattern').agg(
            shots=('shot_id', 'count'),
            goals=('is_goal', 'sum'),
            total_xg=('statsbomb_xg', 'sum'),
            mean_xg=('statsbomb_xg', 'mean'),
        ).reset_index()
        by_pattern['conversion_rate'] = 100 * by_pattern['goals'] / by_pattern['shots']
        by_pattern = by_pattern.sort_values('shots', ascending=False)
        by_pattern.to_csv(output_dir / "shots_by_play_pattern.csv", index=False)
        analyses.append(("Play Pattern", by_pattern))
    
    for name, analysis_df in analyses:
        logger.info(f"\nShots by {name}:")
        print(analysis_df.head(10).to_string(index=False))
    
    return analyses


def run_phase2_eda(output_dir: Path = None) -> Dict[str, Any]:
    """Run Phase 2 EDA: Shots Analysis."""
    
    repo_root = _get_repo_root()
    if output_dir is None:
        output_dir = repo_root / "outputs" / "analysis" / "cxa" / "eda" / "phase2_shots"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("PHASE 2 EDA: Shots Analysis")
    logger.info("=" * 60)
    
    # Load shots
    shots = pd.read_parquet(repo_root / "feature_store" / "cxa" / "shots.parquet")
    logger.info(f"Loaded {len(shots):,} shots")
    
    results = {}
    
    # 1. Basic Stats
    logger.info("\n--- 1. Basic Statistics ---")
    logger.info(f"Total shots: {len(shots):,}")
    logger.info(f"Total goals: {shots['is_goal'].sum():,}")
    logger.info(f"Conversion rate: {100 * shots['is_goal'].mean():.2f}%")
    logger.info(f"Total xG: {shots['statsbomb_xg'].sum():.2f}")
    logger.info(f"Mean xG: {shots['statsbomb_xg'].mean():.3f}")
    
    results["basic_stats"] = {
        "total_shots": len(shots),
        "total_goals": int(shots['is_goal'].sum()),
        "conversion_rate": float(shots['is_goal'].mean()),
        "total_xg": float(shots['statsbomb_xg'].sum()),
    }
    
    # 2. xG Distribution
    logger.info("\n--- 2. xG Distribution ---")
    plot_xg_distribution(shots, output_dir)
    
    # 3. xG Calibration
    logger.info("\n--- 3. xG Calibration ---")
    calibration = plot_xg_calibration(shots, output_dir)
    results["calibration"] = calibration
    
    # 4. Shot Locations
    logger.info("\n--- 4. Shot Location Analysis ---")
    plot_shot_locations(shots, output_dir)
    
    # 5. Shot Types Analysis
    logger.info("\n--- 5. Shot Type Analysis ---")
    shot_analyses = analyze_shot_types(shots, output_dir)
    
    # 6. Goal vs Non-Goal Feature Comparison
    logger.info("\n--- 6. Goal vs Non-Goal Comparison ---")
    numeric_cols = ['shot_x', 'shot_y', 'statsbomb_xg']
    numeric_cols = [c for c in numeric_cols if c in shots.columns]
    
    comparison_rows = []
    for col in numeric_cols:
        goals = shots[shots['is_goal'] == 1][col].dropna()
        non_goals = shots[shots['is_goal'] == 0][col].dropna()
        
        t_stat, p_value = stats.ttest_ind(goals, non_goals)
        
        comparison_rows.append({
            "feature": col,
            "mean_goal": goals.mean(),
            "mean_non_goal": non_goals.mean(),
            "diff": goals.mean() - non_goals.mean(),
            "t_statistic": t_stat,
            "p_value": p_value,
        })
    
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(output_dir / "goal_vs_nongoal_comparison.csv", index=False)
    print(comparison_df.to_string(index=False))
    
    # 7. Distance to Goal Analysis
    logger.info("\n--- 7. Distance to Goal ---")
    shots['distance_to_goal'] = np.sqrt((120 - shots['shot_x'])**2 + (40 - shots['shot_y'])**2)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(shots[shots['is_goal'] == 0]['distance_to_goal'], bins=30, alpha=0.6, 
            label='Non-goal', density=True)
    ax.hist(shots[shots['is_goal'] == 1]['distance_to_goal'], bins=30, alpha=0.6,
            label='Goal', density=True)
    ax.set_xlabel('Distance to Goal')
    ax.set_ylabel('Density')
    ax.set_title('Shot Distance Distribution')
    ax.legend()
    plt.savefig(output_dir / "distance_to_goal.png", dpi=150)
    plt.close()
    
    logger.info(f"\nPhase 2 EDA complete. Outputs saved to {output_dir}")
    
    return results


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    run_phase2_eda()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
