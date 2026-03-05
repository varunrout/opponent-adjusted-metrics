"""Phase 6 EDA: cXA-xG Validation Visualizations.

Visualizations specific to cXA-xG model validation:
- Credit distribution analysis
- Position decay curve
- Top contributors (players/teams)
- Credit by action type
- Validation against xG totals
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any, List

import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


def _get_repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _convert_to_long_format(actions: pd.DataFrame) -> pd.DataFrame:
    """Convert wide-format action_sequences to long-format scored actions."""
    rows = []
    
    for _, seq in actions.iterrows():
        shot_xg = seq['shot_xg']
        num_actions = int(seq['num_actions']) if pd.notna(seq['num_actions']) else 0
        
        if num_actions == 0 or pd.isna(shot_xg) or shot_xg <= 0:
            continue
        
        # Calculate softmax weights for positions (position 1 = closest to shot)
        positions = list(range(1, num_actions + 1))
        weights = np.array([np.exp(-0.5 * (p - 1)) for p in positions])  # Decay from position 1
        weights = weights / weights.sum()
        
        for pos in range(1, min(num_actions + 1, 6)):  # Max 5 actions
            action_type = seq.get(f'action{pos}_type')
            if pd.isna(action_type):
                continue
                
            credit = shot_xg * weights[pos - 1] if pos <= len(weights) else 0
            
            rows.append({
                'sequence_id': seq['sequence_id'],
                'shot_id': seq['shot_id'],
                'shot_xg': shot_xg,
                'is_goal': seq.get('is_goal', False),
                'position_in_window': pos,
                'action_type': action_type,
                'player_id': seq.get(f'action{pos}_player_id'),
                'player_name': seq.get(f'action{pos}_player_name'),
                'team_id': seq.get('team_id'),
                'start_x': seq.get(f'action{pos}_start_x'),
                'start_y': seq.get(f'action{pos}_start_y'),
                'end_x': seq.get(f'action{pos}_end_x'),
                'end_y': seq.get(f'action{pos}_end_y'),
                'cxa_xg': credit,
            })
    
    return pd.DataFrame(rows)


def plot_credit_distribution(scored_actions: pd.DataFrame, output_dir: Path):
    """Plot distribution of cXA-xG credits."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    credits = scored_actions['cxa_xg'].dropna()
    
    # 1. Raw distribution
    axes[0, 0].hist(credits, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(credits.mean(), color='red', linestyle='--', 
                       label=f'Mean: {credits.mean():.4f}')
    axes[0, 0].set_title('cXA-xG Credit Distribution')
    axes[0, 0].set_xlabel('cXA-xG')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].legend()
    
    # 2. Log scale
    credits_log = np.log10(credits + 0.0001)
    axes[0, 1].hist(credits_log, bins=50, edgecolor='black', alpha=0.7, color='coral')
    axes[0, 1].set_title('cXA-xG Distribution (log10 scale)')
    axes[0, 1].set_xlabel('log10(cXA-xG + 0.0001)')
    axes[0, 1].set_ylabel('Count')
    
    # 3. Cumulative distribution
    sorted_credits = np.sort(credits)[::-1]  # Descending
    cumsum = np.cumsum(sorted_credits) / sorted_credits.sum()
    axes[1, 0].plot(range(len(cumsum)), cumsum)
    axes[1, 0].axhline(0.5, color='red', linestyle='--', alpha=0.5)
    axes[1, 0].axhline(0.8, color='red', linestyle='--', alpha=0.5)
    axes[1, 0].set_title('Cumulative Credit Distribution')
    axes[1, 0].set_xlabel('Action Rank (by cXA-xG)')
    axes[1, 0].set_ylabel('Cumulative % of Total Credit')
    
    # 4. Box plot by position
    if 'position_in_window' in scored_actions.columns:
        pos_data = scored_actions[scored_actions['position_in_window'] <= 5]
        pos_data.boxplot(column='cxa_xg', by='position_in_window', ax=axes[1, 1])
        axes[1, 1].set_title('cXA-xG by Position')
        axes[1, 1].set_xlabel('Position (1=last before shot)')
        axes[1, 1].set_ylabel('cXA-xG')
        plt.suptitle('')
    
    plt.tight_layout()
    plt.savefig(output_dir / "credit_distribution.png", dpi=150)
    plt.close()
    
    # Stats
    logger.info(f"\ncXA-xG Credit Statistics:")
    logger.info(f"  Total credit: {credits.sum():.2f}")
    logger.info(f"  Mean: {credits.mean():.4f}")
    logger.info(f"  Median: {credits.median():.4f}")
    logger.info(f"  Max: {credits.max():.4f}")


def plot_position_decay_curve(scored_actions: pd.DataFrame, output_dir: Path):
    """Visualize how credit decays by position in window."""
    
    if 'position_in_window' not in scored_actions.columns:
        logger.warning("No position_in_window column")
        return
    
    pos_stats = scored_actions.groupby('position_in_window').agg(
        n_actions=('cxa_xg', 'count'),
        total_credit=('cxa_xg', 'sum'),
        mean_credit=('cxa_xg', 'mean'),
    ).reset_index()
    
    pos_stats = pos_stats[pos_stats['position_in_window'] <= 10]
    pos_stats['pct_of_total'] = 100 * pos_stats['total_credit'] / pos_stats['total_credit'].sum()
    
    pos_stats.to_csv(output_dir / "credit_by_position.csv", index=False)
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # 1. Total credit by position
    axes[0].bar(pos_stats['position_in_window'], pos_stats['total_credit'], color='steelblue')
    axes[0].set_title('Total cXA-xG by Position')
    axes[0].set_xlabel('Position (1=last before shot)')
    axes[0].set_ylabel('Total cXA-xG')
    
    # 2. Mean credit by position (decay curve)
    axes[1].plot(pos_stats['position_in_window'], pos_stats['mean_credit'], 
                 marker='o', linewidth=2, color='coral')
    axes[1].set_title('Mean cXA-xG by Position (Decay Curve)')
    axes[1].set_xlabel('Position')
    axes[1].set_ylabel('Mean cXA-xG')
    axes[1].grid(True, alpha=0.3)
    
    # 3. Percentage of total credit
    axes[2].bar(pos_stats['position_in_window'], pos_stats['pct_of_total'], color='green')
    axes[2].set_title('% of Total Credit by Position')
    axes[2].set_xlabel('Position')
    axes[2].set_ylabel('% of Total cXA-xG')
    
    # Add annotation for position 1
    pos1_pct = pos_stats[pos_stats['position_in_window'] == 1]['pct_of_total'].values[0]
    axes[2].annotate(f'{pos1_pct:.1f}%', xy=(1, pos1_pct), fontsize=12, ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / "position_decay_curve.png", dpi=150)
    plt.close()
    
    logger.info(f"\nPosition 1 gets {pos1_pct:.1f}% of total credit")


def plot_credit_by_action_type(scored_actions: pd.DataFrame, output_dir: Path):
    """Compare credit allocation between passes and carries."""
    
    if 'action_type' not in scored_actions.columns:
        logger.warning("No action_type column")
        return
    
    type_stats = scored_actions.groupby('action_type').agg(
        n_actions=('cxa_xg', 'count'),
        total_credit=('cxa_xg', 'sum'),
        mean_credit=('cxa_xg', 'mean'),
    ).reset_index()
    
    type_stats['pct_of_total'] = 100 * type_stats['total_credit'] / type_stats['total_credit'].sum()
    type_stats['pct_of_actions'] = 100 * type_stats['n_actions'] / type_stats['n_actions'].sum()
    
    type_stats.to_csv(output_dir / "credit_by_action_type.csv", index=False)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Action count
    axes[0].pie(type_stats['n_actions'], labels=type_stats['action_type'], 
                autopct='%1.1f%%', colors=['steelblue', 'coral', 'green'][:len(type_stats)])
    axes[0].set_title('Action Type Distribution')
    
    # 2. Credit share
    axes[1].pie(type_stats['total_credit'], labels=type_stats['action_type'],
                autopct='%1.1f%%', colors=['steelblue', 'coral', 'green'][:len(type_stats)])
    axes[1].set_title('cXA-xG Credit Share')
    
    # 3. Mean credit comparison
    axes[2].bar(type_stats['action_type'], type_stats['mean_credit'], 
                color=['steelblue', 'coral', 'green'][:len(type_stats)])
    axes[2].set_title('Mean cXA-xG by Action Type')
    axes[2].set_ylabel('Mean cXA-xG')
    
    plt.tight_layout()
    plt.savefig(output_dir / "credit_by_action_type.png", dpi=150)
    plt.close()
    
    logger.info("\nCredit by Action Type:")
    print(type_stats.to_string(index=False))


def plot_top_contributors(scored_actions: pd.DataFrame, output_dir: Path):
    """Show top players and teams by cXA-xG."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Top players
    if 'player_name' in scored_actions.columns:
        player_totals = scored_actions.groupby('player_name')['cxa_xg'].sum().sort_values(ascending=False)
        top_players = player_totals.head(15)
        
        axes[0].barh(range(len(top_players)), top_players.values, color='steelblue')
        axes[0].set_yticks(range(len(top_players)))
        axes[0].set_yticklabels(top_players.index)
        axes[0].set_xlabel('Total cXA-xG')
        axes[0].set_title('Top 15 Players by cXA-xG')
        axes[0].invert_yaxis()
        
        player_totals.head(20).to_csv(output_dir / "top_players_cxa_xg.csv")
    
    # Top teams
    if 'team_id' in scored_actions.columns:
        team_totals = scored_actions.groupby('team_id')['cxa_xg'].sum().sort_values(ascending=False)
        top_teams = team_totals.head(15)
        
        axes[1].barh(range(len(top_teams)), top_teams.values, color='coral')
        axes[1].set_yticks(range(len(top_teams)))
        axes[1].set_yticklabels([f'Team {t}' for t in top_teams.index])
        axes[1].set_xlabel('Total cXA-xG')
        axes[1].set_title('Top 15 Teams by cXA-xG')
        axes[1].invert_yaxis()
        
        team_totals.to_csv(output_dir / "team_cxa_xg.csv")
    
    plt.tight_layout()
    plt.savefig(output_dir / "top_contributors.png", dpi=150)
    plt.close()


def plot_xg_calibration_check(scored_actions: pd.DataFrame, shots: pd.DataFrame, output_dir: Path):
    """Verify that sum of cXA-xG equals sum of xG per shot."""
    
    # Sum credit per shot
    credit_per_shot = scored_actions.groupby('shot_id')['cxa_xg'].sum().reset_index()
    credit_per_shot.columns = ['shot_id', 'total_credit']
    
    # Merge with shot xG
    merged = credit_per_shot.merge(
        shots[['shot_id', 'statsbomb_xg', 'is_goal']], 
        on='shot_id', 
        how='left'
    )
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # 1. Credit vs xG scatter
    axes[0].scatter(merged['statsbomb_xg'], merged['total_credit'], alpha=0.3)
    max_val = max(merged['statsbomb_xg'].max(), merged['total_credit'].max())
    axes[0].plot([0, max_val], [0, max_val], 'r--', label='Perfect match')
    axes[0].set_xlabel('Shot xG')
    axes[0].set_ylabel('Sum of Action Credits')
    axes[0].set_title('Credit Calibration: Credit vs xG')
    axes[0].legend()
    
    # 2. Residuals
    merged['residual'] = merged['total_credit'] - merged['statsbomb_xg']
    axes[1].hist(merged['residual'], bins=50, edgecolor='black', alpha=0.7)
    axes[1].axvline(0, color='red', linestyle='--')
    axes[1].set_xlabel('Residual (Credit - xG)')
    axes[1].set_ylabel('Count')
    axes[1].set_title(f'Residuals (Mean: {merged["residual"].mean():.6f})')
    
    # 3. Totals comparison
    totals = {
        'Total xG': merged['statsbomb_xg'].sum(),
        'Total Credit': merged['total_credit'].sum(),
        'Actual Goals': merged['is_goal'].sum(),
    }
    axes[2].bar(totals.keys(), totals.values(), color=['steelblue', 'coral', 'green'])
    axes[2].set_ylabel('Total')
    axes[2].set_title('Calibration Check: Totals')
    for i, (k, v) in enumerate(totals.items()):
        axes[2].text(i, v + 5, f'{v:.1f}', ha='center')
    
    plt.tight_layout()
    plt.savefig(output_dir / "xg_calibration_check.png", dpi=150)
    plt.close()
    
    logger.info(f"\nCalibration Check:")
    for k, v in totals.items():
        logger.info(f"  {k}: {v:.2f}")
    
    merged.to_csv(output_dir / "shot_credit_vs_xg.csv", index=False)


def plot_credit_heatmap(scored_actions: pd.DataFrame, output_dir: Path):
    """Spatial heatmap of where credit is earned."""
    
    if 'end_x' not in scored_actions.columns or 'end_y' not in scored_actions.columns:
        logger.warning("No end_x/end_y columns")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Action locations (unweighted)
    h1 = axes[0].hist2d(
        scored_actions['end_x'], scored_actions['end_y'],
        bins=20, cmap='YlOrRd'
    )
    axes[0].set_title('Action Locations')
    axes[0].set_xlabel('End X')
    axes[0].set_ylabel('End Y')
    plt.colorbar(h1[3], ax=axes[0])
    
    # 2. Credit-weighted heatmap
    # Create bins and weight by credit
    x_bins = np.linspace(scored_actions['end_x'].min(), scored_actions['end_x'].max(), 21)
    y_bins = np.linspace(scored_actions['end_y'].min(), scored_actions['end_y'].max(), 21)
    
    h2 = axes[1].hist2d(
        scored_actions['end_x'], scored_actions['end_y'],
        bins=[x_bins, y_bins], 
        weights=scored_actions['cxa_xg'],
        cmap='YlOrRd'
    )
    axes[1].set_title('Credit-Weighted Heatmap')
    axes[1].set_xlabel('End X')
    axes[1].set_ylabel('End Y')
    plt.colorbar(h2[3], ax=axes[1], label='Total cXA-xG')
    
    plt.tight_layout()
    plt.savefig(output_dir / "credit_heatmap.png", dpi=150)
    plt.close()


def run_phase6_eda(output_dir: Path = None) -> Dict[str, Any]:
    """Run Phase 6 EDA: cXA-xG Validation."""
    
    repo_root = _get_repo_root()
    if output_dir is None:
        output_dir = repo_root / "outputs" / "analysis" / "cxa" / "eda" / "phase6_cxa_xg_validation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("PHASE 6 EDA: cXA-xG Validation")
    logger.info("=" * 60)
    
    # Load and convert action_sequences to long format with credits
    actions = pd.read_parquet(repo_root / "feature_store" / "cxa" / "action_sequences.parquet")
    logger.info(f"Loaded {len(actions):,} action sequences")
    
    scored_actions = _convert_to_long_format(actions)
    logger.info(f"Converted to {len(scored_actions):,} scored actions")
    
    # Save for future use
    scored_actions.to_parquet(output_dir / "scored_actions.parquet")
    
    shots = pd.read_parquet(repo_root / "feature_store" / "cxa" / "shots.parquet")
    
    results = {}
    
    # 1. Credit Distribution
    logger.info("\n--- 1. Credit Distribution ---")
    plot_credit_distribution(scored_actions, output_dir)
    
    # 2. Position Decay Curve
    logger.info("\n--- 2. Position Decay Curve ---")
    plot_position_decay_curve(scored_actions, output_dir)
    
    # 3. Credit by Action Type
    logger.info("\n--- 3. Credit by Action Type ---")
    plot_credit_by_action_type(scored_actions, output_dir)
    
    # 4. Top Contributors
    logger.info("\n--- 4. Top Contributors ---")
    plot_top_contributors(scored_actions, output_dir)
    
    # 5. xG Calibration Check
    logger.info("\n--- 5. xG Calibration Check ---")
    plot_xg_calibration_check(scored_actions, shots, output_dir)
    
    # 6. Credit Heatmap
    logger.info("\n--- 6. Credit Heatmap ---")
    plot_credit_heatmap(scored_actions, output_dir)
    
    logger.info(f"\nPhase 6 EDA complete. Outputs saved to {output_dir}")
    
    return results


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    run_phase6_eda()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
