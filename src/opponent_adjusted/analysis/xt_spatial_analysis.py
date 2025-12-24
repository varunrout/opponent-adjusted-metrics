"""
Expected Threat (xT) Spatial Analysis

Replaces zone-based analysis with threat-level progression analysis.
Analyzes how passes move through threat landscape and contribute to shot creation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

from opponent_adjusted.features.xt_model import compute_xt_for_passes, get_xt_model


def analyze_xt_threat_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze pass distribution across xT threat levels.
    
    Groups passes by threat level at destination (where shot is taken).
    
    Args:
        df: Pass-level DataFrame with xT columns
        
    Returns:
        DataFrame with threat level aggregations
    """
    df = compute_xt_for_passes(df) if 'xt_end' not in df.columns else df.copy()
    
    # Aggregate by threat levels at destination
    threat_analysis = []
    
    for end_threat in ['Low', 'Medium', 'High']:
        subset = df[df['xt_threat_end'] == end_threat]
        
        if len(subset) > 0:
            threat_analysis.append({
                'threat_level': end_threat,
                'passes': len(subset),
                'xa_plus_sum': subset['xa_plus'].sum(),
                'xa_plus_mean': subset['xa_plus'].mean(),
                'xt_mean': subset['xt_end'].mean(),
                'shot_assisted_pct': (subset.get('shot_assisted', pd.Series(0)) == 1).sum() / len(subset) if 'shot_assisted' in subset.columns else 0,
            })
    
    return pd.DataFrame(threat_analysis)


def analyze_xt_by_position(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze xT at shot location by position groups.
    
    Shows which positions create shots from which threat levels.
    
    Args:
        df: Pass-level DataFrame with position and xT columns
        
    Returns:
        DataFrame with position-based xT statistics
    """
    df = compute_xt_for_passes(df) if 'xt_end' not in df.columns else df.copy()
    
    # Add position if not present (infer from pass_end_x)
    if 'position_group' not in df.columns:
        df['position_group'] = df['pass_end_x'].apply(_infer_position_group)
    
    position_analysis = []
    
    for position in ['Defenders', 'Midfielders', 'Forwards']:
        subset = df[df['position_group'] == position]
        
        if len(subset) > 0:
            position_analysis.append({
                'position': position,
                'passes': len(subset),
                'xt_shot_location_mean': subset['xt_end'].mean(),
                'xa_plus_sum': subset['xa_plus'].sum(),
                'xa_plus_per_pass': subset['xa_plus'].mean(),
                'threat_distribution': _get_threat_distribution(subset, 'xt_threat_end'),
            })
    
    return pd.DataFrame(position_analysis)


def analyze_xt_by_team_cluster(df: pd.DataFrame, team_clusters: pd.DataFrame) -> Dict[int, pd.DataFrame]:
    """
    Analyze xT threat distribution by team style cluster.
    
    Compares how different playing styles progress through threat space.
    
    Args:
        df: Pass-level DataFrame
        team_clusters: Team cluster assignments
        
    Returns:
        Dictionary mapping cluster_id to xT analysis dictionary
    """
    df = compute_xt_for_passes(df) if 'xt_end' not in df.columns else df.copy()
    
    # Add cluster info
    cluster_map = dict(zip(team_clusters['team_id'], team_clusters['cluster']))
    df['cluster'] = df['team_id'].map(cluster_map)
    
    cluster_analyses = {}
    
    for cluster_id in sorted(df['cluster'].dropna().unique()):
        cluster_df = df[df['cluster'] == cluster_id]
        
        cluster_analyses[cluster_id] = {
            'passes': len(cluster_df),
            'xt_shot_location_mean': cluster_df['xt_end'].mean(),
            'xt_shot_location_median': cluster_df['xt_end'].median(),
            'high_threat_pct': (cluster_df['xt_threat_end'] == 'High').sum() / len(cluster_df),
            'low_threat_pct': (cluster_df['xt_threat_end'] == 'Low').sum() / len(cluster_df),
            'xa_plus_sum': cluster_df['xa_plus'].sum(),
            'threat_distribution': _get_threat_distribution(cluster_df, 'xt_threat_end'),
        }
    
    return cluster_analyses


def _infer_position_group(x: float) -> str:
    """Infer position from pass origin x-coordinate."""
    if x < 40:
        return 'Defenders'
    elif x < 80:
        return 'Midfielders'
    else:
        return 'Forwards'


def _get_threat_distribution(df: pd.DataFrame, column: str) -> Dict[str, float]:
    """Get distribution of threat levels."""
    total = len(df)
    return {
        'Low': (df[column] == 'Low').sum() / total,
        'Medium': (df[column] == 'Medium').sum() / total,
        'High': (df[column] == 'High').sum() / total,
    }


def plot_xt_threat_heatmap(
    threat_df: pd.DataFrame,
    output_path: Path,
    title: str = "Shot Location xT Distribution"
) -> None:
    """
    Visualize threat level distribution at shot locations.
    
    Bar chart showing shot frequency and xA contribution by threat level.
    
    Args:
        threat_df: Output from analyze_xt_threat_distribution()
        output_path: Path to save plot
        title: Plot title
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Shot distribution
    ax = axes[0]
    threat_order = ['Low', 'Medium', 'High']
    threat_df_ordered = threat_df.set_index('threat_level').reindex(threat_order).reset_index()
    
    colors = ['#FF6B6B', '#FFD93D', '#6BCB77']
    ax.bar(threat_df_ordered['threat_level'], threat_df_ordered['passes'], color=colors, alpha=0.8, edgecolor='black')
    ax.set_title('Shots by Threat Level', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Shots', fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # xA contribution
    ax = axes[1]
    ax.bar(threat_df_ordered['threat_level'], threat_df_ordered['xa_plus_sum'], color=colors, alpha=0.8, edgecolor='black')
    ax.set_title('xA+ Contribution by Threat Level', fontsize=12, fontweight='bold')
    ax.set_ylabel('Total xA+', fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_xt_gain_by_position(
    position_df: pd.DataFrame,
    output_path: Path
) -> None:
    """
    Visualize shot location xT by position.
    
    Bar chart showing threat level of shots created from each position.
    
    Args:
        position_df: Output from analyze_xt_by_position()
        output_path: Path to save plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # xT at shot location
    ax = axes[0]
    positions = position_df['position'].tolist()
    x_pos = np.arange(len(positions))
    
    ax.bar(x_pos, position_df['xt_shot_location_mean'], alpha=0.8, color='steelblue')
    
    ax.set_ylabel('Expected Threat', fontsize=11)
    ax.set_title('Shot Location xT by Position', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(positions)
    ax.grid(axis='y', alpha=0.3)
    
    # xA per pass by position
    ax = axes[1]
    ax.bar(positions, position_df['xa_plus_per_pass'], color='steelblue', alpha=0.8)
    ax.set_ylabel('xA+ per Pass', fontsize=11)
    ax.set_title('Shot Creation by Position', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_xt_by_cluster(
    cluster_analyses: Dict[int, Dict],
    team_labels: Dict[int, str],
    output_path: Path
) -> None:
    """
    Compare xT profiles across team style clusters.
    
    Multi-panel visualization showing threat progression by cluster.
    
    Args:
        cluster_analyses: Output from analyze_xt_by_team_cluster()
        team_labels: Mapping of cluster_id to label name
        output_path: Path to save plot
    """
    n_clusters = len(cluster_analyses)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, (cluster_id, stats) in enumerate(sorted(cluster_analyses.items())):
        ax = axes[idx]
        
        # Create threat progression visualization
        label = team_labels.get(cluster_id, f"Cluster {cluster_id}")
        
        # Show threat distribution
        threat_dist = stats['threat_distribution']
        threats = list(threat_dist.keys())
        values = [threat_dist[t] * 100 for t in threats]
        
        colors = {'Low': '#FF6B6B', 'Medium': '#FFD93D', 'High': '#6BCB77'}
        bar_colors = [colors.get(t, 'gray') for t in threats]
        
        ax.bar(threats, values, color=bar_colors, alpha=0.8, edgecolor='black')
        ax.set_ylim(0, 100)
        
        ax.set_ylabel('% of Shots', fontsize=10)
        ax.set_title(f"{label}\n({stats['passes']:,} shots, avg xT={stats['xt_shot_location_mean']:.3f})",
                    fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_clusters, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Expected Threat Profiles by Team Style', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def run_xt_spatial_analysis(
    df: pd.DataFrame,
    team_clusters: pd.DataFrame,
    output_dir: Path
) -> Tuple[pd.DataFrame, Dict, Dict]:
    """
    Run complete xT-based spatial analysis.
    
    Replaces zone-based approach with threat level progression analysis.
    
    Args:
        df: Pass-level baseline data
        team_clusters: Team cluster assignments
        output_dir: Output directory for CSVs and plots
        
    Returns:
        Tuple of (threat_df, position_stats, cluster_stats)
    """
    # Compute xT for all passes
    df = compute_xt_for_passes(df)
    
    # Analysis 1: Threat distribution
    threat_df = analyze_xt_threat_distribution(df)
    threat_df.to_csv(output_dir / 'csv' / 'cxa_xt_threat_distribution.csv', index=False)
    
    # Analysis 2: By position
    position_df = analyze_xt_by_position(df)
    position_df.to_csv(output_dir / 'csv' / 'cxa_xt_by_position.csv', index=False)
    position_stats = position_df.to_dict('records')
    
    # Analysis 3: By cluster
    cluster_analyses = analyze_xt_by_team_cluster(df, team_clusters)
    cluster_stats = cluster_analyses
    
    # Save cluster stats
    cluster_summary = []
    for cluster_id, stats in cluster_analyses.items():
        stats['cluster_id'] = cluster_id
        cluster_summary.append(stats)
    
    pd.DataFrame(cluster_summary).to_csv(
        output_dir / 'csv' / 'cxa_xt_by_cluster.csv', 
        index=False
    )
    
    # Visualizations
    team_labels = dict(zip(
        team_clusters['cluster'],
        team_clusters['cluster_label']
    ))
    
    plot_xt_threat_heatmap(threat_df, output_dir / 'plots' / 'cxa_xt_threat_heatmap.png')
    plot_xt_gain_by_position(position_df, output_dir / 'plots' / 'cxa_xt_by_position.png')
    plot_xt_by_cluster(cluster_analyses, team_labels, output_dir / 'plots' / 'cxa_xt_by_cluster.png')
    
    return threat_df, position_stats, cluster_stats
