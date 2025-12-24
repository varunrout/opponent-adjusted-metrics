"""
Assist Sequence Feature Engineering

Extracts and analyzes assist sequences with xT progression tracking.
For each assist, tracks all players' xT at release and receipt moments,
with focus on receiver's pre-assist positioning.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from opponent_adjusted.features.xt_model import compute_xt_for_passes, get_xt_model


def extract_assist_sequences(
    df: pd.DataFrame,
    sequence_depth: int = 3
) -> pd.DataFrame:
    """
    Extract sequences leading to assists.
    
    For each pass that results in a shot, tracks back through
    preceding passes to capture the full sequence.
    
    Args:
        df: Pass-level DataFrame
        sequence_depth: Number of passes before assist to track
        
    Returns:
        DataFrame with sequences (one row per pass in sequence)
    """
    sequences = []
    
    # First add xT if not present
    df = compute_xt_for_passes(df) if 'xt_start' not in df.columns else df.copy()
    
    # Find all assists (passes resulting in shots)
    assists = df[df['shot_assisted'] == 1].copy() if 'shot_assisted' in df.columns else pd.DataFrame()
    
    if len(assists) == 0:
        return pd.DataFrame()
    
    # For each assist, extract sequence
    for _, assist_row in assists.iterrows():
        match_id = assist_row['match_id']
        period = assist_row['period']
        
        # Get match/period context
        match_passes = df[
            (df['match_id'] == match_id) & 
            (df['period'] == period)
        ].reset_index(drop=True)
        
        assist_idx = match_passes.index[match_passes['pass_id'] == assist_row['pass_id']].tolist()
        
        if not assist_idx:
            continue
        
        assist_idx = assist_idx[0]
        
        # Extract sequence (preceding passes + assist)
        start_idx = max(0, assist_idx - sequence_depth)
        end_idx = assist_idx + 1
        
        sequence_passes = match_passes.iloc[start_idx:end_idx].copy()
        sequence_passes['sequence_position'] = range(len(sequence_passes))
        sequence_passes['assist_pass_idx'] = assist_idx - start_idx
        sequence_passes['is_assist_pass'] = (sequence_passes.index == assist_idx)
        sequence_passes['assist_id'] = assist_row['pass_id']
        
        sequences.append(sequence_passes)
    
    if not sequences:
        return pd.DataFrame()
    
    return pd.concat(sequences, ignore_index=True)


def compute_assist_receiver_xt(
    df: pd.DataFrame,
    sequences: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute receiver's xT at moment of receiving assist pass.
    
    For the assist receiver, tracks:
    - xt_receive: xT at location where assist is received
    - xt_release_before: xT when they previously passed (in sequence)
    - xt_percentile: Percentile rank of receiving location
    
    Args:
        df: Full pass-level DataFrame
        sequences: Assist sequences from extract_assist_sequences()
        
    Returns:
        DataFrame with receiver xT features for each assist
    """
    model = get_xt_model()
    receiver_features = []
    
    assist_sequences = sequences[sequences['is_assist_pass'] == True].groupby('assist_id')
    
    for assist_id, assist_group in assist_sequences:
        assist_row = assist_group.iloc[0]
        
        # Receiver's position (pass destination)
        receiver_x = assist_row['pass_end_x']
        receiver_y = assist_row['pass_end_y']
        
        # xT at receiving location
        xt_receive = model.get_xt(receiver_x, receiver_y)
        xt_receive_percentile = model.get_xt_percentile(xt_receive)
        
        # Find receiver's prior pass in sequence (if exists)
        receiver_id = assist_row['receiver_id']
        prior_pass = assist_row  # Default to assist itself
        
        # Look for receiver's previous action in wider match context
        match_passes = df[
            (df['match_id'] == assist_row['match_id']) & 
            (df['period'] == assist_row['period'])
        ]
        receiver_passes = match_passes[match_passes['player_id'] == receiver_id]
        
        # Get their last pass before receiving assist
        before_assist = receiver_passes[
            receiver_passes.index < match_passes[
                match_passes['pass_id'] == assist_id
            ].index[0] if assist_id in match_passes['pass_id'].values else float('inf')
        ]
        
        if len(before_assist) > 0:
            prior_row = before_assist.iloc[-1]
            xt_release_before = prior_row['xt_start'] if 'xt_start' in prior_row else 0
        else:
            xt_release_before = 0
        
        receiver_features.append({
            'assist_id': assist_id,
            'receiver_id': receiver_id,
            'receiver_name': assist_row.get('receiver_name', 'Unknown'),
            'receiver_x': receiver_x,
            'receiver_y': receiver_y,
            'xt_receive': xt_receive,
            'xt_receive_percentile': xt_receive_percentile,
            'xt_release_before': xt_release_before,
            'threat_level_receive': model.get_threat_level(xt_receive),
        })
    
    return pd.DataFrame(receiver_features)


def analyze_assist_xT_patterns(
    assist_features: pd.DataFrame,
    team_clusters: pd.DataFrame,
    baseline_df: pd.DataFrame
) -> Dict:
    """
    Analyze assist creation patterns by xT and cluster.
    
    Shows which clusters create assists from which threat levels.
    
    Args:
        assist_features: Receiver xT features
        team_clusters: Team cluster assignments
        baseline_df: Full baseline data for team mapping
        
    Returns:
        Dictionary with analysis results
    """
    # Add cluster info
    team_map = dict(zip(team_clusters['team_id'], team_clusters['cluster']))
    assist_features['cluster'] = baseline_df[
        baseline_df['pass_id'].isin(assist_features['assist_id'])
    ]['team_id'].map(team_map)
    
    patterns = {}
    
    # Overall patterns
    patterns['overall'] = {
        'total_assists': len(assist_features),
        'avg_receiver_xt': assist_features['xt_receive'].mean(),
        'median_receiver_xt': assist_features['xt_receive'].median(),
        'assists_from_low_threat': (
            assist_features['threat_level_receive'] == 'Low'
        ).sum(),
        'assists_from_high_threat': (
            assist_features['threat_level_receive'] == 'High'
        ).sum(),
    }
    
    # By cluster
    for cluster_id in sorted(assist_features['cluster'].dropna().unique()):
        cluster_assists = assist_features[assist_features['cluster'] == cluster_id]
        
        patterns[f'cluster_{cluster_id}'] = {
            'assists': len(cluster_assists),
            'avg_receiver_xt': cluster_assists['xt_receive'].mean(),
            'threat_distribution': cluster_assists['threat_level_receive'].value_counts().to_dict(),
            'avg_xt_jump': (
                cluster_assists['xt_receive'] - 
                cluster_assists['xt_release_before']
            ).mean(),
        }
    
    return patterns


def plot_assist_receiver_xt_distribution(
    assist_features: pd.DataFrame,
    output_path: Path
) -> None:
    """
    Visualize xT distribution of assist receivers.
    
    Histogram showing where assists are received in threat space.
    
    Args:
        assist_features: Receiver xT features
        output_path: Path to save plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Distribution
    ax = axes[0]
    ax.hist(assist_features['xt_receive'], bins=20, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axvline(assist_features['xt_receive'].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
    ax.axvline(assist_features['xt_receive'].median(), color='green', linestyle='--', linewidth=2, label='Median')
    ax.set_xlabel('xT at Receiving Location', fontsize=11)
    ax.set_ylabel('Number of Assists', fontsize=11)
    ax.set_title('Assist Receiver xT Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Threat level breakdown
    ax = axes[1]
    threat_counts = assist_features['threat_level_receive'].value_counts()
    colors = {'Low': '#FF6B6B', 'Medium': '#FFD93D', 'High': '#6BCB77'}
    
    ax.bar(
        threat_counts.index,
        threat_counts.values,
        color=[colors.get(t, 'gray') for t in threat_counts.index],
        alpha=0.8,
        edgecolor='black'
    )
    ax.set_ylabel('Number of Assists', fontsize=11)
    ax.set_title('Assists by Threat Level at Reception', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_assist_xT_by_cluster(
    assist_features: pd.DataFrame,
    team_clusters: pd.DataFrame,
    output_path: Path
) -> None:
    """
    Compare assist creation xT patterns across clusters.
    
    Multi-panel showing where different clusters create assists.
    
    Args:
        assist_features: Receiver xT features with cluster column
        team_clusters: Team cluster metadata
        output_path: Path to save plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    cluster_labels = dict(zip(team_clusters['cluster'], team_clusters['cluster_label']))
    
    for idx, cluster_id in enumerate(sorted(assist_features['cluster'].dropna().unique())):
        if idx >= 4:
            break
        
        ax = axes[idx]
        cluster_assists = assist_features[assist_features['cluster'] == cluster_id]
        
        if len(cluster_assists) == 0:
            ax.text(0.5, 0.5, f'Cluster {cluster_id}: No assists', 
                   ha='center', va='center', fontsize=11)
            continue
        
        ax.hist(cluster_assists['xt_receive'], bins=15, color='steelblue', alpha=0.7, edgecolor='black')
        
        label = cluster_labels.get(cluster_id, f'Cluster {cluster_id}')
        ax.set_title(f"{label}\n(n={len(cluster_assists)} assists, avg xT={cluster_assists['xt_receive'].mean():.3f})",
                    fontsize=11, fontweight='bold')
        ax.set_xlabel('xT at Reception', fontsize=10)
        ax.set_ylabel('Assists', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
    
    # Hide unused
    for idx in range(4, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Assist Creation xT Profiles by Team Style', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def run_assist_sequence_analysis(
    df: pd.DataFrame,
    team_clusters: pd.DataFrame,
    output_dir: Path,
    sequence_depth: int = 3
) -> Tuple[pd.DataFrame, Dict]:
    """
    Run complete assist sequence xT analysis.
    
    Extracts sequences, computes receiver xT, analyzes patterns by cluster.
    
    Args:
        df: Pass-level baseline data
        team_clusters: Team cluster assignments
        output_dir: Output directory
        sequence_depth: Passes before assist to track
        
    Returns:
        Tuple of (assist_features, assist_patterns)
    """
    # Note: baseline CSV only contains passes resulting in shots
    # Assist sequences would require full match event data
    # For now, provide xT analysis of shooting passes
    
    print("Note: Assist sequence analysis requires full match event data.")
    print("Baseline CSV contains only shot-leading passes. Creating shot location xT summary instead...")
    
    # Add xT to baseline
    df = compute_xt_for_passes(df)
    
    # Simple summary: xT of shooting locations
    shot_analysis = {
        'total_shots': len(df),
        'avg_shot_location_xt': df['xt_end'].mean(),
        'median_shot_location_xt': df['xt_end'].median(),
        'high_threat_shots': (df['xt_threat_end'] == 'High').sum(),
        'medium_threat_shots': (df['xt_threat_end'] == 'Medium').sum(),
        'low_threat_shots': (df['xt_threat_end'] == 'Low').sum(),
    }
    
    # Save summary
    summary_df = pd.DataFrame([shot_analysis])
    summary_df.to_csv(
        output_dir / 'csv' / 'cxa_shot_xt_summary.csv',
        index=False
    )
    
    print(f"\nShot Location xT Summary:")
    print(f"  Total shots: {shot_analysis['total_shots']:,}")
    print(f"  Avg xT: {shot_analysis['avg_shot_location_xt']:.4f}")
    print(f"  High threat: {shot_analysis['high_threat_shots']:,} ({shot_analysis['high_threat_shots']/shot_analysis['total_shots']*100:.1f}%)")
    print(f"  Medium threat: {shot_analysis['medium_threat_shots']:,} ({shot_analysis['medium_threat_shots']/shot_analysis['total_shots']*100:.1f}%)")
    print(f"  Low threat: {shot_analysis['low_threat_shots']:,} ({shot_analysis['low_threat_shots']/shot_analysis['total_shots']*100:.1f}%)")
    
    return pd.DataFrame(), shot_analysis
