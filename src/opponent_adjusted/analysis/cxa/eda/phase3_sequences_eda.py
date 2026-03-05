"""Phase 3 EDA: Pass Sequences Analysis.

Comprehensive EDA for pass sequences data (wide format):
- Sequence length distributions
- Pass chain patterns  
- Goal sequence characteristics
- Pass feature analysis by position
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


def analyze_sequence_lengths(df: pd.DataFrame, output_dir: Path):
    """Analyze sequence length distributions using num_passes_in_sequence column."""
    
    seq_lengths = df['num_passes_in_sequence'].dropna()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Overall distribution
    axes[0, 0].hist(seq_lengths, bins=range(1, 12), edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(seq_lengths.mean(), color='red', linestyle='--',
                       label=f"Mean: {seq_lengths.mean():.2f}")
    axes[0, 0].set_title('Passes per Sequence')
    axes[0, 0].set_xlabel('Number of Passes')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].legend()
    
    # 2. Cumulative
    seq_sorted = np.sort(seq_lengths)
    cumsum = np.arange(1, len(seq_sorted) + 1) / len(seq_sorted)
    axes[0, 1].plot(seq_sorted, cumsum)
    axes[0, 1].axhline(0.5, color='red', linestyle='--', alpha=0.5)
    axes[0, 1].axhline(0.9, color='red', linestyle='--', alpha=0.5)
    axes[0, 1].set_title('Cumulative Distribution')
    axes[0, 1].set_xlabel('Passes per Sequence')
    axes[0, 1].set_ylabel('Cumulative %')
    
    # 3. Percentile breakdown
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    pct_values = np.percentile(seq_lengths, percentiles)
    axes[1, 0].bar(range(len(percentiles)), pct_values, color='steelblue')
    axes[1, 0].set_xticks(range(len(percentiles)))
    axes[1, 0].set_xticklabels([f'P{p}' for p in percentiles])
    axes[1, 0].set_title('Sequence Length Percentiles')
    axes[1, 0].set_ylabel('Number of Passes')
    for i, v in enumerate(pct_values):
        axes[1, 0].text(i, v + 0.2, f'{v:.0f}', ha='center', fontsize=9)
    
    # 4. Value counts
    value_counts = seq_lengths.value_counts().head(10).sort_index()
    axes[1, 1].bar(value_counts.index, value_counts.values, color='coral')
    axes[1, 1].set_title('Sequence Length Value Counts')
    axes[1, 1].set_xlabel('Number of Passes')
    axes[1, 1].set_ylabel('Number of Sequences')
    
    plt.tight_layout()
    plt.savefig(output_dir / "sequence_length_distribution.png", dpi=150)
    plt.close()
    
    # Save statistics
    stats_df = pd.DataFrame({
        'statistic': ['count', 'mean', 'std', 'min', 'p25', 'p50', 'p75', 'max'],
        'value': [
            len(seq_lengths),
            seq_lengths.mean(),
            seq_lengths.std(),
            seq_lengths.min(),
            seq_lengths.quantile(0.25),
            seq_lengths.median(),
            seq_lengths.quantile(0.75),
            seq_lengths.max(),
        ]
    })
    stats_df.to_csv(output_dir / "sequence_length_stats.csv", index=False)
    
    return seq_lengths


def analyze_sequence_by_outcome(df: pd.DataFrame, shots: pd.DataFrame, output_dir: Path):
    """Compare sequences leading to goals vs non-goals."""
    
    # Merge outcome
    merged = df[['shot_id', 'num_passes_in_sequence']].merge(
        shots[['shot_id', 'is_goal']], 
        on='shot_id', 
        how='left'
    )
    
    goals = merged[merged['is_goal'] == 1]['num_passes_in_sequence']
    non_goals = merged[merged['is_goal'] == 0]['num_passes_in_sequence']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Side-by-side histograms
    bins = range(1, 12)
    axes[0].hist(non_goals, bins=bins, alpha=0.6, 
                 label=f'Non-goals (n={len(non_goals):,})', density=True)
    axes[0].hist(goals, bins=bins, alpha=0.6,
                 label=f'Goals (n={len(goals):,})', density=True)
    axes[0].set_title('Sequence Length by Outcome')
    axes[0].set_xlabel('Number of Passes')
    axes[0].set_ylabel('Density')
    axes[0].legend()
    
    # 2. Box plot
    axes[1].boxplot([non_goals.dropna(), goals.dropna()], 
                    labels=['Non-goal', 'Goal'])
    axes[1].set_title('Sequence Length Comparison')
    axes[1].set_ylabel('Number of Passes')
    
    plt.tight_layout()
    plt.savefig(output_dir / "sequence_by_outcome.png", dpi=150)
    plt.close()
    
    # Statistical test
    if len(goals) > 1 and len(non_goals) > 1:
        t_stat, p_value = stats.ttest_ind(goals.dropna(), non_goals.dropna())
    else:
        t_stat, p_value = np.nan, np.nan
    
    comparison = pd.DataFrame({
        'outcome': ['Non-goal', 'Goal', 'T-test'],
        'n': [len(non_goals), len(goals), ''],
        'mean': [non_goals.mean(), goals.mean(), ''],
        'std': [non_goals.std(), goals.std(), ''],
        't_stat': ['', '', t_stat],
        'p_value': ['', '', p_value],
    })
    comparison.to_csv(output_dir / "sequence_outcome_comparison.csv", index=False)
    
    logger.info(f"\nGoal sequences: mean={goals.mean():.2f}")
    logger.info(f"Non-goal sequences: mean={non_goals.mean():.2f}")
    logger.info(f"T-test p-value: {p_value:.4e}" if not np.isnan(p_value) else "T-test: N/A")
    
    return comparison


def analyze_pass_by_position(df: pd.DataFrame, output_dir: Path):
    """Analyze pass features by position (pass1, pass2, etc.)."""
    
    # Find which pass columns exist
    max_passes = 0
    for col in df.columns:
        if col.startswith('pass') and '_id' in col:
            try:
                num = int(col.replace('pass', '').replace('_id', ''))
                max_passes = max(max_passes, num)
            except ValueError:
                pass
    
    if max_passes == 0:
        logger.warning("No pass position columns found")
        return None
    
    logger.info(f"Found pass columns up to pass{max_passes}")
    
    # Aggregate features by position
    position_stats = []
    for pos in range(1, max_passes + 1):
        prefix = f'pass{pos}_'
        
        # Count how many sequences have this position
        id_col = f'{prefix}id'
        if id_col in df.columns:
            n_with_pass = df[id_col].notna().sum()
        else:
            n_with_pass = 0
        
        # Get key features
        stats_row = {
            'position': pos,
            'n_sequences_with_pass': n_with_pass,
            'pct_sequences': 100 * n_with_pass / len(df) if len(df) > 0 else 0,
        }
        
        # Add feature means if columns exist
        for feat in ['xt_delta', 'end_x', 'end_y', 'is_progressive', 'is_cross', 'is_through_ball']:
            col = f'{prefix}{feat}'
            if col in df.columns:
                stats_row[f'mean_{feat}'] = df[col].mean()
        
        position_stats.append(stats_row)
    
    position_df = pd.DataFrame(position_stats)
    position_df.to_csv(output_dir / "pass_by_position_stats.csv", index=False)
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Volume by position
    axes[0, 0].bar(position_df['position'], position_df['n_sequences_with_pass'], color='steelblue')
    axes[0, 0].set_title('Sequences with Pass at Each Position')
    axes[0, 0].set_xlabel('Pass Position (1=last before shot)')
    axes[0, 0].set_ylabel('Count')
    
    # 2. Mean xT delta by position
    if 'mean_xt_delta' in position_df.columns:
        axes[0, 1].bar(position_df['position'], position_df['mean_xt_delta'], color='coral')
        axes[0, 1].axhline(0, color='black', linestyle='--')
        axes[0, 1].set_title('Mean xT Delta by Position')
        axes[0, 1].set_xlabel('Pass Position')
        axes[0, 1].set_ylabel('Mean xT Delta')
    
    # 3. Mean end_x by position
    if 'mean_end_x' in position_df.columns:
        axes[1, 0].plot(position_df['position'], position_df['mean_end_x'], marker='o', linewidth=2)
        axes[1, 0].set_title('Mean End X by Position')
        axes[1, 0].set_xlabel('Pass Position')
        axes[1, 0].set_ylabel('Mean End X')
    
    # 4. Is progressive rate by position
    if 'mean_is_progressive' in position_df.columns:
        axes[1, 1].bar(position_df['position'], position_df['mean_is_progressive'] * 100, color='green')
        axes[1, 1].set_title('Progressive Pass Rate by Position')
        axes[1, 1].set_xlabel('Pass Position')
        axes[1, 1].set_ylabel('Progressive Rate (%)')
    
    plt.tight_layout()
    plt.savefig(output_dir / "pass_features_by_position.png", dpi=150)
    plt.close()
    
    return position_df


def analyze_sequence_types(df: pd.DataFrame, output_dir: Path):
    """Analyze boolean features of sequences."""
    
    bool_cols = ['num_crosses', 'num_through_ball', 'num_progressive']
    bool_cols = [c for c in bool_cols if c in df.columns]
    
    if len(bool_cols) == 0:
        logger.warning("No sequence type columns found")
        return None
    
    type_stats = []
    for col in bool_cols:
        series = df[col].dropna()
        type_stats.append({
            'feature': col,
            'mean': series.mean(),
            'sum': series.sum(),
            'has_at_least_one': (series > 0).sum(),
            'pct_with_feature': 100 * (series > 0).sum() / len(series),
        })
    
    type_df = pd.DataFrame(type_stats)
    type_df.to_csv(output_dir / "sequence_type_stats.csv", index=False)
    
    logger.info("\nSequence Type Statistics:")
    print(type_df.to_string(index=False))
    
    return type_df


def analyze_temporal_features(df: pd.DataFrame, output_dir: Path):
    """Analyze temporal features of sequences."""
    
    temporal_cols = ['sequence_duration_seconds', 'sequence_start_minute', 'sequence_start_second']
    temporal_cols = [c for c in temporal_cols if c in df.columns]
    
    if len(temporal_cols) == 0:
        return None
    
    fig, axes = plt.subplots(1, len(temporal_cols), figsize=(5 * len(temporal_cols), 5))
    if len(temporal_cols) == 1:
        axes = [axes]
    
    for i, col in enumerate(temporal_cols):
        series = df[col].dropna()
        axes[i].hist(series, bins=30, edgecolor='black', alpha=0.7)
        axes[i].axvline(series.mean(), color='red', linestyle='--', label=f'Mean: {series.mean():.1f}')
        axes[i].set_title(col.replace('_', ' ').title())
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Count')
        axes[i].legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "temporal_features.png", dpi=150)
    plt.close()
    
    # Stats
    temporal_stats = []
    for col in temporal_cols:
        series = df[col].dropna()
        temporal_stats.append({
            'feature': col,
            'mean': series.mean(),
            'median': series.median(),
            'std': series.std(),
            'min': series.min(),
            'max': series.max(),
        })
    
    temporal_df = pd.DataFrame(temporal_stats)
    temporal_df.to_csv(output_dir / "temporal_stats.csv", index=False)
    
    return temporal_df


def run_phase3_eda(output_dir: Path = None) -> Dict[str, Any]:
    """Run Phase 3 EDA: Pass Sequences Analysis."""
    
    repo_root = _get_repo_root()
    if output_dir is None:
        output_dir = repo_root / "outputs" / "analysis" / "cxa" / "eda" / "phase3_sequences"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("PHASE 3 EDA: Pass Sequences Analysis")
    logger.info("=" * 60)
    
    # Load data
    sequences = pd.read_parquet(repo_root / "feature_store" / "cxa" / "sequences.parquet")
    shots = pd.read_parquet(repo_root / "feature_store" / "cxa" / "shots.parquet")
    
    logger.info(f"Loaded {len(sequences):,} sequences (wide format)")
    logger.info(f"Columns: {list(sequences.columns[:10])}...")
    
    results = {}
    
    # 1. Sequence Length Analysis
    logger.info("\n--- 1. Sequence Length Analysis ---")
    seq_lengths = analyze_sequence_lengths(sequences, output_dir)
    results["seq_lengths"] = seq_lengths
    
    logger.info(f"Sequences analyzed: {len(seq_lengths):,}")
    logger.info(f"Mean passes per sequence: {seq_lengths.mean():.2f}")
    logger.info(f"Median passes per sequence: {seq_lengths.median():.0f}")
    
    # 2. Goal vs Non-Goal Sequences
    logger.info("\n--- 2. Outcome Comparison ---")
    outcome_comparison = analyze_sequence_by_outcome(sequences, shots, output_dir)
    results["outcome_comparison"] = outcome_comparison
    
    # 3. Pass Features by Position
    logger.info("\n--- 3. Pass Features by Position ---")
    position_stats = analyze_pass_by_position(sequences, output_dir)
    results["position_stats"] = position_stats
    
    # 4. Sequence Types
    logger.info("\n--- 4. Sequence Types ---")
    type_stats = analyze_sequence_types(sequences, output_dir)
    results["type_stats"] = type_stats
    
    # 5. Temporal Features
    logger.info("\n--- 5. Temporal Features ---")
    temporal_stats = analyze_temporal_features(sequences, output_dir)
    results["temporal_stats"] = temporal_stats
    
    # 6. Summary
    logger.info("\n--- 6. Summary ---")
    summary = {
        "total_sequences": len(sequences),
        "mean_passes_per_seq": float(seq_lengths.mean()),
        "max_passes_per_seq": int(seq_lengths.max()),
        "sequences_with_1_pass": int((seq_lengths == 1).sum()),
        "sequences_with_2plus_passes": int((seq_lengths >= 2).sum()),
    }
    
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(output_dir / "sequence_summary.csv", index=False)
    
    for k, v in summary.items():
        logger.info(f"  {k}: {v}")
    
    logger.info(f"\nPhase 3 EDA complete. Outputs saved to {output_dir}")
    
    return results


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    run_phase3_eda()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
