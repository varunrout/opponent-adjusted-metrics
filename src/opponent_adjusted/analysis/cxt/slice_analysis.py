"""
CxT Pre-Model Slice Analysis

Validates feature signals through quartile lifts on outcome (success rate).
Analyzes signal strength across key dimensions before modeling.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def compute_lift_table(
    df: pd.DataFrame,
    feature_col: str,
    outcome_col: str = "success",
    bins: int = 4,
    labels: list[str] | None = None,
) -> pd.DataFrame:
    """
    Compute lift table for a numeric feature.
    
    Bins the feature into quantiles and computes:
    - Mean outcome per bin
    - Lift vs overall mean
    - Count per bin
    """
    overall_mean = df[outcome_col].mean()
    
    # Create bins
    try:
        df_temp = df.copy()
        df_temp["_bin"] = pd.qcut(df_temp[feature_col], q=bins, duplicates="drop")
        
        if labels and len(labels) == len(df_temp["_bin"].cat.categories):
            df_temp["_bin"] = df_temp["_bin"].cat.rename_categories(labels)
    except ValueError:
        # If quantile fails, use fixed bins
        df_temp = df.copy()
        df_temp["_bin"] = pd.cut(df_temp[feature_col], bins=bins, duplicates="drop")
    
    # Aggregate
    agg = df_temp.groupby("_bin", observed=True).agg(
        count=(outcome_col, "count"),
        mean_outcome=(outcome_col, "mean"),
    ).reset_index()
    
    agg["lift"] = agg["mean_outcome"] / overall_mean
    agg["feature"] = feature_col
    agg = agg.rename(columns={"_bin": "bin"})
    
    return agg[["feature", "bin", "count", "mean_outcome", "lift"]]


def compute_categorical_lift(
    df: pd.DataFrame,
    feature_col: str,
    outcome_col: str = "success",
) -> pd.DataFrame:
    """Compute lift table for a categorical feature."""
    overall_mean = df[outcome_col].mean()
    
    agg = df.groupby(feature_col, observed=True).agg(
        count=(outcome_col, "count"),
        mean_outcome=(outcome_col, "mean"),
    ).reset_index()
    
    agg["lift"] = agg["mean_outcome"] / overall_mean
    agg["feature"] = feature_col
    agg = agg.rename(columns={feature_col: "bin"})
    
    return agg[["feature", "bin", "count", "mean_outcome", "lift"]]


def analyze_opponent_slices(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze signal by opponent strength slices."""
    logger.info("Analyzing opponent strength slices...")
    
    # Use opponent_global_rating
    if "opponent_global_rating" not in df.columns:
        logger.warning("No opponent_global_rating column found")
        return pd.DataFrame()
    
    labels = ["Weak", "Medium-Weak", "Medium-Strong", "Strong"]
    return compute_lift_table(df, "opponent_global_rating", bins=4, labels=labels)


def analyze_action_type_slices(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze signal by action type."""
    logger.info("Analyzing action type slices...")
    return compute_categorical_lift(df, "action_type")


def analyze_zone_slices(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze signal by zone dimensions."""
    logger.info("Analyzing zone slices...")
    
    results = []
    
    # Third
    if "third" in df.columns:
        results.append(compute_categorical_lift(df, "third"))
    
    # is_central
    if "is_central" in df.columns:
        results.append(compute_categorical_lift(df, "is_central"))
    
    # Macro zone (1-9 grid)
    if "macro_zone_start" in df.columns:
        results.append(compute_categorical_lift(df, "macro_zone_start"))
    
    # Combined zone (DEF_CENTRAL, MID_WIDE_R, etc.)
    if "start_zone_name" in df.columns:
        results.append(compute_categorical_lift(df, "start_zone_name"))
    elif "start_zone" in df.columns:
        results.append(compute_categorical_lift(df, "start_zone"))
    
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


def analyze_pressure_slices(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze signal by pressure context."""
    logger.info("Analyzing pressure slices...")
    
    results = []
    
    if "under_pressure" in df.columns:
        results.append(compute_categorical_lift(df, "under_pressure"))
    
    if "pressure_adjusted_xt" in df.columns:
        labels = ["Low", "Medium-Low", "Medium-High", "High"]
        results.append(compute_lift_table(df, "pressure_adjusted_xt", bins=4, labels=labels))
    
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


def analyze_game_state_slices(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze signal by game state."""
    logger.info("Analyzing game state slices...")
    
    results = []
    
    # Minute normalized quartiles
    if "minute_normalized" in df.columns:
        labels = ["Early", "Mid-Early", "Mid-Late", "Late"]
        results.append(compute_lift_table(df, "minute_normalized", bins=4, labels=labels))
    
    # Period
    if "period" in df.columns:
        results.append(compute_categorical_lift(df, "period"))
    
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


def compute_signal_strength(lift_df: pd.DataFrame) -> dict[str, float]:
    """
    Compute signal strength metrics for a lift table.
    
    Returns:
        - lift_range: max_lift - min_lift
        - lift_ratio: max_lift / min_lift
        - weighted_spread: weighted std of lifts
    """
    if lift_df.empty:
        return {"lift_range": 0, "lift_ratio": 1, "weighted_spread": 0}
    
    lifts = lift_df["lift"].values
    counts = lift_df["count"].values
    
    return {
        "lift_range": float(lifts.max() - lifts.min()),
        "lift_ratio": float(lifts.max() / lifts.min()) if lifts.min() > 0 else float("inf"),
        "weighted_spread": float(np.sqrt(np.average((lifts - lifts.mean())**2, weights=counts))),
    }


def run_slice_analysis(
    df: pd.DataFrame,
    output_dir: Path | str | None = None,
) -> dict[str, Any]:
    """
    Run comprehensive slice analysis.
    
    Args:
        df: Featured dataframe with success outcome
        output_dir: Optional directory to save results
        
    Returns:
        Dictionary with slice results and signal strengths
    """
    logger.info("=" * 60)
    logger.info("CxT PRE-MODEL SLICE ANALYSIS")
    logger.info("=" * 60)
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Ensure success column exists
    if "success" not in df.columns:
        # Create from action_outcome if available
        if "action_outcome" in df.columns:
            df["success"] = (df["action_outcome"] == "Complete").astype(int)
        else:
            raise ValueError("No success outcome column found")
    
    results = {}
    signal_summary = []
    
    # Run each slice analysis
    analyses = [
        ("opponent_strength", analyze_opponent_slices),
        ("action_type", analyze_action_type_slices),
        ("zone", analyze_zone_slices),
        ("pressure", analyze_pressure_slices),
        ("game_state", analyze_game_state_slices),
    ]
    
    all_lifts = []
    
    for name, func in analyses:
        lift_df = func(df)
        if not lift_df.empty:
            results[name] = lift_df
            all_lifts.append(lift_df)
            
            # Compute signal strength per unique feature
            for feature in lift_df["feature"].unique():
                feat_df = lift_df[lift_df["feature"] == feature]
                strength = compute_signal_strength(feat_df)
                signal_summary.append({
                    "slice_group": name,
                    "feature": feature,
                    **strength,
                })
                logger.info(
                    f"  {feature}: range={strength['lift_range']:.3f}, "
                    f"ratio={strength['lift_ratio']:.2f}x"
                )
    
    # Combine all lifts
    if all_lifts:
        all_lifts_df = pd.concat(all_lifts, ignore_index=True)
        signal_summary_df = pd.DataFrame(signal_summary)
        
        results["all_lifts"] = all_lifts_df
        results["signal_summary"] = signal_summary_df
        
        # Save outputs
        if output_dir:
            all_lifts_df.to_csv(output_dir / "slice_lifts.csv", index=False)
            signal_summary_df.to_csv(output_dir / "signal_summary.csv", index=False)
            
            # Generate visualizations
            _plot_slice_analysis(all_lifts_df, signal_summary_df, output_dir)
            
            logger.info(f"Results saved to {output_dir}")
    
    # Summary stats
    logger.info("=" * 60)
    logger.info("SIGNAL STRENGTH SUMMARY")
    logger.info("=" * 60)
    
    if signal_summary:
        signal_summary_df = pd.DataFrame(signal_summary)
        top_signals = signal_summary_df.nlargest(5, "lift_ratio")
        logger.info("Top 5 signals by lift ratio:")
        for _, row in top_signals.iterrows():
            logger.info(f"  {row['feature']}: {row['lift_ratio']:.2f}x")
    
    return results


def _plot_slice_analysis(
    lifts_df: pd.DataFrame,
    signal_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Generate slice analysis visualizations."""
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 1. Signal strength bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    signal_sorted = signal_df.sort_values("lift_ratio", ascending=True)
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(signal_sorted)))
    
    y_pos = np.arange(len(signal_sorted))
    ax.barh(y_pos, signal_sorted["lift_ratio"], color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(signal_sorted["feature"])
    ax.set_xlabel("Lift Ratio (max/min)")
    ax.set_title("Feature Signal Strength: Lift Ratio by Slice")
    ax.axvline(x=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(x=1.05, color="green", linestyle="--", alpha=0.5, label="Min threshold (1.05x)")
    ax.legend(loc="lower right")
    
    plt.tight_layout()
    plt.savefig(output_dir / "signal_strength.png", dpi=150)
    plt.close()
    
    # 2. Key slice lift charts
    key_features = ["action_type", "third", "under_pressure", "opponent_global_rating"]
    available_features = [f for f in key_features if f in lifts_df["feature"].values]
    
    # Fill in with other features if needed
    if len(available_features) < 4:
        other_features = [f for f in lifts_df["feature"].unique() if f not in available_features]
        available_features.extend(other_features[:4 - len(available_features)])
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, feature in enumerate(available_features[:4]):
        feat_df = lifts_df[lifts_df["feature"] == feature]
        if feat_df.empty:
            continue
            
        ax = axes[i]
        x = range(len(feat_df))
        
        # Color by lift value
        colors = ['#d62728' if v < 0.95 else '#2ca02c' if v > 1.05 else '#7f7f7f' 
                  for v in feat_df["lift"]]
        
        bars = ax.bar(x, feat_df["lift"], color=colors, alpha=0.8, edgecolor='black')
        ax.set_xticks(x)
        ax.set_xticklabels(feat_df["bin"].astype(str), rotation=45, ha="right")
        ax.set_ylabel("Lift vs Overall")
        ax.set_title(f"Success Rate Lift: {feature}")
        ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.7, label="Baseline")
        
        # Annotate counts
        for j, (bar, count) in enumerate(zip(bars, feat_df["count"])):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"n={count:,}",
                ha="center",
                fontsize=8,
            )
    
    plt.tight_layout()
    plt.savefig(output_dir / "key_slice_lifts.png", dpi=150)
    plt.close()
    
    # 3. Lift range comparison (new chart)
    fig, ax = plt.subplots(figsize=(12, 6))
    
    signal_sorted_range = signal_df.sort_values("lift_range", ascending=True)
    colors_range = ['#2ca02c' if r > 0.1 else '#ff7f0e' if r > 0.05 else '#d62728' 
                    for r in signal_sorted_range["lift_range"]]
    
    y_pos = np.arange(len(signal_sorted_range))
    bars = ax.barh(y_pos, signal_sorted_range["lift_range"], color=colors_range)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(signal_sorted_range["feature"])
    ax.set_xlabel("Lift Range (max - min)")
    ax.set_title("Feature Discriminative Power: Lift Range")
    
    # Add value labels
    for bar, val in zip(bars, signal_sorted_range["lift_range"]):
        ax.text(val + 0.005, bar.get_y() + bar.get_height()/2, 
               f'{val:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / "lift_range_comparison.png", dpi=150)
    plt.close()
    
    # 4. Success rate by slice group (grouped bar chart)
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Group by slice_group
    groups = signal_df.groupby("slice_group")["lift_ratio"].agg(["mean", "max", "min"]).reset_index()
    
    x = np.arange(len(groups))
    width = 0.25
    
    ax.bar(x - width, groups["min"], width, label="Min Lift", color='#d62728', alpha=0.8)
    ax.bar(x, groups["mean"], width, label="Mean Lift", color='#7f7f7f', alpha=0.8)
    ax.bar(x + width, groups["max"], width, label="Max Lift", color='#2ca02c', alpha=0.8)
    
    ax.set_xticks(x)
    ax.set_xticklabels(groups["slice_group"], rotation=45, ha="right")
    ax.set_ylabel("Lift Ratio")
    ax.set_title("Signal Strength by Analysis Group")
    ax.axhline(y=1.0, color="black", linestyle="--", alpha=0.5)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "slice_group_summary.png", dpi=150)
    plt.close()
    
    # 5. All features dot plot with error bars (showing range)
    fig, ax = plt.subplots(figsize=(10, 8))
    
    y_pos = np.arange(len(signal_df))
    feature_names = signal_df["feature"].values
    
    # Plot min and max as error bars
    lift_ratios = signal_df["lift_ratio"].values
    lift_ranges = signal_df["lift_range"].values
    
    # Create horizontal lines showing the range
    for i, (name, ratio, rng) in enumerate(zip(feature_names, lift_ratios, lift_ranges)):
        ax.plot([ratio - rng/2, ratio + rng/2], [i, i], 'o-', color='steelblue', 
               markersize=8, linewidth=2, alpha=0.7)
    
    ax.scatter(lift_ratios, y_pos, s=100, c='steelblue', zorder=5, edgecolors='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_names)
    ax.set_xlabel("Lift Ratio (with range shown)")
    ax.set_title("Feature Signal: Lift Ratio with Discriminative Range")
    ax.axvline(x=1.0, color="red", linestyle="--", alpha=0.5, label="No signal")
    ax.legend()
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "lift_ratio_dots.png", dpi=150)
    plt.close()
    
    # 6. Distribution of lifts histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    
    all_lifts = lifts_df["lift"].values
    ax.hist(all_lifts, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='No lift (1.0)')
    ax.axvline(x=all_lifts.mean(), color='green', linestyle='--', linewidth=2, 
               label=f'Mean ({all_lifts.mean():.2f})')
    ax.set_xlabel("Lift Value")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Success Rate Lifts Across All Slices")
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "lift_distribution.png", dpi=150)
    plt.close()
    
    # 7. Zone heatmap (enhanced)
    # Use start_zone_name if available, fallback to start_zone
    zone_feature = None
    for feat_name in ["start_zone_name", "start_zone"]:
        if feat_name in lifts_df["feature"].values:
            zone_feature = feat_name
            break
    
    if zone_feature:
        zone_df = lifts_df[lifts_df["feature"] == zone_feature].copy()
        
        # Parse zone codes (e.g., DEF_CENTRAL, MID_WIDE_R)
        zone_df["third"] = zone_df["bin"].str.split("_").str[0]
        zone_df["width"] = zone_df["bin"].str.split("_").str[1:].\
            apply(lambda x: "_".join(x) if isinstance(x, list) else x)
        
        third_order = ["DEF", "MID", "ATT"]
        width_order = ["WIDE_L", "CENTRAL", "WIDE_R"]
        
        try:
            # Use pivot_table with aggfunc to handle any duplicates
            pivot = zone_df.pivot_table(index="third", columns="width", 
                                        values="lift", aggfunc="mean")
            pivot = pivot.reindex(index=third_order, columns=width_order)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto", vmin=0.8, vmax=1.2)
            ax.set_xticks(range(len(width_order)))
            ax.set_xticklabels(width_order)
            ax.set_yticks(range(len(third_order)))
            ax.set_yticklabels(third_order)
            ax.set_title("Success Rate Lift by Start Zone\n(Green = higher success, Red = lower)")
            
            # Annotate
            for i in range(len(third_order)):
                for j in range(len(width_order)):
                    val = pivot.iloc[i, j]
                    if not np.isnan(val):
                        text_color = 'white' if abs(val - 1.0) > 0.1 else 'black'
                        ax.text(j, i, f"{val:.2f}", ha="center", va="center", 
                               fontsize=14, color=text_color, fontweight='bold')
            
            plt.colorbar(im, ax=ax, label="Lift vs Overall")
            plt.tight_layout()
            plt.savefig(output_dir / "zone_lift_heatmap.png", dpi=150)
            plt.close()
        except Exception as e:
            logger.warning(f"Could not generate zone heatmap: {e}")
    
    # 8. Macro zone comparison (new)
    macro_zone_df = lifts_df[lifts_df["feature"].str.contains("macro_zone", case=False)]
    if macro_zone_df.empty:
        # Try to find zone data from different column
        zone_cols = [c for c in lifts_df["feature"].unique() if "zone" in c.lower()]
        if zone_cols:
            macro_zone_df = lifts_df[lifts_df["feature"] == zone_cols[0]]
    
    if not macro_zone_df.empty and len(macro_zone_df) > 1:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        zone_sorted = macro_zone_df.sort_values("lift", ascending=False)
        colors = ['#2ca02c' if v > 1.05 else '#d62728' if v < 0.95 else '#7f7f7f' 
                 for v in zone_sorted["lift"]]
        
        bars = ax.bar(range(len(zone_sorted)), zone_sorted["lift"], color=colors, 
                     edgecolor='black', alpha=0.8)
        ax.set_xticks(range(len(zone_sorted)))
        ax.set_xticklabels([str(x) for x in zone_sorted["bin"]], rotation=45, ha="right")
        ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.7)
        ax.set_xlabel("Zone")
        ax.set_ylabel("Lift vs Overall")
        ax.set_title("Success Rate Lift by Macro Zone")
        
        # Add count annotations
        for bar, count in zip(bars, zone_sorted["count"]):
            ax.text(bar.get_x() + bar.get_width()/2, 0.02, f'n={count:,}', 
                   ha='center', fontsize=8, rotation=90, va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_dir / "macro_zone_lift.png", dpi=150)
        plt.close()
    
    logger.info("Visualizations saved")


def validate_signals(signal_df: pd.DataFrame, min_ratio: float = 1.05) -> tuple[bool, list[str]]:
    """
    Validate that features show meaningful signal.
    
    Args:
        signal_df: Signal summary dataframe
        min_ratio: Minimum lift ratio to consider signal meaningful
        
    Returns:
        (passed, weak_features) - Whether validation passed and list of weak features
    """
    weak_features = signal_df[signal_df["lift_ratio"] < min_ratio]["feature"].tolist()
    passed = len(weak_features) <= len(signal_df) // 2  # At least half should have signal
    
    return passed, weak_features
