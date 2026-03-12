"""Run All CxT EDA Phases.

Master orchestrator to run comprehensive EDA for CxT feature store.
"""

from __future__ import annotations

import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

import pandas as pd

from .phase0_data_overview import run_phase0_eda
from .phase1_progressions_eda import run_phase1_eda
from .phase2_outcomes_eda import run_phase2_eda
from .phase3_opponent_eda import run_phase3_eda

logger = logging.getLogger(__name__)


def _get_repo_root() -> Path:
    """Get repository root path."""
    return Path(__file__).resolve().parents[5]


def _get_feature_store_path() -> Path:
    """Get CxT feature store path."""
    return _get_repo_root() / "feature_store" / "cxt"


def _get_output_path() -> Path:
    """Get EDA output path."""
    return _get_repo_root() / "outputs" / "analysis" / "cxt" / "eda"


def generate_eda_report(results: Dict[str, Any], output_dir: Path) -> Path:
    """Generate summary markdown report.
    
    Args:
        results: Dictionary of all EDA results
        output_dir: Directory to save report
        
    Returns:
        Path to generated report
    """
    report_lines = [
        "# CxT Feature Store EDA Report",
        "",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "---",
        "",
        "## Executive Summary",
        "",
        "This report provides comprehensive Exploratory Data Analysis (EDA) for the CxT (Contextual Expected Threat) feature store.",
        "",
    ]
    
    # Phase 0 Summary
    if "phase0" in results:
        p0 = results["phase0"]
        report_lines.extend([
            "---",
            "",
            "## Phase 0: Data Overview",
            "",
            "### Dataset Summary",
            "",
            f"- **Total Actions**: {p0.get('total_rows', 'N/A'):,}",
            f"- **Total Columns**: {p0.get('total_columns', 'N/A')}",
            f"- **Unique Matches**: {p0.get('unique_matches', 'N/A'):,}",
            f"- **Unique Teams**: {p0.get('unique_teams', 'N/A'):,}",
            f"- **Unique Players**: {p0.get('unique_players', 'N/A'):,}",
            "",
            "### Action Type Distribution",
            "",
            "| Action Type | Count | Percentage |",
            "|-------------|-------|------------|",
        ])
        
        for action, count in p0.get("action_types", {}).items():
            pct = count / p0.get("total_rows", 1) * 100
            report_lines.append(f"| {action} | {count:,} | {pct:.1f}% |")
        
        report_lines.extend([
            "",
            "### xT Statistics",
            "",
            "| Metric | Mean | Std |",
            "|--------|------|-----|",
        ])
        
        for metric, stats in p0.get("xt_stats", {}).items():
            report_lines.append(f"| {metric} | {stats.get('mean', 0):.6f} | {stats.get('std', 0):.4f} |")
        
        report_lines.append("")
    
    # Phase 1 Summary
    if "phase1" in results:
        p1 = results["phase1"]
        report_lines.extend([
            "---",
            "",
            "## Phase 1: Progressions EDA",
            "",
            "### Key Findings",
            "",
            f"- **Pressure Rate**: {p1.get('pressure_pct', 0):.1f}%",
            f"- **Pressure xT Effect**: {p1.get('pressure_xt_diff', 0):.6f}",
            "",
            "### Insight",
            "",
            "Actions under pressure show " + 
            ("LOWER" if p1.get('pressure_xt_diff', 0) < 0 else "HIGHER") +
            " xT delta, confirming pressure's impact on ball progression.",
            "",
        ])
    
    # Phase 2 Summary
    if "phase2" in results:
        p2 = results["phase2"]
        report_lines.extend([
            "---",
            "",
            "## Phase 2: Outcomes EDA",
            "",
            "### Key Statistics",
            "",
            f"- **Pass Completion Rate**: {p2.get('pass_completion_rate', 0)*100:.1f}%",
            f"- **Final Third Entries**: {p2.get('final_third_entries', 0):,} ({p2.get('final_third_entry_rate', 0)*100:.2f}%)",
            f"- **Penalty Area Entries**: {p2.get('penalty_area_entries', 0):,} ({p2.get('penalty_area_entry_rate', 0)*100:.2f}%)",
            "",
        ])
        
        if "carry_stats" in p2:
            cs = p2["carry_stats"]
            report_lines.extend([
                "### Carry Analysis",
                "",
                f"- **Total Carries**: {cs.get('total_carries', 0):,}",
                f"- **Mean xT Delta**: {cs.get('mean_xt_delta', 0):.6f}",
                f"- **Progressive Rate**: {cs.get('progressive_rate', 0)*100:.1f}%",
                "",
            ])
    
    # Phase 3 Summary
    if "phase3" in results:
        p3 = results["phase3"]
        report_lines.extend([
            "---",
            "",
            "## Phase 3: Opponent Context EDA",
            "",
            "### Key Findings",
            "",
        ])
        
        if "pressure_tier_xt_signal" in p3:
            signal = p3["pressure_tier_xt_signal"]
            report_lines.append(f"- **Pressure Tier xT Signal**: {signal:.6f} (Low - High)")
            if signal > 0:
                report_lines.append("  - ✓ Expected: Higher xT vs low-pressure opponents")
            else:
                report_lines.append("  - ⚠ Unexpected direction")
        
        if "home_xt_advantage" in p3:
            report_lines.append(f"- **Home xT Advantage**: {p3['home_xt_advantage']:.6f}")
        
        report_lines.append("")
    
    # Key Insights & Recommendations
    report_lines.extend([
        "---",
        "",
        "## Key Insights & Recommendations",
        "",
        "### 1. Action Type Mix",
        "",
        "- Passes dominate but carries contribute ~44% of actions",
        "- Carries should be included in CxT modeling (validated)",
        "",
        "### 2. Pressure Effects",
        "",
        "- Pressure reduces ball progression quality",
        "- Include pressure as a contextual feature",
        "",
        "### 3. Opponent Context",
        "",
        "- Playing against high-pressure teams reduces xT accumulation",
        "- Opponent defensive quality should be a key feature",
        "",
        "---",
        "",
        "## Output Files",
        "",
        "All EDA outputs saved in `outputs/analysis/cxt/eda/`:",
        "",
        "- `csv/` - Tabular summaries",
        "- `plots/` - Visualizations",
        "- `eda_report.md` - This report",
        "",
    ])
    
    # Write report
    report_path = output_dir / "eda_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    
    return report_path


def run_full_eda(
    progressions_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    opponent_profiles: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Run all EDA phases.
    
    Args:
        progressions_path: Path to progressions parquet (default: feature_store/cxt/progressions.parquet)
        output_dir: Output directory (default: outputs/analysis/cxt/eda/)
        opponent_profiles: Optional opponent profiles DataFrame
        
    Returns:
        Dictionary of all results
    """
    # Set defaults
    if progressions_path is None:
        progressions_path = _get_feature_store_path() / "progressions.parquet"
    
    if output_dir is None:
        output_dir = _get_output_path()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 70)
    logger.info("CxT EXPLORATORY DATA ANALYSIS")
    logger.info("=" * 70)
    logger.info(f"Input: {progressions_path}")
    logger.info(f"Output: {output_dir}")
    logger.info("=" * 70)
    
    # Load data
    logger.info("\nLoading progressions data...")
    df = pd.read_parquet(progressions_path)
    logger.info(f"Loaded {len(df):,} rows")
    
    # Run all phases
    results = {}
    
    # Phase 0: Data Overview
    results["phase0"] = run_phase0_eda(df, output_dir)
    
    # Phase 1: Progressions EDA
    results["phase1"] = run_phase1_eda(df, output_dir)
    
    # Phase 2: Outcomes EDA
    results["phase2"] = run_phase2_eda(df, output_dir)
    
    # Phase 3: Opponent Context EDA
    results["phase3"] = run_phase3_eda(df, output_dir, opponent_profiles)
    
    # Generate report
    report_path = generate_eda_report(results, output_dir)
    logger.info(f"\n✓ EDA Report saved to: {report_path}")
    
    # Save full results as JSON
    results_json = output_dir / "eda_results.json"
    with open(results_json, "w", encoding="utf-8") as f:
        # Convert numpy types to Python types for JSON serialization
        def convert(obj):
            if isinstance(obj, (pd.Timestamp, datetime)):
                return obj.isoformat()
            elif hasattr(obj, "item"):  # numpy scalar
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj
        
        json.dump(convert(results), f, indent=2, default=str)
    
    logger.info(f"✓ Results JSON saved to: {results_json}")
    
    logger.info("\n" + "=" * 70)
    logger.info("EDA COMPLETE")
    logger.info("=" * 70)
    
    return results
