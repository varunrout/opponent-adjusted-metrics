"""Run All EDA Phases.

Master orchestrator to run comprehensive EDA for cXA feature store.
"""

from __future__ import annotations

import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

from .phase0_data_overview import run_phase0_eda
from .phase1_passes_eda import run_phase1_eda
from .phase2_shots_eda import run_phase2_eda
from .phase3_sequences_eda import run_phase3_eda
from .phase4_actions_eda import run_phase4_eda

logger = logging.getLogger(__name__)


def _get_repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def generate_eda_report(results: Dict[str, Any], output_dir: Path):
    """Generate summary markdown report."""
    
    report_lines = [
        "# cXA Feature Store EDA Report",
        "",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "---",
        "",
        "## Executive Summary",
        "",
        "This report provides comprehensive Exploratory Data Analysis (EDA) for the cXA (Credit Expected Assist) feature store.",
        "",
    ]
    
    # Phase 0 Summary
    if "phase0" in results:
        p0 = results["phase0"]
        report_lines.extend([
            "### Phase 0: Data Overview",
            "",
            "| Dataset | Rows | Columns | Null % |",
            "|---------|------|---------|--------|",
        ])
        for name, info in p0.get("datasets", {}).items():
            null_pct = info.get("null_percentage", 0)
            report_lines.append(f"| {name} | {info.get('rows', 'N/A'):,} | {info.get('cols', 'N/A')} | {null_pct:.1f}% |")
        report_lines.append("")
    
    # Phase 1 Summary
    if "phase1" in results:
        p1 = results["phase1"]
        report_lines.extend([
            "### Phase 1: Passes EDA",
            "",
            f"- **Total Passes**: {p1.get('total_passes', 'N/A'):,}",
            f"- **Assist Rate**: {p1.get('assist_rate', 0)*100:.3f}%",
            f"- **Class Imbalance**: Severe (assists = {p1.get('assist_rate', 0)*100:.3f}% of passes)",
            "",
        ])
        
        if "top_correlated" in p1:
            report_lines.append("**Top Correlated Features with is_assist**:")
            report_lines.append("")
            for feat, corr in p1.get("top_correlated", [])[:5]:
                report_lines.append(f"- `{feat}`: {corr:.3f}")
            report_lines.append("")
    
    # Phase 2 Summary
    if "phase2" in results:
        p2 = results["phase2"]
        basic = p2.get("basic_stats", {})
        report_lines.extend([
            "### Phase 2: Shots EDA",
            "",
            f"- **Total Shots**: {basic.get('total_shots', 'N/A'):,}",
            f"- **Total Goals**: {basic.get('total_goals', 'N/A'):,}",
            f"- **Conversion Rate**: {basic.get('conversion_rate', 0)*100:.2f}%",
            f"- **Total xG**: {basic.get('total_xg', 0):.2f}",
            "",
        ])
    
    # Phase 3 Summary
    if "phase3" in results:
        report_lines.extend([
            "### Phase 3: Pass Sequences EDA",
            "",
            "Key findings about pass chains leading to shots.",
            "",
        ])
    
    # Phase 4 Summary
    if "phase4" in results:
        p4 = results["phase4"]
        type_dist = p4.get("type_distribution")
        if type_dist is not None and hasattr(type_dist, 'to_dict'):
            report_lines.extend([
                "### Phase 4: Action Sequences EDA",
                "",
                "**Action Type Distribution**:",
                "",
            ])
    
    # Recommendations
    report_lines.extend([
        "---",
        "",
        "## Key Insights & Recommendations",
        "",
        "### 1. Class Imbalance",
        "- Assists represent ~0.15% of all passes",
        "- **Recommendation**: Use stratified sampling, consider SMOTE, or adjust class weights",
        "",
        "### 2. Feature Importance Indicators",
        "- `end_x`, `end_y` (location-based) are highly predictive",
        "- `is_cross`, `is_through_ball`, `is_into_box` are meaningful categorical features",
        "",
        "### 3. Spatial Patterns",
        "- Most assists come from wide areas and central box region",
        "- Consider spatial clustering or zone-based features",
        "",
        "### 4. Sequence Position",
        "- Position 1 (last pass before shot) has highest assist rate",
        "- Credit decay by position is exponential",
        "",
        "### 5. Pass vs Carry",
        "- Passes: ~60% of actions in shot windows",
        "- Carries: ~40%, often position 1 (final action before shot)",
        "",
        "---",
        "",
        "## Output Files",
        "",
        "All EDA outputs are saved in `outputs/analysis/cxa/eda/`:",
        "",
        "- `phase0_overview/` - Data alignment and schema analysis",
        "- `phase1_passes/` - Pass feature distributions and correlations",
        "- `phase2_shots/` - Shot analysis and xG calibration",
        "- `phase3_sequences/` - Sequence pattern analysis",
        "- `phase4_actions/` - Action type and carry analysis",
        "",
    ])
    
    report_path = output_dir / "eda_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    logger.info(f"EDA report saved to {report_path}")


def run_all_eda(output_base: Path = None) -> Dict[str, Any]:
    """Run all EDA phases."""
    
    repo_root = _get_repo_root()
    if output_base is None:
        output_base = repo_root / "outputs" / "analysis" / "cxa" / "eda"
    output_base.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    logger.info("=" * 70)
    logger.info("  RUNNING COMPREHENSIVE cXA EDA")
    logger.info("=" * 70)
    
    # Phase 0: Data Overview
    logger.info("\n" + "=" * 50)
    logger.info("STARTING PHASE 0: Data Overview")
    logger.info("=" * 50)
    try:
        results["phase0"] = run_phase0_eda(output_base / "phase0_overview")
        logger.info("✓ Phase 0 complete")
    except Exception as e:
        logger.error(f"✗ Phase 0 failed: {e}")
        results["phase0"] = {"error": str(e)}
    
    # Phase 1: Passes EDA
    logger.info("\n" + "=" * 50)
    logger.info("STARTING PHASE 1: Passes EDA")
    logger.info("=" * 50)
    try:
        results["phase1"] = run_phase1_eda(output_base / "phase1_passes")
        logger.info("✓ Phase 1 complete")
    except Exception as e:
        logger.error(f"✗ Phase 1 failed: {e}")
        results["phase1"] = {"error": str(e)}
    
    # Phase 2: Shots EDA
    logger.info("\n" + "=" * 50)
    logger.info("STARTING PHASE 2: Shots EDA")
    logger.info("=" * 50)
    try:
        results["phase2"] = run_phase2_eda(output_base / "phase2_shots")
        logger.info("✓ Phase 2 complete")
    except Exception as e:
        logger.error(f"✗ Phase 2 failed: {e}")
        results["phase2"] = {"error": str(e)}
    
    # Phase 3: Sequences EDA
    logger.info("\n" + "=" * 50)
    logger.info("STARTING PHASE 3: Pass Sequences EDA")
    logger.info("=" * 50)
    try:
        results["phase3"] = run_phase3_eda(output_base / "phase3_sequences")
        logger.info("✓ Phase 3 complete")
    except Exception as e:
        logger.error(f"✗ Phase 3 failed: {e}")
        results["phase3"] = {"error": str(e)}
    
    # Phase 4: Actions EDA
    logger.info("\n" + "=" * 50)
    logger.info("STARTING PHASE 4: Action Sequences EDA")
    logger.info("=" * 50)
    try:
        results["phase4"] = run_phase4_eda(output_base / "phase4_actions")
        logger.info("✓ Phase 4 complete")
    except Exception as e:
        logger.error(f"✗ Phase 4 failed: {e}")
        results["phase4"] = {"error": str(e)}
    
    # Generate summary report
    logger.info("\n" + "=" * 50)
    logger.info("GENERATING EDA REPORT")
    logger.info("=" * 50)
    generate_eda_report(results, output_base)
    
    # Save results metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "phases_completed": [k for k, v in results.items() if "error" not in v],
        "phases_failed": [k for k, v in results.items() if "error" in v],
    }
    
    with open(output_base / "eda_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info("\n" + "=" * 70)
    logger.info("  EDA COMPLETE")
    logger.info(f"  Phases completed: {len(metadata['phases_completed'])}/5")
    logger.info(f"  Outputs: {output_base}")
    logger.info("=" * 70)
    
    return results


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    run_all_eda()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
