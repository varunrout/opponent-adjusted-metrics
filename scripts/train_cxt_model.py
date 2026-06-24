#!/usr/bin/env python
"""
Train CxT Model

Trains the contextual xT model with opponent adjustments.
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from opponent_adjusted.config import settings
from opponent_adjusted.modeling.cxt.contextual_model import (
    train_cxt_model,
    evaluate_cxt_model,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train CxT model")
    parser.add_argument(
        "--input",
        type=Path,
        default=settings.feature_store_path / "cxt" / "progressions_featured.parquet",
        help="Input featured parquet file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/modeling/cxt"),
        help="Output directory for model artifacts",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state for reproducibility",
    )
    args = parser.parse_args()

    start_time = datetime.now()

    logger.info("=" * 70)
    logger.info("CxT MODEL TRAINING")
    logger.info("=" * 70)
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"CV Folds: {args.cv_folds}")
    logger.info("=" * 70)

    # Load data
    logger.info("\nLoading featured data...")
    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        return 1

    df = pd.read_parquet(args.input)
    logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")

    # Train model
    logger.info("\nTraining model...")
    model, metrics = train_cxt_model(
        df,
        n_splits=args.cv_folds,
        random_state=args.random_state,
    )

    # Evaluate on full dataset
    logger.info("\nFinal evaluation...")
    eval_metrics = evaluate_cxt_model(model, df)

    # Save model
    run_dir = args.output_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    model.save(run_dir)

    # Save latest symlink/copy
    latest_dir = args.output_dir / "latest"
    if latest_dir.exists():
        import shutil

        shutil.rmtree(latest_dir)
    model.save(latest_dir)

    # Generate predictions
    logger.info("\nGenerating predictions...")
    df["cxt_completion_prob"] = model.predict_completion_prob(df)
    df["cxt_expected_xt_gain"] = model.predict_xt_gain(df)
    df["cxt_value"] = model.predict_cxt(df)

    # Save predictions
    pred_path = run_dir / "predictions.parquet"
    df.to_parquet(pred_path, index=False)
    logger.info(f"Predictions saved: {pred_path}")

    # Generate report
    report_path = run_dir / "training_report.md"
    _generate_report(model, metrics, eval_metrics, report_path)

    elapsed = datetime.now() - start_time

    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Completion AUC: {eval_metrics['completion_auc']:.3f}")
    logger.info(f"xT Gain R²: {eval_metrics['xt_gain_r2']:.3f}")
    logger.info(f"CxT-Actual Correlation: {eval_metrics['cxt_actual_corr']:.3f}")
    logger.info(f"Model saved: {run_dir}")
    logger.info(f"Elapsed time: {elapsed}")
    logger.info("=" * 70)

    return 0


def _generate_report(
    model,
    cv_metrics: dict,
    eval_metrics: dict,
    output_path: Path,
) -> None:
    """Generate training report."""

    lines = [
        "# CxT Model Training Report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Model Overview",
        "",
        "The CxT (Contextual Expected Threat) model predicts the expected threat value",
        "of ball progressions (passes, carries, dribbles) adjusted for opponent strength",
        "and game context.",
        "",
        "**Model Architecture:**",
        "- Completion Model: Logistic Regression (predicts P(success))",
        "- xT Gain Model: Ridge Regression (predicts E[xT_delta | success])",
        "- CxT = P(success) × E[xT_delta | success]",
        "",
        "## Cross-Validation Metrics",
        "",
        "### Completion Model",
        "",
        "| Metric | Mean | Std |",
        "|--------|------|-----|",
        f"| AUC | {cv_metrics['completion']['auc_mean']:.3f} | {cv_metrics['completion']['auc_std']:.3f} |",
        f"| Brier | {cv_metrics['completion']['brier_mean']:.4f} | {cv_metrics['completion']['brier_std']:.4f} |",
        f"| Log Loss | {cv_metrics['completion']['logloss_mean']:.4f} | {cv_metrics['completion']['logloss_std']:.4f} |",
        "",
        "### xT Gain Model",
        "",
        "| Metric | Mean | Std |",
        "|--------|------|-----|",
        f"| R² | {cv_metrics['xt_gain']['r2_mean']:.3f} | {cv_metrics['xt_gain']['r2_std']:.3f} |",
        f"| MAE | {cv_metrics['xt_gain']['mae_mean']:.4f} | {cv_metrics['xt_gain']['mae_std']:.4f} |",
        f"| RMSE | {cv_metrics['xt_gain']['rmse_mean']:.4f} | {cv_metrics['xt_gain']['rmse_std']:.4f} |",
        "",
        "## Final Evaluation",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Completion AUC | {eval_metrics['completion_auc']:.3f} |",
        f"| Completion Brier | {eval_metrics['completion_brier']:.4f} |",
        f"| xT Gain R² | {eval_metrics['xt_gain_r2']:.3f} |",
        f"| xT Gain MAE | {eval_metrics['xt_gain_mae']:.4f} |",
        f"| CxT-Actual Correlation | {eval_metrics['cxt_actual_corr']:.3f} |",
        "",
        "## Feature Summary",
        "",
        f"- Numeric features: {len(model.numeric_features)}",
        f"- Binary features: {len(model.binary_features)}",
        f"- Categorical features: {len(model.categorical_features)}",
        "",
        "### Numeric Features",
        "",
    ]

    for feat in model.numeric_features:
        lines.append(f"- {feat}")

    lines.extend(
        [
            "",
            "### Binary Features",
            "",
        ]
    )

    for feat in model.binary_features[:10]:  # First 10
        lines.append(f"- {feat}")
    if len(model.binary_features) > 10:
        lines.append(f"- ... and {len(model.binary_features) - 10} more")

    lines.extend(
        [
            "",
            "### Categorical Features",
            "",
        ]
    )

    for feat in model.categorical_features:
        lines.append(f"- {feat}")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- **CxT > 0**: Action expected to add threat (positive progression)",
            "- **CxT < 0**: Action expected to reduce threat (negative progression)",
            "- Higher CxT indicates more dangerous progressions considering context",
            "",
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Report saved: {output_path}")


if __name__ == "__main__":
    sys.exit(main())
