"""Convenience runner that scores a dataset and builds aggregate reports."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from opponent_adjusted.modeling.modeling_utilities import get_modeling_output_dir
from opponent_adjusted.prediction.aggregate_reports import aggregate_to_matches, aggregate_to_table
from opponent_adjusted.prediction.score_dataset import score_dataset

PREDICTION_OUTPUT_ROOT = get_modeling_output_dir("cxg", subdir="prediction_runs")


@dataclass
class PredictionArtifacts:
    """Collection of paths emitted by the prediction pipeline."""

    run_dir: Path
    scored_dataset: Path
    match_aggregates: Path
    team_aggregates: Path


def _resolve_run_directory(output_dir: Optional[Path], tag: Optional[str]) -> Path:
    if output_dir is not None:
        return output_dir
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    default_dir = PREDICTION_OUTPUT_ROOT / (tag or timestamp)
    return default_dir


def run_prediction_pipeline(
    dataset_path: Path,
    output_dir: Optional[Path] = None,
    *,
    model_label: str = "contextual_model",
    tag: Optional[str] = None,
    scored_format: str = "parquet",
) -> PredictionArtifacts:
    dataset_path = Path(dataset_path)
    run_dir = _resolve_run_directory(output_dir, tag)
    run_dir.mkdir(parents=True, exist_ok=True)

    if scored_format not in {"csv", "parquet"}:
        raise ValueError("scored_format must be either 'csv' or 'parquet'")

    scored_name = f"scored_shots.{scored_format}"
    scored_path = run_dir / scored_name
    match_path = run_dir / "match_aggregates.csv"
    team_path = run_dir / "team_aggregates.csv"

    score_dataset(dataset_path, scored_path, model_label)
    match_df = aggregate_to_matches(scored_path, match_path)
    aggregate_to_table(match_df, team_path)

    return PredictionArtifacts(
        run_dir=run_dir,
        scored_dataset=scored_path,
        match_aggregates=match_path,
        team_aggregates=team_path,
    )


if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="Run the end-to-end CxG prediction pipeline")
    parser.add_argument("dataset", type=Path, help="Path to the enriched dataset to score")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory to store outputs (defaults to outputs/modeling/cxg/prediction_runs/<timestamp>)",
    )
    parser.add_argument("--model-label", default="contextual_model", help="Model artifact label")
    parser.add_argument("--tag", default=None, help="Override for timestamped run directory name")
    parser.add_argument(
        "--scored-format",
        default="parquet",
        choices=["csv", "parquet"],
        help="File format for the intermediate scored dataset",
    )
    args = parser.parse_args()

    artifacts = run_prediction_pipeline(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        model_label=args.model_label,
        tag=args.tag,
        scored_format=args.scored_format,
    )
    print(f"Prediction artifacts written to {artifacts.run_dir}")
