#!/usr/bin/env python
"""Validate the generated CxG output contract."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


DEFAULT_FEATURE_STORE_DIR = Path("feature_store") / "cxg"
DEFAULT_MODELING_DIR = Path("outputs") / "modeling" / "cxg"


@dataclass(frozen=True)
class CxGOutputContract:
    """Expected generated CxG output locations."""

    feature_store_dir: Path
    model_path: Path
    metadata_path: Path
    metrics_path: Path
    predictions_path: Path
    player_aggregates_path: Path
    team_aggregates_path: Path
    model_card_path: Path

    @classmethod
    def from_roots(
        cls,
        feature_store_dir: Path = DEFAULT_FEATURE_STORE_DIR,
        modeling_dir: Path = DEFAULT_MODELING_DIR,
    ) -> "CxGOutputContract":
        return cls(
            feature_store_dir=feature_store_dir,
            model_path=modeling_dir / "models" / "contextual_model.joblib",
            metadata_path=modeling_dir / "models" / "contextual_model.json",
            metrics_path=modeling_dir / "reports" / "metrics.json",
            predictions_path=modeling_dir / "predictions" / "shot_predictions.parquet",
            player_aggregates_path=modeling_dir / "aggregates" / "player_cxg.parquet",
            team_aggregates_path=modeling_dir / "aggregates" / "team_cxg.parquet",
            model_card_path=modeling_dir / "reports" / "model_card.md",
        )

    @property
    def files(self) -> tuple[Path, ...]:
        return (
            self.model_path,
            self.metadata_path,
            self.metrics_path,
            self.predictions_path,
            self.player_aggregates_path,
            self.team_aggregates_path,
            self.model_card_path,
        )


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _assert_parquet_has_rows(path: Path) -> None:
    df = pd.read_parquet(path)
    if df.empty:
        raise ValueError(f"{path} must contain at least one row")


def _relative_to_repo(path: Path, repo_root: Path) -> str:
    return path.resolve().relative_to(repo_root.resolve()).as_posix()


def assert_git_ignored(paths: tuple[Path, ...], repo_root: Path) -> None:
    """Assert generated paths are ignored by Git."""

    relative_paths = [_relative_to_repo(path, repo_root) for path in paths]
    result = subprocess.run(
        ["git", "check-ignore", *relative_paths],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    ignored = {line.strip() for line in result.stdout.splitlines()}
    missing = sorted(set(relative_paths).difference(ignored))
    if result.returncode not in (0, 1) or missing:
        raise ValueError(
            "Expected generated CxG paths to be ignored by Git: "
            + ", ".join(missing or relative_paths)
        )


def validate_cxg_outputs(
    contract: CxGOutputContract,
    *,
    repo_root: Path | None = None,
    check_git_ignore: bool = True,
) -> dict[str, str]:
    """Validate generated CxG files and return a path summary."""

    missing = [path for path in (contract.feature_store_dir, *contract.files) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing expected CxG generated outputs: " + ", ".join(str(path) for path in missing)
        )

    model = joblib.load(contract.model_path)
    if not hasattr(model, "predict_proba"):
        raise ValueError(f"{contract.model_path} must load a model with predict_proba")

    metadata = _read_json(contract.metadata_path)
    if metadata.get("artifact_path") != str(contract.model_path):
        raise ValueError(f"{contract.metadata_path} artifact_path must match {contract.model_path}")

    metrics = _read_json(contract.metrics_path)
    for key in ("brier_mean", "log_loss_mean", "n_rows", "n_splits"):
        if key not in metrics:
            raise ValueError(f"{contract.metrics_path} is missing required key {key!r}")

    _assert_parquet_has_rows(contract.predictions_path)
    _assert_parquet_has_rows(contract.player_aggregates_path)
    _assert_parquet_has_rows(contract.team_aggregates_path)

    model_card = contract.model_card_path.read_text(encoding="utf-8")
    if "CxG" not in model_card:
        raise ValueError(f"{contract.model_card_path} does not look like a CxG model card")

    if check_git_ignore:
        assert_git_ignored(
            (
                Path("feature_store"),
                Path("outputs"),
                contract.feature_store_dir,
                contract.model_path,
                contract.predictions_path,
            ),
            repo_root or Path.cwd(),
        )

    return {
        "feature_store_dir": str(contract.feature_store_dir),
        "model_path": str(contract.model_path),
        "metadata_path": str(contract.metadata_path),
        "metrics_path": str(contract.metrics_path),
        "predictions_path": str(contract.predictions_path),
        "player_aggregates_path": str(contract.player_aggregates_path),
        "team_aggregates_path": str(contract.team_aggregates_path),
        "model_card_path": str(contract.model_card_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate generated CxG output files")
    parser.add_argument("--feature-store-dir", type=Path, default=DEFAULT_FEATURE_STORE_DIR)
    parser.add_argument("--modeling-dir", type=Path, default=DEFAULT_MODELING_DIR)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--skip-git-ignore",
        action="store_true",
        help="Only validate files; do not run git check-ignore.",
    )
    args = parser.parse_args()

    summary = validate_cxg_outputs(
        CxGOutputContract.from_roots(args.feature_store_dir, args.modeling_dir),
        repo_root=args.repo_root,
        check_git_ignore=not args.skip_git_ignore,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
