"""Validate a tabular dataset against a feature contract.

The validator is intentionally lightweight. It checks whether required columns
exist, whether forbidden columns are present, and whether the configured split
group column is available. It does not mutate data.

Examples:
    poetry run python scripts/validate_feature_contract.py \
        --contract configs/feature_contracts/cxg_v1.json \
        --data feature_store/cxg/shots.parquet
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from opponent_adjusted.utils.logging import get_logger  # noqa: E402

logger = get_logger(__name__)


def _flatten_features(contract: dict[str, Any]) -> set[str]:
    columns: set[str] = set()
    for key in (
        "identity_columns",
        "target_columns",
        "required_numeric_features",
        "required_binary_features",
        "required_categorical_features",
    ):
        columns.update(contract.get(key, []))

    for nested_key in ("completion_model_features", "value_model_features"):
        nested = contract.get(nested_key, {})
        for values in nested.values():
            columns.update(values)

    return columns


def _forbidden_features(contract: dict[str, Any]) -> set[str]:
    columns: set[str] = set()
    for key in (
        "forbidden_training_features",
        "forbidden_completion_features",
    ):
        columns.update(contract.get(key, []))
    return columns


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".json", ".jsonl"}:
        return pd.read_json(path, lines=suffix == ".jsonl")
    raise ValueError(f"Unsupported data format: {path.suffix}")


def validate(
    contract_path: Path, data_path: Path, *, allow_forbidden: bool = False
) -> dict[str, Any]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    data = _read_table(data_path)
    actual_columns = set(data.columns)

    required = _flatten_features(contract)
    forbidden = _forbidden_features(contract)
    split_group_column = contract.get("split_group_column")

    missing_required = sorted(required - actual_columns)
    present_forbidden = sorted(forbidden & actual_columns)
    missing_split_group = bool(split_group_column and split_group_column not in actual_columns)

    report = {
        "contract": str(contract_path),
        "data": str(data_path),
        "metric": contract.get("metric"),
        "version": contract.get("version"),
        "rows": int(len(data)),
        "columns": int(len(data.columns)),
        "missing_required": missing_required,
        "present_forbidden": present_forbidden,
        "missing_split_group": missing_split_group,
        "valid": not missing_required
        and not missing_split_group
        and (allow_forbidden or not present_forbidden),
    }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate data against a feature contract")
    parser.add_argument("--contract", type=Path, required=True, help="Feature contract JSON path")
    parser.add_argument(
        "--data", type=Path, required=True, help="CSV, JSON, JSONL, or parquet data path"
    )
    parser.add_argument(
        "--allow-forbidden",
        action="store_true",
        help="Do not fail when forbidden columns are present",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON report path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = validate(args.contract, args.data, allow_forbidden=args.allow_forbidden)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        logger.info("Feature contract report written to %s", args.output)
    print(json.dumps(report, indent=2))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
