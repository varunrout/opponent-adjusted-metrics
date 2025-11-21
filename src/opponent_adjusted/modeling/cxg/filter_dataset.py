"""Filter the raw CxG dataset according to governance rules before modeling."""

from __future__ import annotations

import json
from dataclasses import asdict

import pandas as pd

from opponent_adjusted.modeling.cxg.data_filters import apply_modeling_filters, get_filter_scope
from opponent_adjusted.modeling.modeling_utilities import get_modeling_output_dir


def _load_raw_dataset() -> pd.DataFrame:
    base_dir = get_modeling_output_dir("cxg")
    parquet_path = base_dir / "cxg_dataset.parquet"
    csv_path = base_dir / "cxg_dataset.csv"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError("CxG dataset not found. Run build_cxg_dataset.py first.")


def main() -> None:
    dataset = _load_raw_dataset()
    allowed_matches = get_filter_scope()
    filtered_df, summary = apply_modeling_filters(dataset, allowed_match_ids=allowed_matches)

    base_dir = get_modeling_output_dir("cxg")
    parquet_path = base_dir / "cxg_dataset_filtered.parquet"
    csv_path = base_dir / "cxg_dataset_filtered.csv"
    report_path = base_dir / "filter_report.json"

    filtered_df.to_parquet(parquet_path, index=False)
    filtered_df.to_csv(csv_path, index=False)
    with report_path.open("w", encoding="utf-8") as fp:
        json.dump(summary.to_dict(), fp, indent=2)

    print(
        "Applied modeling filters: removed"
        f" {summary.total_rows - summary.to_dict()['total_after']} shots;"
        f" {summary.to_dict()['total_after']} remain."
    )


if __name__ == "__main__":  # pragma: no cover
    main()
