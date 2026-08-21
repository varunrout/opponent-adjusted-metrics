"""Clean CxG analysis back to a univariate-only result state."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from google.cloud import bigquery, storage

PROJECT_ID = "oam-varun-260819"
LOCATION = "europe-west2"
DATASET = "oam_analysis"
BUCKET = "oam-varun-260819-artifacts"
RUN_ID = "cxg-analysis-20260820T201934Z"
ROOT = Path(__file__).resolve().parents[1]

DELETE_TABLES = [
    "cxg_above_baseline_model_spec_v1",
    "cxg_baseline_spec_v1",
    "cxg_model_spec_v1",
    "cxg_binned_target_response_v1",
    "cxg_correlation_v1",
    "cxg_cross_family_correlation_v1",
    "cxg_family_summary_v1",
    "cxg_feature_decision_manifest_v1",
    "cxg_final_feature_selection_v1",
    "cxg_findings_chart_registry_v1",
    "cxg_findings_detail_chart_registry_v1",
    "cxg_pair_interaction_target_v1",
    "cxg_redundancy_pairs_v1",
    "cxg_split_bivariate_v1",
    "cxg_split_chart_registry_v1",
    "cxg_split_model_predictions_v1",
    "cxg_split_multivariate_results_v1",
]

PROTECTED_TABLES = [
    "cxg_analysis_event_v1",
    "cxg_analysis_360_v1",
    "cxg_feature_inventory_v1",
    "cxg_null_profile_v1",
    "cxg_summary_stats_v1",
    "cxg_eda_distribution_bins_v1",
    "cxg_univariate_target_v1",
    "cxg_match_splits_v1",
    "cxg_event_model_matrix_v1",
    "cxg_plus_360_model_matrix_v1",
    "cxg_split_balance_v1",
    "cxg_split_univariate_v1",
    "cxg_chart_registry_v1",
    "cxg_rendered_chart_registry_v1",
    "cxg_feature_eda_chart_registry_v1",
]

LOCAL_DELETE_DIRS = [
    "baseline_multivariate",
    "baseline_multivariate_charts",
    "findings_charts",
    "findings_detail_charts",
    "split_analysis_charts",
    "split_model_freeze",
    "model_specs",
    "result_based_model_specs",
]

LOCAL_DELETE_FILES = [
    "cxg_final_feature_selection_report.md",
    "findings_analysis_checkpoint.md",
]

GCS_DELETE_PREFIXES = [
    "baseline_multivariate_charts/",
    "findings_charts/",
    "findings_detail_charts/",
    "split_analysis_charts/",
]


def table_ref(name: str) -> str:
    return f"{PROJECT_ID}.{DATASET}.{name}"


def main() -> None:
    bq = bigquery.Client(project=PROJECT_ID)
    gcs = storage.Client(project=PROJECT_ID)

    deleted_tables: list[str] = []
    for name in DELETE_TABLES:
        bq.delete_table(table_ref(name), not_found_ok=True)
        deleted_tables.append(name)

    run_dir = ROOT / "audit_outputs" / "cxg_analysis" / RUN_ID
    deleted_local_dirs: list[str] = []
    for rel in LOCAL_DELETE_DIRS:
        path = run_dir / rel
        if path.exists():
            shutil.rmtree(path)
            deleted_local_dirs.append(str(path))

    deleted_local_files: list[str] = []
    for rel in LOCAL_DELETE_FILES:
        path = run_dir / rel
        if path.exists():
            path.unlink()
            deleted_local_files.append(str(path))

    bucket = gcs.bucket(BUCKET)
    deleted_gcs_objects = 0
    for rel_prefix in GCS_DELETE_PREFIXES:
        prefix = f"analysis/cxg/{RUN_ID}/{rel_prefix}"
        blobs = list(bucket.list_blobs(prefix=prefix))
        for blob in blobs:
            blob.delete()
        deleted_gcs_objects += len(blobs)

    remaining = {
        row.table_name
        for row in bq.query(
            f"SELECT table_name FROM `{PROJECT_ID}.{DATASET}.INFORMATION_SCHEMA.TABLES`",
            location=LOCATION,
        ).result()
    }
    validation = {
        "deleted_tables_absent": [name for name in DELETE_TABLES if name not in remaining],
        "deleted_tables_still_present": [name for name in DELETE_TABLES if name in remaining],
        "protected_tables_present": [name for name in PROTECTED_TABLES if name in remaining],
        "protected_tables_missing": [name for name in PROTECTED_TABLES if name not in remaining],
        "remaining_table_count": len(remaining),
    }
    validation["passed"] = (
        len(validation["deleted_tables_still_present"]) == 0
        and len(validation["protected_tables_missing"]) == 0
    )
    if not validation["passed"]:
        raise RuntimeError(f"Cleanup validation failed: {validation}")

    out = {
        "deleted_tables": deleted_tables,
        "deleted_local_dirs": deleted_local_dirs,
        "deleted_local_files": deleted_local_files,
        "deleted_gcs_objects": deleted_gcs_objects,
        "validation": validation,
    }
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
