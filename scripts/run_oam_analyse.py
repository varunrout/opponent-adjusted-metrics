"""Cloud Run Job entrypoint: oam-analyse (Gold -> oam_analysis tables + GCS chart artifacts).

Deliberately does NOT invoke `materialize_cxg_analysis.py` / `CxGAnalysisMaterializer.run()`:
that class unconditionally rebuilds `cxg_correlation_v1` (bivariate), which conflicts with
the standing split-policy scope boundary this project has held since 21-Aug (see
`docs/cxg_split_policy_and_parallel_plan.md`). The foundational 6-family univariate/EDA
layer it builds is a one-time manual bootstrap, already done, and is not part of this
automated job. This entrypoint only runs the already-proven, non-correlation-touching
scripts that extend/refresh that layer:

  1. materialize_cxg_defensive_involvement.py   -> oam_analysis.cxg_defensive_involvement_v1
  2. materialize_cxg_odi_features.py            -> oam_analysis.cxg_odi_features_v1
  3. materialize_cxg_defensive_profile_clusters.py -> oam_analysis.cxg_defensive_profile_clusters_v1
  4. materialize_cxg_opponent_adjusted_analysis.py -> INSERT rows into cxg_feature_inventory_v1 /
     cxg_null_profile_v1 / cxg_summary_stats_v1 / cxg_eda_distribution_bins_v1 /
     cxg_univariate_target_v1 / cxg_split_univariate_v1 for the opponent_adjusted family
  5. register_chart_registry_for_run.py <run_id>  -> copies the latest known-good chart
     registry batch forward to a fresh run_id (pruning any row whose backing table no
     longer exists)
  6. render_cxg_analysis_charts.py --run-id <run_id>       -> family-overview charts
  7. render_cxg_feature_eda_appendix.py --run-id <run_id>  -> per-feature EDA charts

Known pre-existing dependency (not introduced by this job, not fixed by it): step 3 reads
`oam_analysis.cxg_plus_360_model_matrix_v1` / `cxg_match_splits_v1`, which are materialized
only by `run_cxg_split_analysis.py` -- an excluded, manual, one-off script (part of the
reverted split/bivariate track). Those tables already exist and persist in the live project,
so this job succeeds today; in a from-scratch environment where they don't exist yet, step 3
would fail with a clear BigQuery NotFound error. This entrypoint pre-flight-checks for their
existence and fails loudly with an explicit message rather than silently skipping step 3.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from google.cloud import bigquery

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

PROJECT = "oam-varun-260819"
ANALYSIS_DATASET = "oam_analysis"
LOCATION = "europe-west2"

REQUIRED_PREEXISTING_TABLES = ("cxg_match_splits_v1", "cxg_plus_360_model_matrix_v1")


def default_run_id() -> str:
    return "cxg-analysis-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="oam-analyse: Gold -> oam_analysis + charts")
    parser.add_argument("--run-id", default=None, help="Chart run_id (default: fresh UTC timestamp, cxg-analysis-<ISO>)")
    parser.add_argument("--skip-upload", action="store_true", help="Render charts locally only, skip GCS upload")
    parser.add_argument("--local-output-dir", default="audit_outputs/cxg_analysis")
    parser.add_argument("--skip-preflight", action="store_true", help="Skip the required-table existence check")
    return parser.parse_args()


def _run(cmd: list[str], step: str) -> None:
    print(f"[oam-analyse] >>> {step}: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        raise SystemExit(f"[oam-analyse] step failed: {step} (exit={result.returncode})")
    print(f"[oam-analyse] <<< {step}: ok", flush=True)


def _preflight(client: bigquery.Client) -> None:
    missing = []
    for table in REQUIRED_PREEXISTING_TABLES:
        try:
            client.get_table(f"{PROJECT}.{ANALYSIS_DATASET}.{table}")
        except Exception:
            missing.append(table)
    if missing:
        raise SystemExit(
            "[oam-analyse] pre-flight failed: required pre-existing table(s) not found: "
            f"{missing}. These are materialized only by the excluded/manual "
            "run_cxg_split_analysis.py script and are not built by this automated job. "
            "Run that script manually first, or pass --skip-preflight to proceed anyway "
            "(materialize_cxg_defensive_profile_clusters.py will then fail on its own)."
        )
    print(f"[oam-analyse] pre-flight ok: {REQUIRED_PREEXISTING_TABLES} present", flush=True)


def main() -> None:
    args = parse_args()
    run_id = args.run_id or default_run_id()

    if not args.skip_preflight:
        _preflight(bigquery.Client(project=PROJECT))

    _run([sys.executable, str(SCRIPTS / "materialize_cxg_defensive_involvement.py")], "materialize_cxg_defensive_involvement")
    _run([sys.executable, str(SCRIPTS / "materialize_cxg_odi_features.py")], "materialize_cxg_odi_features")
    _run([sys.executable, str(SCRIPTS / "materialize_cxg_defensive_profile_clusters.py")], "materialize_cxg_defensive_profile_clusters")
    _run([sys.executable, str(SCRIPTS / "materialize_cxg_opponent_adjusted_analysis.py")], "materialize_cxg_opponent_adjusted_analysis")

    _run([sys.executable, str(SCRIPTS / "register_chart_registry_for_run.py"), run_id], "register_chart_registry_for_run")

    render_cmd = [
        sys.executable, str(SCRIPTS / "render_cxg_analysis_charts.py"),
        "--run-id", run_id,
        "--local-output-dir", args.local_output_dir,
    ]
    if args.skip_upload:
        render_cmd.append("--skip-upload")
    _run(render_cmd, "render_cxg_analysis_charts")

    appendix_cmd = [
        sys.executable, str(SCRIPTS / "render_cxg_feature_eda_appendix.py"),
        "--run-id", run_id,
        "--local-output-dir", args.local_output_dir,
    ]
    if args.skip_upload:
        appendix_cmd.append("--skip-upload")
    _run(appendix_cmd, "render_cxg_feature_eda_appendix")

    print(f"[oam-analyse] all steps complete, run_id={run_id}", flush=True)


if __name__ == "__main__":
    main()
