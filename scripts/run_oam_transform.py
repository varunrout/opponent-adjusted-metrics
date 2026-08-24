"""Cloud Run Job entrypoint: oam-transform (Bronze -> Silver -> oam_core -> Gold features).

Orchestrates, in order, the existing tested scripts/modules -- no existing module's
source is modified. Each step is invoked either as a subprocess (for scripts that
already have their own CLI) or via direct import for the one step that needs a
module-level constant routed around a confirmed divergence (see step 3 docstring).

  1. build_statsbomb_silver.py   Bronze GCS       -> Silver GCS (ordered-upload, immutable)
  2. publish_oam_core.py         Silver GCS       -> BigQuery oam_core
  3. materialize_gold_cxg (Gold shot-level feature JSON) -> BigQuery oam_features.cxg_shot_features
  4. materialize_cxg_feature_family_tables.py -> BigQuery oam_features family tables + cxg_training_matrix_v1 view

Step 3 note: `materialize_gold_cxg.py` imports its write-target `DATASET` constant from
`audit_cxg_e13_f1_f15.py`, which is currently `"oam_core"`. The live `cxg_shot_features`
table (and everything that reads it downstream, including step 4's default
`--dataset oam_features`) is actually in `oam_features` -- confirmed by direct query
before this orchestrator was written. Rather than edit either module's source, this
entrypoint imports `audit_cxg_e13_f1_f15` first, overrides its `DATASET` attribute to
`"oam_features"`, and only then imports `materialize_gold_cxg` (which computes its own
`DATASET`/`TABLE` module constants from the already-patched value at import time). This
routes the write to the dataset that's actually live and actually read downstream,
without changing either file on disk.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from google.cloud import storage

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for p in (SRC, SCRIPTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

DEFAULT_DATA_VERSION = "b0bc9f22dd77c206ddedc1d742893b3bbe64baec"
DEFAULT_SCHEMA_SUFFIX = "-remediated"  # matches the compliant, canonical Silver prefix (see silver acceptance closure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="oam-transform: Bronze -> Silver -> oam_core -> Gold")
    parser.add_argument("--project-id", default="oam-varun-260819")
    parser.add_argument("--region", default="europe-west2")
    parser.add_argument("--bucket", default="oam-varun-260819-data")
    parser.add_argument("--artifacts-bucket", default="oam-varun-260819-artifacts")
    parser.add_argument("--data-version", default=DEFAULT_DATA_VERSION)
    parser.add_argument("--source-ref", default=DEFAULT_DATA_VERSION)
    parser.add_argument(
        "--silver-output-prefix",
        default=None,
        help="Override the Silver GCS output prefix (defaults to the canonical remediated prefix)",
    )
    parser.add_argument("--skip-silver", action="store_true", help="Skip Silver publish (e.g. already published)")
    parser.add_argument("--skip-core", action="store_true", help="Skip oam_core BigQuery publish")
    parser.add_argument("--skip-gold", action="store_true", help="Skip Gold shot-feature + family-table materialization")
    return parser.parse_args()


def _run(cmd: list[str], step: str) -> None:
    print(f"[oam-transform] >>> {step}: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        raise SystemExit(f"[oam-transform] step failed: {step} (exit={result.returncode})")
    print(f"[oam-transform] <<< {step}: ok", flush=True)


def _silver_output_prefix(args: argparse.Namespace) -> str:
    if args.silver_output_prefix:
        return args.silver_output_prefix
    return f"staged/statsbomb/{args.data_version}/statsbomb_silver_v1_2{DEFAULT_SCHEMA_SUFFIX}"


def _silver_already_published(args: argparse.Namespace, output_prefix: str) -> bool:
    """Same existence check `build_statsbomb_silver` runs internally (the `_SUCCESS` marker
    at the deterministic, data_version-keyed output prefix) -- run here first so the common
    no-new-data case (the Workflows chain always invokes this job with default args, with no
    way to know in advance whether oam-ingest fetched anything new) is a clean skip instead of
    a hard failure. This does NOT weaken the immutability guard itself: `build_statsbomb_silver`
    is untouched and still raises loudly if it's ever invoked against a prefix that already has
    a `_SUCCESS` marker -- this only decides whether to invoke it at all."""
    client = storage.Client(project=args.project_id)
    blob = client.bucket(args.bucket).blob(f"{output_prefix.rstrip('/')}/_SUCCESS")
    return blob.exists(client=client)


def main() -> None:
    args = parse_args()
    output_prefix = _silver_output_prefix(args)

    if args.skip_silver:
        print("[oam-transform] skip-silver: assuming Silver prefix already published", flush=True)
    elif _silver_already_published(args, output_prefix):
        print(
            f"[oam-transform] Silver output already published at "
            f"gs://{args.bucket}/{output_prefix}/_SUCCESS -- no new data_version, skipping rebuild "
            "(this is the common Workflows-chain case, not an error)",
            flush=True,
        )
    else:
        _run(
            [
                sys.executable,
                str(SCRIPTS / "build_statsbomb_silver.py"),
                "--bucket", args.bucket,
                "--artifacts-bucket", args.artifacts_bucket,
                "--project-id", args.project_id,
                "--data-version", args.data_version,
                "--source-ref", args.source_ref,
                "--output-prefix", output_prefix,
            ],
            "build_statsbomb_silver",
        )

    if not args.skip_core:
        _run(
            [
                sys.executable,
                str(SCRIPTS / "publish_oam_core.py"),
                "--project-id", args.project_id,
                "--location", args.region,
                "--bucket", args.bucket,
                "--data-version", args.data_version,
                "--output-prefix", output_prefix,
            ],
            "publish_oam_core",
        )
    else:
        print("[oam-transform] skip-core: assuming oam_core already reconciled", flush=True)

    if not args.skip_gold:
        print("[oam-transform] >>> materialize_gold_cxg (routed to oam_features)", flush=True)
        # Import normally first: audit_cxg_e13_f1_f15.DATASET must stay "oam_core" so its own
        # _match_ids/_fetch_events/_fetch_frames (which read oam_core.events/three_sixty_*)
        # keep working -- patching that shared constant globally (an earlier version of this
        # entrypoint did exactly that) broke those read functions during real end-to-end
        # validation. Only materialize_gold_cxg's own copied DATASET/TABLE bindings -- used
        # solely for its write target -- are overridden, after import.
        import materialize_gold_cxg as gold_mod  # noqa: E402

        gold_mod.DATASET = "oam_features"
        gold_mod.TABLE = f"{gold_mod.PROJECT}.oam_features.cxg_shot_features"
        gold_mod.main()
        print("[oam-transform] <<< materialize_gold_cxg: ok", flush=True)

        _run(
            [
                sys.executable,
                str(SCRIPTS / "materialize_cxg_feature_family_tables.py"),
                "--project-id", args.project_id,
                "--dataset", "oam_features",
                "--location", args.region,
            ],
            "materialize_cxg_feature_family_tables",
        )
    else:
        print("[oam-transform] skip-gold: assuming Gold tables already materialized", flush=True)

    print("[oam-transform] all steps complete", flush=True)


if __name__ == "__main__":
    main()
