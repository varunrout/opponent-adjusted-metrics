"""Render per-feature CxG EDA appendix charts for every non-lineage feature."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
from google.cloud import bigquery, storage

PROJECT_ID = "oam-varun-260819"
LOCATION = "europe-west2"
ANALYSIS_DATASET = "oam_analysis"
ARTIFACT_BUCKET = "oam-varun-260819-artifacts"
ARTIFACT_PREFIX = "analysis/cxg"

# Explicit feature_family -> raw-value analysis surface mapping. Kept identical to
# `opponent_adjusted.analysis.cxg_charts.FAMILY_SURFACE` -- deliberately not a `_360`-suffix
# heuristic: `opponent_adjusted` is a 360-population family but doesn't end in `_360`, and a
# suffix check would silently mis-route it (or any future non-matching family name).
FAMILY_SURFACE = {
    "base_identity_target": "cxg_analysis_event_v1",
    "shot_geometry": "cxg_analysis_event_v1",
    "event_context": "cxg_analysis_event_v1",
    "buildup": "cxg_analysis_event_v1",
    "defensive_360": "cxg_analysis_360_v1",
    "goalkeeper_360": "cxg_analysis_360_v1",
    "line_shape_360": "cxg_analysis_360_v1",
    "opponent_adjusted": "cxg_analysis_opponent_adjusted_v1",
}

# `defensive_profile_cluster` is stored INT64 (`cluster_label`) with only 4 distinct values.
# This renderer's dtype-based branch (`dtype in {"STRING", "BOOL"}`) would otherwise route it
# into the numeric histogram path. Force it categorical explicitly.
CATEGORICAL_OVERRIDE_COLUMNS = {"defensive_profile_cluster"}

FEATURE_EDA_CHART_REGISTRY_SCHEMA = [
    bigquery.SchemaField("run_id", "STRING"),
    bigquery.SchemaField("feature_family", "STRING"),
    bigquery.SchemaField("column_name", "STRING"),
    bigquery.SchemaField("html_uri", "STRING"),
    bigquery.SchemaField("png_uri", "STRING"),
    bigquery.SchemaField("rendered_at", "TIMESTAMP"),
]

ID_COLUMNS = {
    "event_id",
    "match_id",
    "player_id",
    "team_id",
    "data_version",
    "feature_version",
    "silver_schema_version",
    "materialized_at",
    "analysis_materialized_at",
}
TARGET_COLUMNS = {"is_goal", "outcome_name"}


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value)[:180]


def surface_for_family(family: str) -> str:
    try:
        return FAMILY_SURFACE[family]
    except KeyError:
        raise ValueError(f"No analysis surface mapping registered for feature_family={family!r}")


class AppendixRenderer:
    def __init__(self, run_id: str, skip_upload: bool = False, local_output_dir: Path | None = None) -> None:
        self.run_id = run_id
        self.skip_upload = skip_upload
        self.local_output_dir = local_output_dir or Path("audit_outputs/cxg_analysis")
        self.bq = bigquery.Client(project=PROJECT_ID)
        self.gcs = storage.Client(project=PROJECT_ID)
        self.out_dir = self.local_output_dir / run_id / "eda_appendix"
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def query_df(self, sql: str) -> pd.DataFrame:
        rows = list(self.bq.query(sql, location=LOCATION).result())
        return pd.DataFrame([dict(row.items()) for row in rows]) if rows else pd.DataFrame()

    def features(self) -> pd.DataFrame:
        return self.query_df(f"""
        SELECT feature_family, column_name, data_type
        FROM `{PROJECT_ID}.{ANALYSIS_DATASET}.cxg_feature_inventory_v1`
        WHERE column_role = 'feature'
          AND column_name NOT IN ({', '.join(repr(c) for c in sorted(ID_COLUMNS | TARGET_COLUMNS))})
          AND feature_family NOT IN ('buildup', 'shot_geometry')
        ORDER BY feature_family, column_name
        """)

    def render(self) -> dict[str, object]:
        rendered = []
        features = self.features()
        for _, row in features.iterrows():
            family = row["feature_family"]
            column = row["column_name"]
            dtype = row["data_type"]
            surface = surface_for_family(family)
            stem = f"{safe_name(family)}__{safe_name(column)}__eda"
            html_path = self.out_dir / f"{stem}.html"
            png_path = self.out_dir / f"{stem}.png"
            if dtype in {"STRING", "BOOL"} or column in CATEGORICAL_OVERRIDE_COLUMNS:
                df = self.query_df(f"""
                SELECT CAST(`{column}` AS STRING) AS value, COUNT(*) AS row_count,
                       AVG(IF(is_goal, 1.0, 0.0)) AS goal_rate
                FROM `{PROJECT_ID}.{ANALYSIS_DATASET}.{surface}`
                WHERE `{column}` IS NOT NULL
                GROUP BY value
                ORDER BY row_count DESC
                LIMIT 30
                """)
                self.render_categorical(df, family, column, html_path, png_path)
            else:
                df = self.query_df(f"""
                SELECT CAST(`{column}` AS FLOAT64) AS value, IF(is_goal, 1.0, 0.0) AS target
                FROM `{PROJECT_ID}.{ANALYSIS_DATASET}.{surface}`
                WHERE `{column}` IS NOT NULL
                """)
                self.render_numeric(df, family, column, html_path, png_path)
            if not self.skip_upload:
                self.upload(html_path)
                self.upload(png_path)
            rendered.append(
                {
                    "feature_family": family,
                    "column_name": column,
                    "html_uri": self.uri(html_path.name),
                    "png_uri": self.uri(png_path.name),
                }
            )
        manifest = {
            "run_id": self.run_id,
            "rendered_at": datetime.now(UTC).isoformat(),
            "feature_chart_count": len(rendered),
            "html_count": len(rendered),
            "png_count": len(rendered),
            "local_prefix": str(self.out_dir),
            "artifact_prefix": f"gs://{ARTIFACT_BUCKET}/{ARTIFACT_PREFIX}/{self.run_id}/eda_appendix/",
            "upload_skipped": self.skip_upload,
            "charts": rendered,
        }
        manifest_path = self.out_dir / "eda_appendix_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        if not self.skip_upload:
            self.upload(manifest_path)
            self.materialize_registry(rendered)
        return manifest

    def render_numeric(self, df: pd.DataFrame, family: str, column: str, html_path: Path, png_path: Path) -> None:
        title = f"{family}: {column} distribution"
        if df.empty:
            fig = px.scatter(title=f"{title} - no non-null data")
        else:
            fig = px.histogram(df, x="value", color="target", nbins=40, barmode="overlay", opacity=0.72,
                               labels={"value": column, "target": "is_goal"}, title=title)
        fig.write_html(html_path, include_plotlyjs="cdn", full_html=True)
        plt_fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=150)
        if df.empty:
            ax.text(0.5, 0.5, "No non-null data", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
        else:
            non_goals = df[df["target"] == 0.0]["value"]
            goals = df[df["target"] == 1.0]["value"]
            ax.hist(non_goals, bins=40, alpha=0.65, label="non-goal", color="#2563eb")
            ax.hist(goals, bins=40, alpha=0.75, label="goal", color="#dc2626")
            ax.set_xlabel(column)
            ax.set_ylabel("Rows")
            ax.grid(axis="y", alpha=0.16)
            ax.legend(fontsize=8)
        ax.set_title(title, fontsize=14, pad=14)
        plt_fig.tight_layout()
        plt_fig.savefig(png_path, format="png")
        plt.close(plt_fig)

    def render_categorical(self, df: pd.DataFrame, family: str, column: str, html_path: Path, png_path: Path) -> None:
        title = f"{family}: {column} categories"
        if df.empty:
            fig = px.scatter(title=f"{title} - no non-null data")
        else:
            fig = px.bar(df, x="value", y="row_count", color="goal_rate", title=title,
                         labels={"value": column, "row_count": "Rows", "goal_rate": "Goal rate"})
        fig.write_html(html_path, include_plotlyjs="cdn", full_html=True)
        plt_fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=150)
        if df.empty:
            ax.text(0.5, 0.5, "No non-null data", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
        else:
            labels = [str(v)[:24] for v in df["value"]]
            bars = ax.bar(range(len(df)), df["row_count"], color="#2563eb", alpha=0.82)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
            ax.set_ylabel("Rows")
            ax.grid(axis="y", alpha=0.16)
            ax2 = ax.twinx()
            ax2.plot(range(len(df)), df["goal_rate"], color="#dc2626", marker="o", linewidth=1.5, label="goal rate")
            ax2.set_ylabel("Goal rate")
            ax2.set_ylim(0, max(0.25, float(df["goal_rate"].max()) * 1.25 if len(df) else 0.25))
        ax.set_title(title, fontsize=14, pad=14)
        plt_fig.tight_layout()
        plt_fig.savefig(png_path, format="png")
        plt.close(plt_fig)

    def uri(self, name: str) -> str:
        return f"gs://{ARTIFACT_BUCKET}/{ARTIFACT_PREFIX}/{self.run_id}/eda_appendix/{name}"

    def upload(self, path: Path) -> None:
        content_type = "text/html" if path.suffix == ".html" else "image/png" if path.suffix == ".png" else "application/json"
        blob_name = f"{ARTIFACT_PREFIX}/{self.run_id}/eda_appendix/{path.name}"
        self.gcs.bucket(ARTIFACT_BUCKET).blob(blob_name).upload_from_filename(path, content_type=content_type)

    def materialize_registry(self, rendered: list[dict[str, object]]) -> None:
        """Additive per run_id: delete this run_id's rows (if any, e.g. a re-render),
        then insert its current rows. Never touches any other run_id's rows -- the
        previous `CREATE OR REPLACE TABLE ... AS SELECT <this run only>` implementation
        silently destroyed every other run's chart-registry history on every upload."""
        if not rendered:
            return
        table_ref = f"{PROJECT_ID}.{ANALYSIS_DATASET}.cxg_feature_eda_chart_registry_v1"
        self.bq.create_table(bigquery.Table(table_ref, schema=FEATURE_EDA_CHART_REGISTRY_SCHEMA), exists_ok=True)

        self.bq.query(
            f"DELETE FROM `{table_ref}` WHERE run_id = @run_id",
            job_config=bigquery.QueryJobConfig(
                query_parameters=[bigquery.ScalarQueryParameter("run_id", "STRING", self.run_id)]
            ),
            location=LOCATION,
        ).result()

        rows = [
            {
                "run_id": self.run_id,
                "feature_family": chart["feature_family"],
                "column_name": chart["column_name"],
                "html_uri": chart["html_uri"],
                "png_uri": chart["png_uri"],
                "rendered_at": datetime.now(UTC).isoformat(),
            }
            for chart in rendered
        ]
        job_config = bigquery.LoadJobConfig(schema=FEATURE_EDA_CHART_REGISTRY_SCHEMA, write_disposition="WRITE_APPEND")
        self.bq.load_table_from_json(rows, table_ref, job_config=job_config, location=LOCATION).result()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--skip-upload", action="store_true")
    parser.add_argument("--local-output-dir", default="audit_outputs/cxg_analysis")
    args = parser.parse_args()
    renderer = AppendixRenderer(args.run_id, skip_upload=args.skip_upload, local_output_dir=Path(args.local_output_dir))
    print(json.dumps(renderer.render(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
