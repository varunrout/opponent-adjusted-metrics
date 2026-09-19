"""Render CxG+ Phase B defender-style cluster-profile charts.

One grouped bar chart per cluster: the cluster's centroid action-mix share
against the cohort mean for the same action, so a reader can see WHICH
actions the archetype is actually built on rather than just its name. Plus
one overview chart showing every cluster's centroid in scaled (z-score)
space -- the space the archetype labels were actually derived from.

Bar rather than radar, deliberately: the seven action shares differ by two
orders of magnitude (Pressure ~0.60, 50/50 ~0.005), and a radar's area
encoding would render the small categories invisible while implying a
cyclic ordering the action taxonomy does not have.

Follows the repository's local-render-first pattern (see
`scripts/render_cxg_feature_eda_appendix.py`): render HTML + PNG locally,
then upload to GCS, then register in `cxg_rendered_chart_registry_v1` with a
SCOPED delete-then-insert on this run_id only. The registry table is created
`exists_ok=True` and never CREATE OR REPLACE'd, so no other run's chart
history is destroyed. The default run_id is phase-scoped
(`phase-b-defstyle-<UTC timestamp>`) so the delete can only ever match rows
this script itself wrote.
"""

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
import plotly.graph_objects as go
from google.cloud import bigquery, storage

from opponent_adjusted.analysis.defstyle.features import STYLE_FEATURES

PROJECT_ID = "oam-varun-260819"
LOCATION = "europe-west2"
ANALYSIS_DATASET = "oam_analysis"
FEATURE_DATASET = "oam_features"
ARTIFACT_BUCKET = "oam-varun-260819-artifacts"
ARTIFACT_PREFIX = "analysis/cxg"

PROFILE_TABLE = f"{PROJECT_ID}.{ANALYSIS_DATASET}.cxg_defender_style_cluster_profile_v1"
CLUSTERS_TABLE = f"{PROJECT_ID}.{ANALYSIS_DATASET}.cxg_defender_style_clusters_v1"
REGISTRY_TABLE = f"{PROJECT_ID}.{ANALYSIS_DATASET}.cxg_rendered_chart_registry_v1"

RENDERED_CHART_REGISTRY_SCHEMA = [
    bigquery.SchemaField("run_id", "STRING"),
    bigquery.SchemaField("chart_name", "STRING"),
    bigquery.SchemaField("html_uri", "STRING"),
    bigquery.SchemaField("png_uri", "STRING"),
    bigquery.SchemaField("rendered_at", "TIMESTAMP"),
]

SHORT_LABEL = {f: f.replace("_share", "").replace("_", " ") for f in STYLE_FEATURES}


class DefenderStyleChartRenderer:
    def __init__(
        self,
        run_id: str,
        skip_upload: bool = False,
        local_output_dir: Path | None = None,
    ) -> None:
        self.run_id = run_id
        self.skip_upload = skip_upload
        base = local_output_dir or (ROOT / "audit_outputs" / "cxg_analysis")
        self.out_dir = base / run_id / "defender_style_charts"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.bq = bigquery.Client(project=PROJECT_ID)
        self.gcs = storage.Client(project=PROJECT_ID) if not skip_upload else None

    def _rows(self, sql: str) -> list[dict]:
        return [dict(row.items()) for row in self.bq.query(sql, location=LOCATION).result()]

    def cohort_means(self) -> dict[str, float]:
        cols = ", ".join(f"AVG(`{f}`) AS `{f}`" for f in STYLE_FEATURES)
        sql = f"SELECT {cols} FROM `{CLUSTERS_TABLE}` WHERE style_archetype IS NOT NULL"
        return self._rows(sql)[0]

    def render(self) -> dict[str, object]:
        profile = sorted(self._rows(f"SELECT * FROM `{PROFILE_TABLE}`"), key=lambda r: r["cluster_label"])
        means = self.cohort_means()

        rendered: list[dict[str, object]] = []
        for row in profile:
            name = f"cxg_defstyle_cluster_{row['cluster_label']}_{row['style_archetype']}"
            rendered.append(self._render_cluster_bar(name, row, means))
        rendered.append(self._render_z_overview("cxg_defstyle_centroid_z_overview", profile))

        manifest = {
            "run_id": self.run_id,
            "rendered_at": datetime.now(UTC).isoformat(),
            "chart_count": len(rendered),
            "local_prefix": str(self.out_dir),
            "artifact_prefix": f"gs://{ARTIFACT_BUCKET}/{ARTIFACT_PREFIX}/{self.run_id}/defender_style_charts/",
            "upload_skipped": self.skip_upload,
            "charts": rendered,
        }
        manifest_path = self.out_dir / "rendered_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        if not self.skip_upload:
            self._upload(manifest_path)
            self._materialize_registry(rendered)
        return manifest

    def _render_cluster_bar(self, name: str, row: dict, means: dict[str, float]) -> dict[str, object]:
        labels = [SHORT_LABEL[f] for f in STYLE_FEATURES]
        centroid = [row[f"{f}_centroid"] for f in STYLE_FEATURES]
        cohort = [means[f] for f in STYLE_FEATURES]
        muddy = " [MUDDY]" if row["is_muddy"] else ""
        title = (
            f"Cluster {row['cluster_label']}: {row['style_archetype']}{muddy} "
            f"(n={row['cluster_size']}, {row['cluster_fraction']:.1%} of clustered defenders)"
        )

        fig = go.Figure(
            [
                go.Bar(name="cluster centroid", x=labels, y=centroid),
                go.Bar(name="cohort mean", x=labels, y=cohort),
            ]
        )
        fig.update_layout(barmode="group", title=title, yaxis_title="share of defensive actions")
        return self._write(name, fig, labels, centroid, cohort, title, "share of defensive actions")

    def _render_z_overview(self, name: str, profile: list[dict]) -> dict[str, object]:
        labels = [SHORT_LABEL[f] for f in STYLE_FEATURES]
        fig = go.Figure()
        series = []
        for row in profile:
            values = [row[f"{f}_z"] for f in STYLE_FEATURES]
            legend = f"{row['cluster_label']}: {row['style_archetype']}"
            fig.add_bar(name=legend, x=labels, y=values)
            series.append((legend, values))
        title = "Defender style centroids in scaled (z-score) space -- basis for archetype labels"
        fig.update_layout(barmode="group", title=title, yaxis_title="cohort standard deviations")

        html_path = self.out_dir / f"{name}.html"
        png_path = self.out_dir / f"{name}.png"
        fig.write_html(html_path, include_plotlyjs="cdn", full_html=True)
        plt_fig, ax = plt.subplots(figsize=(11, 5))
        width = 0.8 / max(len(series), 1)
        for i, (legend, values) in enumerate(series):
            ax.bar([x + i * width for x in range(len(labels))], values, width=width, label=legend)
        ax.set_xticks([x + 0.4 - width / 2 for x in range(len(labels))])
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel("cohort standard deviations")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7)
        plt_fig.tight_layout()
        plt_fig.savefig(png_path, format="png", dpi=120)
        plt.close(plt_fig)
        return self._finish(name, html_path, png_path)

    def _write(
        self,
        name: str,
        fig: go.Figure,
        labels: list[str],
        centroid: list[float],
        cohort: list[float],
        title: str,
        ylabel: str,
    ) -> dict[str, object]:
        html_path = self.out_dir / f"{name}.html"
        png_path = self.out_dir / f"{name}.png"
        fig.write_html(html_path, include_plotlyjs="cdn", full_html=True)

        plt_fig, ax = plt.subplots(figsize=(9, 4.5))
        positions = range(len(labels))
        ax.bar([p - 0.2 for p in positions], centroid, width=0.4, label="cluster centroid")
        ax.bar([p + 0.2 for p in positions], cohort, width=0.4, label="cohort mean")
        ax.set_xticks(list(positions))
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=8)
        plt_fig.tight_layout()
        plt_fig.savefig(png_path, format="png", dpi=120)
        plt.close(plt_fig)
        return self._finish(name, html_path, png_path)

    def _finish(self, name: str, html_path: Path, png_path: Path) -> dict[str, object]:
        if not self.skip_upload:
            self._upload(html_path)
            self._upload(png_path)
        return {
            "chart_name": name,
            "html_uri": self._uri(html_path.name),
            "png_uri": self._uri(png_path.name),
        }

    def _uri(self, name: str) -> str:
        return f"gs://{ARTIFACT_BUCKET}/{ARTIFACT_PREFIX}/{self.run_id}/defender_style_charts/{name}"

    def _upload(self, path: Path) -> None:
        content_type = {
            ".html": "text/html",
            ".png": "image/png",
            ".json": "application/json",
        }.get(path.suffix, "application/octet-stream")
        blob = f"{ARTIFACT_PREFIX}/{self.run_id}/defender_style_charts/{path.name}"
        self.gcs.bucket(ARTIFACT_BUCKET).blob(blob).upload_from_filename(path, content_type=content_type)

    def _materialize_registry(self, rendered: list[dict[str, object]]) -> None:
        """Scoped delete-then-insert for THIS run_id only. Never CREATE OR REPLACE."""
        if not rendered:
            return
        self.bq.create_table(
            bigquery.Table(REGISTRY_TABLE, schema=RENDERED_CHART_REGISTRY_SCHEMA), exists_ok=True
        )
        self.bq.query(
            f"DELETE FROM `{REGISTRY_TABLE}` WHERE run_id = @run_id",
            job_config=bigquery.QueryJobConfig(
                query_parameters=[bigquery.ScalarQueryParameter("run_id", "STRING", self.run_id)]
            ),
            location=LOCATION,
        ).result()
        now = datetime.now(UTC).isoformat()
        rows = [
            {
                "run_id": self.run_id,
                "chart_name": chart["chart_name"],
                "html_uri": chart["html_uri"],
                "png_uri": chart["png_uri"],
                "rendered_at": now,
            }
            for chart in rendered
        ]
        self.bq.load_table_from_json(
            rows,
            REGISTRY_TABLE,
            job_config=bigquery.LoadJobConfig(
                schema=RENDERED_CHART_REGISTRY_SCHEMA, write_disposition="WRITE_APPEND"
            ),
            location=LOCATION,
        ).result()


def default_run_id() -> str:
    return f"phase-b-defstyle-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--skip-upload", action="store_true")
    parser.add_argument("--local-output-dir", default=None)
    args = parser.parse_args()
    renderer = DefenderStyleChartRenderer(
        args.run_id or default_run_id(),
        skip_upload=args.skip_upload,
        local_output_dir=Path(args.local_output_dir) if args.local_output_dir else None,
    )
    print(json.dumps(renderer.render(), indent=2))


if __name__ == "__main__":
    main()
