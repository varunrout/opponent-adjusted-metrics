"""Render detailed CxG findings charts for broader feature review."""

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
DATASET = "oam_analysis"
BUCKET = "oam-varun-260819-artifacts"
PREFIX = "analysis/cxg"
FAMILIES = ["event_context", "defensive_360", "goalkeeper_360", "line_shape_360"]


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value)[:180]


class DetailRenderer:
    def __init__(self, run_id: str, local_output_dir: Path, skip_upload: bool = False) -> None:
        self.run_id = run_id
        self.skip_upload = skip_upload
        self.out_dir = local_output_dir / run_id / "findings_detail_charts"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.bq = bigquery.Client(project=PROJECT_ID)
        self.gcs = storage.Client(project=PROJECT_ID)

    def query_df(self, sql: str) -> pd.DataFrame:
        rows = list(self.bq.query(sql, location=LOCATION).result())
        return pd.DataFrame([dict(row.items()) for row in rows]) if rows else pd.DataFrame()

    def render(self) -> dict[str, object]:
        specs = self.specs()
        rendered = []
        for spec in specs:
            name = spec["chart_name"]
            title = spec["chart_title"]
            df = self.data(spec)
            html_path = self.out_dir / f"{name}.html"
            png_path = self.out_dir / f"{name}.png"
            fig = self.plot(df, spec)
            fig.write_html(html_path, include_plotlyjs="cdn", full_html=True)
            self.write_png(df, spec, png_path)
            if not self.skip_upload:
                self.upload(html_path)
                self.upload(png_path)
            rendered.append({
                "chart_name": name,
                "chart_title": title,
                "chart_scope": spec["scope"],
                "backing_table": spec["backing_table"],
                "html_uri": self.uri(html_path.name),
                "png_uri": self.uri(png_path.name),
            })
        manifest = {
            "run_id": self.run_id,
            "rendered_at": datetime.now(UTC).isoformat(),
            "chart_count": len(rendered),
            "html_count": len(rendered),
            "png_count": len(rendered),
            "local_prefix": str(self.out_dir),
            "artifact_prefix": f"gs://{BUCKET}/{PREFIX}/{self.run_id}/findings_detail_charts/",
            "upload_skipped": self.skip_upload,
            "charts": rendered,
        }
        manifest_path = self.out_dir / "findings_detail_charts_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        if not self.skip_upload:
            self.upload(manifest_path)
            self.materialize_registry(rendered)
        return manifest

    def specs(self) -> list[dict[str, str]]:
        specs: list[dict[str, str]] = []
        for family in FAMILIES:
            specs.extend([
                {"chart_name": f"{family}__decision_detail", "chart_title": f"{family}: decision detail", "chart_type": "decision_detail", "scope": family, "backing_table": "cxg_feature_decision_manifest_v1"},
                {"chart_name": f"{family}__target_bins_top", "chart_title": f"{family}: top target-response bins", "chart_type": "target_bins", "scope": family, "backing_table": "cxg_binned_target_response_v1"},
                {"chart_name": f"{family}__pair_interactions_top", "chart_title": f"{family}: top bivariate interactions", "chart_type": "pair_interactions", "scope": family, "backing_table": "cxg_pair_interaction_target_v1"},
                {"chart_name": f"{family}__redundancy_review", "chart_title": f"{family}: redundancy review", "chart_type": "redundancy", "scope": family, "backing_table": "cxg_redundancy_pairs_v1"},
            ])
        for i, left in enumerate(FAMILIES):
            for right in FAMILIES[i + 1:]:
                specs.append({"chart_name": f"{left}__vs__{right}__cross_corr", "chart_title": f"{left} vs {right}: cross-family correlations", "chart_type": "cross_corr", "scope": f"{left}|{right}", "backing_table": "cxg_cross_family_correlation_v1"})
        specs.extend([
            {"chart_name": "all_families__candidate_keep_review", "chart_title": "All families: keep/review candidates", "chart_type": "candidate_detail", "scope": "all", "backing_table": "cxg_feature_decision_manifest_v1"},
            {"chart_name": "all_families__watchlist_sparse", "chart_title": "All families: sparse watchlist", "chart_type": "sparse_detail", "scope": "all", "backing_table": "cxg_feature_decision_manifest_v1"},
            {"chart_name": "all_families__weak_signal_review", "chart_title": "All families: weak signal review", "chart_type": "weak_detail", "scope": "all", "backing_table": "cxg_feature_decision_manifest_v1"},
        ])
        return specs

    def data(self, spec: dict[str, str]) -> pd.DataFrame:
        typ = spec["chart_type"]
        scope = spec["scope"]
        if typ == "decision_detail":
            return self.query_df(f"""
            SELECT column_name, preliminary_decision, non_null_count, null_pct, point_biserial_corr, abs_signal
            FROM `{PROJECT_ID}.{DATASET}.cxg_feature_decision_manifest_v1`
            WHERE feature_family = '{scope}'
            ORDER BY abs_signal DESC, non_null_count DESC
            """)
        if typ == "target_bins":
            return self.query_df(f"""
            SELECT column_name, bin_label, row_count, goal_count, goal_rate, lift_vs_baseline
            FROM `{PROJECT_ID}.{DATASET}.cxg_binned_target_response_v1`
            WHERE feature_family = '{scope}' AND row_count >= 30 AND lift_vs_baseline IS NOT NULL
            ORDER BY lift_vs_baseline DESC, row_count DESC
            LIMIT 40
            """)
        if typ == "pair_interactions":
            return self.query_df(f"""
            SELECT left_column, right_column, left_bin, right_bin, row_count, goal_count, goal_rate, lift_vs_baseline
            FROM `{PROJECT_ID}.{DATASET}.cxg_pair_interaction_target_v1`
            WHERE feature_family = '{scope}' AND row_count >= 60 AND lift_vs_baseline IS NOT NULL
            ORDER BY lift_vs_baseline DESC, row_count DESC
            LIMIT 40
            """)
        if typ == "redundancy":
            return self.query_df(f"""
            SELECT left_column, right_column, pearson_corr, pair_non_null_count, redundancy_band
            FROM `{PROJECT_ID}.{DATASET}.cxg_redundancy_pairs_v1`
            WHERE left_family = '{scope}' OR right_family = '{scope}'
            ORDER BY ABS(pearson_corr) DESC, pair_non_null_count DESC
            LIMIT 40
            """)
        if typ == "cross_corr":
            left, right = scope.split("|")
            return self.query_df(f"""
            SELECT left_column, right_column, pearson_corr, pair_non_null_count
            FROM `{PROJECT_ID}.{DATASET}.cxg_cross_family_correlation_v1`
            WHERE ((left_family = '{left}' AND right_family = '{right}') OR (left_family = '{right}' AND right_family = '{left}'))
              AND pair_non_null_count >= 200
            ORDER BY ABS(pearson_corr) DESC, pair_non_null_count DESC
            LIMIT 40
            """)
        if typ == "candidate_detail":
            return self.query_df(f"""
            SELECT feature_family, column_name, preliminary_decision, non_null_count, null_pct, point_biserial_corr, abs_signal
            FROM `{PROJECT_ID}.{DATASET}.cxg_feature_decision_manifest_v1`
            WHERE preliminary_decision IN ('candidate_keep', 'candidate_review', 'redundant_review')
            ORDER BY abs_signal DESC, non_null_count DESC
            LIMIT 80
            """)
        if typ == "sparse_detail":
            return self.query_df(f"""
            SELECT feature_family, column_name, preliminary_decision, non_null_count, null_pct, point_biserial_corr, abs_signal
            FROM `{PROJECT_ID}.{DATASET}.cxg_feature_decision_manifest_v1`
            WHERE preliminary_decision = 'watchlist_sparse'
            ORDER BY abs_signal DESC, non_null_count DESC
            """)
        if typ == "weak_detail":
            return self.query_df(f"""
            SELECT feature_family, column_name, preliminary_decision, non_null_count, null_pct, point_biserial_corr, abs_signal
            FROM `{PROJECT_ID}.{DATASET}.cxg_feature_decision_manifest_v1`
            WHERE preliminary_decision = 'weak_signal_review'
            ORDER BY feature_family, abs_signal DESC, non_null_count DESC
            LIMIT 90
            """)
        raise ValueError(typ)

    def plot(self, df: pd.DataFrame, spec: dict[str, str]):
        title = spec["chart_title"]
        typ = spec["chart_type"]
        if df.empty:
            fig = px.scatter(title=f"{title} - no data")
        elif typ in {"decision_detail", "candidate_detail", "sparse_detail", "weak_detail"}:
            color = "preliminary_decision" if typ == "decision_detail" else "feature_family"
            label = "column_name"
            fig = px.scatter(df, x="null_pct", y="abs_signal", color=color, size="non_null_count", hover_name=label, hover_data=["point_biserial_corr", "non_null_count"])
            fig.update_xaxes(title="Null %")
            fig.update_yaxes(title="Absolute target signal")
        elif typ == "target_bins":
            tmp = df.assign(label=df["column_name"] + " | " + df["bin_label"].astype(str)).sort_values("lift_vs_baseline")
            fig = px.bar(tmp, x="lift_vs_baseline", y="label", orientation="h", hover_data=["row_count", "goal_count", "goal_rate"])
        elif typ == "pair_interactions":
            tmp = df.assign(label=df["left_column"] + " + " + df["right_column"] + " | " + df["left_bin"].astype(str) + " / " + df["right_bin"].astype(str)).sort_values("lift_vs_baseline")
            fig = px.bar(tmp, x="lift_vs_baseline", y="label", orientation="h", hover_data=["row_count", "goal_count", "goal_rate"])
        elif typ == "cross_corr":
            tmp = df.assign(label=df["left_column"] + " <> " + df["right_column"]).sort_values("pearson_corr")
            fig = px.bar(tmp, x="pearson_corr", y="label", orientation="h", hover_data=["pair_non_null_count"])
        else:
            tmp = df.assign(label=df["left_column"] + " <> " + df["right_column"]).sort_values("pearson_corr")
            fig = px.bar(tmp, x="pearson_corr", y="label", color="redundancy_band", orientation="h", hover_data=["pair_non_null_count"])
        fig.update_layout(title=title, template="plotly_white", font={"family": "Arial", "size": 13}, margin={"l": 72, "r": 30, "t": 72, "b": 72})
        return fig

    def write_png(self, df: pd.DataFrame, spec: dict[str, str], path: Path) -> None:
        title = spec["chart_title"]
        typ = spec["chart_type"]
        fig, ax = plt.subplots(figsize=(14.5, 9.2), dpi=150)
        if df.empty:
            ax.text(0.5, 0.5, "No data available", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
        elif typ in {"decision_detail", "candidate_detail", "sparse_detail", "weak_detail"}:
            color_col = "preliminary_decision" if typ == "decision_detail" else "feature_family"
            groups = list(dict.fromkeys(df[color_col].astype(str)))
            cmap = plt.get_cmap("tab10")
            colors = {g: cmap(i % 10) for i, g in enumerate(groups)}
            for g, group in df.groupby(color_col):
                ax.scatter(group["null_pct"], group["abs_signal"], s=group["non_null_count"].clip(lower=100) / 40, alpha=0.72, label=str(g), color=colors[str(g)])
            top = df.sort_values("abs_signal", ascending=False).head(12)
            for _, row in top.iterrows():
                ax.annotate(str(row["column_name"])[:26], (row["null_pct"], row["abs_signal"]), fontsize=7, xytext=(4, 3), textcoords="offset points")
            ax.set_xlabel("Null %")
            ax.set_ylabel("Absolute target signal")
            ax.grid(True, alpha=0.18)
            ax.legend(fontsize=8, loc="best")
        else:
            if typ == "target_bins":
                tmp = df.assign(label=df["column_name"] + " | " + df["bin_label"].astype(str), x=df["lift_vs_baseline"], group="target bin")
                xlabel = "Lift vs baseline"
            elif typ == "pair_interactions":
                tmp = df.assign(label=df["left_column"] + " + " + df["right_column"], x=df["lift_vs_baseline"], group="pair")
                xlabel = "Lift vs baseline"
            elif typ == "cross_corr":
                tmp = df.assign(label=df["left_column"] + " <> " + df["right_column"], x=df["pearson_corr"], group="correlation")
                xlabel = "Pearson r"
            else:
                tmp = df.assign(label=df["left_column"] + " <> " + df["right_column"], x=df["pearson_corr"], group=df["redundancy_band"])
                xlabel = "Pearson r"
            tmp = tmp.sort_values("x", ascending=True).tail(30)
            groups = list(dict.fromkeys(tmp["group"].astype(str)))
            cmap = plt.get_cmap("tab10")
            colors = {g: cmap(i % 10) for i, g in enumerate(groups)}
            y = list(range(len(tmp)))
            ax.barh(y, tmp["x"], color=[colors[str(v)] for v in tmp["group"]], alpha=0.86)
            labels = [str(v)[:64] + ("..." if len(str(v)) > 64 else "") for v in tmp["label"]]
            ax.set_yticks(y)
            ax.set_yticklabels(labels, fontsize=7)
            ax.set_xlabel(xlabel)
            ax.grid(axis="x", alpha=0.18)
            if len(groups) > 1:
                handles = [plt.Rectangle((0, 0), 1, 1, color=colors[g]) for g in groups[:10]]
                ax.legend(handles, groups[:10], fontsize=8, loc="best")
        ax.set_title(title, fontsize=14, pad=14)
        fig.tight_layout()
        fig.savefig(path, format="png")
        plt.close(fig)

    def uri(self, name: str) -> str:
        return f"gs://{BUCKET}/{PREFIX}/{self.run_id}/findings_detail_charts/{name}"

    def upload(self, path: Path) -> None:
        content_type = "text/html" if path.suffix == ".html" else "image/png" if path.suffix == ".png" else "application/json"
        self.gcs.bucket(BUCKET).blob(f"{PREFIX}/{self.run_id}/findings_detail_charts/{path.name}").upload_from_filename(path, content_type=content_type)

    def materialize_registry(self, rendered: list[dict[str, object]]) -> None:
        rows = []
        for chart in rendered:
            rows.append(
                "SELECT "
                f"'{self.run_id}' AS run_id, "
                f"'{chart['chart_name']}' AS chart_name, "
                f"'{chart['chart_title']}' AS chart_title, "
                f"'{chart['chart_scope']}' AS chart_scope, "
                f"'{chart['backing_table']}' AS backing_table, "
                f"'{chart['html_uri']}' AS html_uri, "
                f"'{chart['png_uri']}' AS png_uri, "
                "CURRENT_TIMESTAMP() AS rendered_at"
            )
        sql = f"CREATE OR REPLACE TABLE `{PROJECT_ID}.{DATASET}.cxg_findings_detail_chart_registry_v1` AS\n" + "\nUNION ALL\n".join(rows)
        self.bq.query(sql, location=LOCATION).result()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--local-output-dir", default="audit_outputs/cxg_analysis")
    parser.add_argument("--skip-upload", action="store_true")
    args = parser.parse_args()
    renderer = DetailRenderer(args.run_id, Path(args.local_output_dir), skip_upload=args.skip_upload)
    print(json.dumps(renderer.render(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
