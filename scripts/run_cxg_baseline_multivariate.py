"""Run CxG true baselines plus candidate and multivariate experiment rankings."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import pandas as pd
from google.cloud import bigquery
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ID = "oam-varun-260819"
LOCATION = "europe-west2"
FEATURE_DATASET = "oam_features"
ANALYSIS_DATASET = "oam_analysis"
RUN_ID = "cxg-analysis-20260820T201934Z"
BASE_FEATURES = ["shot_x_sb", "shot_y_sb"]
FAMILIES = ["event_context", "defensive_360", "goalkeeper_360", "line_shape_360"]
THREE_SIXTY_FAMILIES = {"defensive_360", "goalkeeper_360", "line_shape_360"}


def q(name: str) -> str:
    return f"`{name}`"


class BaselineRunner:
    def __init__(self, run_id: str, output_dir: Path) -> None:
        self.run_id = run_id
        self.output_dir = output_dir / run_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.client = bigquery.Client(project=PROJECT_ID)

    def query_rows(self, sql: str) -> list[dict[str, object]]:
        return [dict(row.items()) for row in self.client.query(sql, location=LOCATION).result()]

    def decisions(self) -> pd.DataFrame:
        rows = self.query_rows(f"""
        SELECT feature_family, column_name, final_analysis_decision, preliminary_decision, abs_signal
        FROM `{PROJECT_ID}.{ANALYSIS_DATASET}.cxg_final_feature_selection_v1`
        WHERE final_analysis_decision IN ('selected', 'shortlist_review')
        ORDER BY feature_family, final_analysis_decision, abs_signal DESC
        """)
        return pd.DataFrame(rows)

    def feature_sets(self, decisions: pd.DataFrame) -> list[dict[str, object]]:
        by_family = {
            family: decisions.loc[decisions["feature_family"] == family, "column_name"].tolist()
            for family in FAMILIES
        }
        selected = decisions.loc[decisions["final_analysis_decision"] == "selected", "column_name"].tolist()
        specs: list[dict[str, object]] = [
            {"model_name": "event_base_xy", "model_stage": "true_baseline", "cohort": "event_all", "families": ["base"], "features": BASE_FEATURES},
            {"model_name": "event_base_plus_event_context", "model_stage": "candidate_model", "cohort": "event_all", "families": ["base", "event_context"], "features": BASE_FEATURES + by_family["event_context"]},
            {"model_name": "360_base_xy", "model_stage": "true_baseline", "cohort": "360_cohort", "families": ["base"], "features": BASE_FEATURES},
            {"model_name": "360_base_plus_selected_core", "model_stage": "candidate_model", "cohort": "360_cohort", "families": ["base", "selected"], "features": BASE_FEATURES + selected},
        ]
        for size in range(1, len(FAMILIES) + 1):
            for combo in combinations(FAMILIES, size):
                if not any(f in THREE_SIXTY_FAMILIES for f in combo):
                    continue
                features = BASE_FEATURES[:]
                for family in combo:
                    features.extend(by_family[family])
                specs.append({
                    "model_name": "360_experiment_base_plus_" + "_".join(combo),
                    "model_stage": "multivariate_experiment",
                    "cohort": "360_cohort",
                    "families": ["base", *combo],
                    "features": list(dict.fromkeys(features)),
                })
        return specs

    def load_matrix(self, features: list[str], cohort: str) -> pd.DataFrame:
        cols = ["event_id", "match_id", "is_goal", "has_360_frame", *features]
        select_cols = ", ".join(q(c) for c in list(dict.fromkeys(cols)))
        where = "WHERE has_360_frame" if cohort == "360_cohort" else ""
        rows = self.query_rows(f"""
        SELECT {select_cols}
        FROM `{PROJECT_ID}.{FEATURE_DATASET}.cxg_training_matrix_v1`
        {where}
        """)
        return pd.DataFrame(rows)

    def evaluate(self, spec: dict[str, object]) -> dict[str, object]:
        features = list(spec["features"])
        df = self.load_matrix(features, str(spec["cohort"]))
        df = df.dropna(subset=["is_goal", "match_id"])
        y = df["is_goal"].astype(int).to_numpy()
        groups = df["match_id"].astype(str).to_numpy()
        x = df[features].copy()
        for col in features:
            x[col] = pd.to_numeric(x[col], errors="coerce")
        folds = min(5, len(set(groups)))
        cv = GroupKFold(n_splits=folds)
        preds = np.zeros(len(df), dtype=float)
        fold_rows = []
        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(x, y, groups), start=1):
            pipe = Pipeline([
                ("prep", ColumnTransformer([("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), features)], remainder="drop")),
                ("model", LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")),
            ])
            pipe.fit(x.iloc[train_idx], y[train_idx])
            p = pipe.predict_proba(x.iloc[test_idx])[:, 1]
            preds[test_idx] = p
            fold_rows.append({
                "run_id": self.run_id,
                "model_name": spec["model_name"],
                "fold_index": fold_idx,
                "row_count": int(len(test_idx)),
                "goal_count": int(y[test_idx].sum()),
                "auc": self.safe_auc(y[test_idx], p),
                "log_loss": float(log_loss(y[test_idx], p, labels=[0, 1])),
                "brier": float(brier_score_loss(y[test_idx], p)),
            })
        result = {
            "run_id": self.run_id,
            "model_name": spec["model_name"],
            "model_stage": spec.get("model_stage", "unspecified"),
            "cohort": spec["cohort"],
            "families": ",".join(spec["families"]),
            "feature_count": len(features),
            "features": ",".join(features),
            "row_count": int(len(df)),
            "goal_count": int(y.sum()),
            "goal_rate": float(y.mean()),
            "auc": self.safe_auc(y, preds),
            "log_loss": float(log_loss(y, preds, labels=[0, 1])),
            "brier": float(brier_score_loss(y, preds)),
            "fold_count": folds,
            "evaluated_at": datetime.now(UTC).isoformat(),
        }
        result["folds"] = fold_rows
        return result

    @staticmethod
    def safe_auc(y: np.ndarray, p: np.ndarray) -> float | None:
        if len(set(y.tolist())) < 2:
            return None
        return float(roc_auc_score(y, p))

    def run(self) -> dict[str, object]:
        decisions = self.decisions()
        specs = self.feature_sets(decisions)
        results = [self.evaluate(spec) for spec in specs]
        summary_rows = [{k: v for k, v in row.items() if k != "folds"} for row in results]
        fold_rows = [fold for row in results for fold in row["folds"]]
        self.write_outputs(summary_rows, fold_rows)
        self.materialize(summary_rows, fold_rows)
        return {
            "run_id": self.run_id,
            "model_count": len(summary_rows),
            "fold_count": len(fold_rows),
            "best_auc": max((r["auc"] or 0) for r in summary_rows),
            "best_model": sorted(summary_rows, key=lambda r: (r["auc"] or 0, -r["log_loss"]), reverse=True)[0]["model_name"],
            "output_dir": str(self.output_dir),
        }

    def write_outputs(self, summary_rows: list[dict[str, object]], fold_rows: list[dict[str, object]]) -> None:
        out = self.output_dir / "baseline_multivariate"
        out.mkdir(parents=True, exist_ok=True)
        (out / "cxg_baseline_multivariate_results.json").write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
        (out / "cxg_baseline_multivariate_folds.json").write_text(json.dumps(fold_rows, indent=2), encoding="utf-8")
        ranked = sorted(summary_rows, key=lambda r: (r["cohort"], -(r["auc"] or 0), r["log_loss"]))
        lines = ["# CxG True Baseline + Candidate/Experiment Ranking", "", f"Run: `{self.run_id}`", "", "| Rank | Stage | Cohort | Model | Families | Features | Rows | Goals | AUC | Log loss | Brier |", "|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|"]
        for i, r in enumerate(ranked, start=1):
            auc = "" if r["auc"] is None else f"{r['auc']:.4f}"
            lines.append(f"| {i} | {r['model_stage']} | {r['cohort']} | `{r['model_name']}` | {r['families']} | {r['feature_count']} | {r['row_count']} | {r['goal_count']} | {auc} | {r['log_loss']:.4f} | {r['brier']:.4f} |")
        lines.append("")
        lines.append("Fold-safe grouping: GroupKFold by match_id. True baseline means shot_x_sb + shot_y_sb only. Candidate models add selected/shortlist features. Multivariate experiments rank broader family combinations and are not the basic baseline.")
        (out / "cxg_baseline_multivariate_report.md").write_text("\n".join(lines), encoding="utf-8")

    def materialize(self, summary_rows: list[dict[str, object]], fold_rows: list[dict[str, object]]) -> None:
        summary_table = f"{PROJECT_ID}.{ANALYSIS_DATASET}.cxg_baseline_multivariate_results_v1"
        fold_table = f"{PROJECT_ID}.{ANALYSIS_DATASET}.cxg_baseline_multivariate_folds_v1"
        job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE", autodetect=True)
        self.client.load_table_from_json(summary_rows, summary_table, job_config=job_config, location=LOCATION).result()
        self.client.load_table_from_json(fold_rows, fold_table, job_config=job_config, location=LOCATION).result()
        baseline_rows = [row for row in summary_rows if row.get("model_stage") == "true_baseline"]
        self.client.load_table_from_json(
            baseline_rows,
            f"{PROJECT_ID}.{ANALYSIS_DATASET}.cxg_true_baseline_results_v1",
            job_config=job_config,
            location=LOCATION,
        ).result()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--output-dir", default="audit_outputs/cxg_analysis")
    args = parser.parse_args()
    runner = BaselineRunner(args.run_id, Path(args.output_dir))
    print(json.dumps(runner.run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

