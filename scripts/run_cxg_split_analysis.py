"""Create CxG/CxG+ splits and run split-aware uni/bi/multivariate analysis."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ID = "oam-varun-260819"
LOCATION = "europe-west2"
FEATURE_DATASET = "oam_features"
ANALYSIS_DATASET = "oam_analysis"
RUN_ID = "cxg-analysis-20260820T201934Z"
SEED = 260821
BASE_FEATURES = ["shot_x_sb", "shot_y_sb"]
FAMILIES = ["event_context", "defensive_360", "goalkeeper_360", "line_shape_360"]
THREE_SIXTY_FAMILIES = {"defensive_360", "goalkeeper_360", "line_shape_360"}
SPLIT_ORDER = ["train", "validation", "test"]
SPLIT_TARGETS = {"train": 0.70, "validation": 0.15, "test": 0.15}


def table(name: str) -> str:
    return f"{PROJECT_ID}.{ANALYSIS_DATASET}.{name}"


def bq_table(name: str) -> str:
    return f"`{PROJECT_ID}.{ANALYSIS_DATASET}.{name}`"


def qcol(name: str) -> str:
    return f"`{name}`"


class SplitAnalysisRunner:
    def __init__(self, run_id: str, output_dir: Path, seed: int) -> None:
        self.run_id = run_id
        self.seed = seed
        self.output_dir = output_dir / run_id / "split_analysis"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.bq = bigquery.Client(project=PROJECT_ID)

    def query_rows(self, sql: str) -> list[dict[str, object]]:
        return [dict(row.items()) for row in self.bq.query(sql, location=LOCATION).result()]

    def load_json(self, rows: list[dict[str, object]], table_name: str) -> None:
        job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE", autodetect=True)
        self.bq.load_table_from_json(rows, table(table_name), job_config=job_config, location=LOCATION).result()

    def run(self) -> dict[str, object]:
        split_rows = self.create_match_splits()
        self.load_json(split_rows, "cxg_match_splits_v1")
        self.materialize_split_matrices()
        split_balance = self.materialize_split_balance()
        univariate = self.materialize_split_univariate()
        bivariate = self.materialize_split_bivariate()
        multivariate = self.run_split_multivariate()
        self.write_report(split_balance, univariate, bivariate, multivariate)
        return {
            "run_id": self.run_id,
            "seed": self.seed,
            "match_splits": len(split_rows),
            "split_balance_rows": len(split_balance),
            "univariate_rows": univariate["rows"],
            "bivariate_rows": bivariate["rows"],
            "multivariate_rows": multivariate["rows"],
            "output_dir": str(self.output_dir),
        }

    def create_match_splits(self) -> list[dict[str, object]]:
        rows = self.query_rows(f"""
        SELECT
          match_id,
          COUNT(*) AS event_shot_count,
          COUNTIF(is_goal) AS event_goal_count,
          COUNTIF(has_360_frame) AS plus_shot_count,
          COUNTIF(has_360_frame AND is_goal) AS plus_goal_count,
          COUNTIF(has_360_frame) > 0 AS has_360_match
        FROM `{PROJECT_ID}.{FEATURE_DATASET}.cxg_training_matrix_v1`
        GROUP BY match_id
        """)
        rng = random.Random(self.seed)
        totals = {
            "match_count": len(rows),
            "event_shot_count": sum(int(r["event_shot_count"]) for r in rows),
            "event_goal_count": sum(int(r["event_goal_count"]) for r in rows),
            "plus_shot_count": sum(int(r["plus_shot_count"]) for r in rows),
            "plus_goal_count": sum(int(r["plus_goal_count"]) for r in rows),
        }
        target_counts = {split: max(1, round(totals["match_count"] * SPLIT_TARGETS[split])) for split in SPLIT_ORDER}
        # Force exact total after rounding.
        target_counts["train"] += totals["match_count"] - sum(target_counts.values())

        def score(assignments: list[str]) -> float:
            split_stats = {split: {k: 0 for k in totals} for split in SPLIT_ORDER}
            for row, split in zip(rows, assignments, strict=False):
                split_stats[split]["match_count"] += 1
                for metric in ["event_shot_count", "event_goal_count", "plus_shot_count", "plus_goal_count"]:
                    split_stats[split][metric] += int(row[metric])
            weights = {
                "match_count": 0.5,
                "event_shot_count": 1.2,
                "event_goal_count": 3.0,
                "plus_shot_count": 3.0,
                "plus_goal_count": 5.0,
            }
            total_loss = 0.0
            for split in SPLIT_ORDER:
                for metric, weight in weights.items():
                    target = totals[metric] * SPLIT_TARGETS[split]
                    if metric == "match_count":
                        target = target_counts[split]
                    denom = max(1.0, target)
                    total_loss += weight * abs(split_stats[split][metric] - target) / denom
                # Penalize validation/test without enough 360 goals for CxG+ evaluation.
                if split in {"validation", "test"} and split_stats[split]["plus_goal_count"] < 50:
                    total_loss += 10.0
            return total_loss

        # Random search over match-level assignments. This is deterministic via seed and avoids
        # the previous greedy failure where 360-heavy matches clumped into validation.
        best_assignments: list[str] | None = None
        best_score = float("inf")
        weighted_labels = []
        for split in SPLIT_ORDER:
            weighted_labels.extend([split] * target_counts[split])
        for _ in range(8000):
            labels = weighted_labels[:]
            rng.shuffle(labels)
            current_score = score(labels)
            if current_score < best_score:
                best_score = current_score
                best_assignments = labels
        if best_assignments is None:
            raise RuntimeError("Failed to create match splits")

        out: list[dict[str, object]] = []
        for row, split in zip(rows, best_assignments, strict=False):
            out.append({
                "run_id": self.run_id,
                "match_id": str(row["match_id"]),
                "split": split,
                "split_seed": self.seed,
                "has_360_match": bool(row["has_360_match"]),
                "event_shot_count": int(row["event_shot_count"]),
                "event_goal_count": int(row["event_goal_count"]),
                "plus_shot_count": int(row["plus_shot_count"]),
                "plus_goal_count": int(row["plus_goal_count"]),
                "split_score": float(best_score),
                "created_at": datetime.now(UTC).isoformat(),
            })
        return out

    @staticmethod
    def assignment_loss(current: dict[str, int], row: dict[str, object], totals: dict[str, int], target: float) -> float:
        loss = 0.0
        weights = {"event_shot_count": 1.0, "event_goal_count": 2.0, "plus_shot_count": 2.0, "plus_goal_count": 3.5}
        for metric, weight in weights.items():
            after = current[metric] + int(row[metric])
            target_count = max(1.0, totals[metric] * target)
            loss += weight * abs(after - target_count) / target_count
        return loss

    def materialize_split_matrices(self) -> None:
        self.bq.query(f"""
        CREATE OR REPLACE TABLE {bq_table('cxg_event_model_matrix_v1')} AS
        SELECT m.*, s.split, s.split_seed
        FROM `{PROJECT_ID}.{FEATURE_DATASET}.cxg_training_matrix_v1` m
        JOIN {bq_table('cxg_match_splits_v1')} s
          ON CAST(m.match_id AS STRING) = CAST(s.match_id AS STRING)
        """, location=LOCATION).result()
        self.bq.query(f"""
        CREATE OR REPLACE TABLE {bq_table('cxg_plus_360_model_matrix_v1')} AS
        SELECT *
        FROM {bq_table('cxg_event_model_matrix_v1')}
        WHERE has_360_frame
        """, location=LOCATION).result()

    def materialize_split_balance(self) -> list[dict[str, object]]:
        rows = self.query_rows(f"""
        SELECT
          'cxg_event' AS track,
          split,
          COUNT(DISTINCT match_id) AS match_count,
          COUNT(*) AS shot_count,
          COUNTIF(is_goal) AS goal_count,
          AVG(IF(is_goal, 1.0, 0.0)) AS goal_rate,
          COUNTIF(has_360_frame) AS plus_shot_count,
          COUNTIF(has_360_frame AND is_goal) AS plus_goal_count,
          AVG(IF(has_360_frame, 1.0, 0.0)) AS plus_coverage_rate
        FROM {bq_table('cxg_event_model_matrix_v1')}
        GROUP BY split
        UNION ALL
        SELECT
          'cxg_plus' AS track,
          split,
          COUNT(DISTINCT match_id) AS match_count,
          COUNT(*) AS shot_count,
          COUNTIF(is_goal) AS goal_count,
          AVG(IF(is_goal, 1.0, 0.0)) AS goal_rate,
          COUNT(*) AS plus_shot_count,
          COUNTIF(is_goal) AS plus_goal_count,
          1.0 AS plus_coverage_rate
        FROM {bq_table('cxg_plus_360_model_matrix_v1')}
        GROUP BY split
        ORDER BY track, split
        """)
        stamped = [dict(row, run_id=self.run_id, materialized_at=datetime.now(UTC).isoformat()) for row in rows]
        self.load_json(stamped, "cxg_split_balance_v1")
        return stamped

    def feature_inventory(self) -> list[dict[str, object]]:
        return self.query_rows(f"""
        SELECT feature_family, column_name, data_type
        FROM {bq_table('cxg_feature_inventory_v1')}
        WHERE column_role = 'feature'
          AND feature_family IN ('base_identity_target','event_context','defensive_360','goalkeeper_360','line_shape_360')
          AND column_name NOT IN ('is_goal','outcome_name')
        ORDER BY feature_family, column_name
        """)

    def surface_for_family(self, family: str) -> str:
        return "cxg_plus_360_model_matrix_v1" if family.endswith("_360") else "cxg_event_model_matrix_v1"

    def materialize_split_univariate(self) -> dict[str, object]:
        features = self.feature_inventory()
        selects: list[str] = []
        for f in features:
            family, col, dtype = f["feature_family"], f["column_name"], str(f["data_type"]).upper()
            if dtype not in {"INT64", "FLOAT64", "NUMERIC", "BIGNUMERIC", "BOOL"}:
                continue
            value = f"CAST({qcol(col)} AS FLOAT64)" if dtype != "BOOL" else f"IF({qcol(col)}, 1.0, 0.0)"
            src = self.surface_for_family(family)
            track = "cxg_plus" if family.endswith("_360") else "cxg_event"
            selects.append(f"""
            SELECT
              '{self.run_id}' AS run_id,
              '{track}' AS track,
              split,
              '{family}' AS feature_family,
              '{col}' AS column_name,
              '{dtype}' AS data_type,
              COUNT(*) AS row_count,
              COUNTIF({qcol(col)} IS NULL) AS null_count,
              COUNTIF({qcol(col)} IS NOT NULL) AS non_null_count,
              SAFE_DIVIDE(COUNTIF({qcol(col)} IS NULL), COUNT(*)) AS null_pct,
              COUNTIF(is_goal) AS goal_count,
              AVG(IF(is_goal, 1.0, 0.0)) AS goal_rate,
              CORR({value}, IF(is_goal, 1.0, 0.0)) AS point_biserial_corr,
              ABS(CORR({value}, IF(is_goal, 1.0, 0.0))) AS abs_signal,
              CURRENT_TIMESTAMP() AS materialized_at
            FROM {bq_table(src)}
            GROUP BY split
            """)
        target = bq_table("cxg_split_univariate_v1")
        self.bq.query(f"CREATE OR REPLACE TABLE {target} AS " + "\nUNION ALL\n".join(selects), location=LOCATION).result()
        count = self.query_rows(f"SELECT COUNT(*) AS n FROM {target}")[0]["n"]
        return {"rows": int(count)}

    def selected_features(self) -> pd.DataFrame:
        rows = self.query_rows(f"""
        SELECT feature_family, column_name, final_analysis_decision, abs_signal
        FROM {bq_table('cxg_final_feature_selection_v1')}
        WHERE final_analysis_decision IN ('selected','shortlist_review')
        ORDER BY feature_family, final_analysis_decision, abs_signal DESC
        """)
        return pd.DataFrame(rows)

    def materialize_split_bivariate(self) -> dict[str, object]:
        decisions = self.selected_features()
        selects: list[str] = []
        for family in FAMILIES:
            cols = decisions.loc[decisions["feature_family"] == family, "column_name"].tolist()[:12]
            src = self.surface_for_family(family)
            track = "cxg_plus" if family.endswith("_360") else "cxg_event"
            min_support = 60 if track == "cxg_plus" else 150
            for left, right in combinations(cols, 2):
                selects.append(f"""
                SELECT * FROM (
                  SELECT
                    '{self.run_id}' AS run_id,
                    '{track}' AS track,
                    split,
                    '{family}' AS feature_family,
                    '{left}' AS left_column,
                    '{right}' AS right_column,
                    CAST(left_bin AS STRING) AS left_bin,
                    CAST(right_bin AS STRING) AS right_bin,
                    COUNT(*) AS row_count,
                    COUNTIF(is_goal) AS goal_count,
                    AVG(IF(is_goal, 1.0, 0.0)) AS goal_rate,
                    AVG(AVG(IF(is_goal, 1.0, 0.0))) OVER(PARTITION BY split) AS split_baseline_goal_rate,
                    SAFE_DIVIDE(AVG(IF(is_goal, 1.0, 0.0)), AVG(AVG(IF(is_goal, 1.0, 0.0))) OVER(PARTITION BY split)) AS lift_vs_split_baseline,
                    CURRENT_TIMESTAMP() AS materialized_at
                  FROM (
                    SELECT
                      split,
                      is_goal,
                      NTILE(4) OVER (PARTITION BY split ORDER BY CAST({qcol(left)} AS FLOAT64)) AS left_bin,
                      NTILE(4) OVER (PARTITION BY split ORDER BY CAST({qcol(right)} AS FLOAT64)) AS right_bin
                    FROM {bq_table(src)}
                    WHERE {qcol(left)} IS NOT NULL AND {qcol(right)} IS NOT NULL
                  )
                  GROUP BY split, left_bin, right_bin
                )
                WHERE row_count >= {min_support}
                """)
        target = bq_table("cxg_split_bivariate_v1")
        self.bq.query(f"CREATE OR REPLACE TABLE {target} AS " + "\nUNION ALL\n".join(selects), location=LOCATION).result()
        count = self.query_rows(f"SELECT COUNT(*) AS n FROM {target}")[0]["n"]
        return {"rows": int(count)}

    def model_specs(self, decisions: pd.DataFrame) -> list[dict[str, object]]:
        by_family = {fam: decisions.loc[decisions["feature_family"] == fam, "column_name"].tolist() for fam in FAMILIES}
        selected = decisions.loc[decisions["final_analysis_decision"] == "selected", "column_name"].tolist()
        specs: list[dict[str, object]] = [
            {"model_name": "cxg_event_base_xy", "track": "cxg_event", "model_stage": "true_baseline", "features": BASE_FEATURES},
            {"model_name": "cxg_event_base_plus_event_context", "track": "cxg_event", "model_stage": "candidate_model", "features": BASE_FEATURES + by_family["event_context"]},
            {"model_name": "cxg_plus_base_xy", "track": "cxg_plus", "model_stage": "true_baseline", "features": BASE_FEATURES},
            {"model_name": "cxg_plus_base_plus_selected_core", "track": "cxg_plus", "model_stage": "candidate_model", "features": BASE_FEATURES + selected},
        ]
        for size in range(1, len(FAMILIES) + 1):
            for combo in combinations(FAMILIES, size):
                if not any(f in THREE_SIXTY_FAMILIES for f in combo):
                    continue
                features = BASE_FEATURES[:]
                for family in combo:
                    features.extend(by_family[family])
                specs.append({
                    "model_name": "cxg_plus_experiment_base_plus_" + "_".join(combo),
                    "track": "cxg_plus",
                    "model_stage": "multivariate_experiment",
                    "features": list(dict.fromkeys(features)),
                })
        return specs

    def load_model_df(self, track: str, features: list[str]) -> pd.DataFrame:
        src = "cxg_plus_360_model_matrix_v1" if track == "cxg_plus" else "cxg_event_model_matrix_v1"
        cols = ["event_id", "match_id", "split", "is_goal", *features]
        rows = self.query_rows(f"SELECT {', '.join(qcol(c) for c in list(dict.fromkeys(cols)))} FROM {bq_table(src)}")
        return pd.DataFrame(rows)

    def run_split_multivariate(self) -> dict[str, object]:
        decisions = self.selected_features()
        specs = self.model_specs(decisions)
        result_rows: list[dict[str, object]] = []
        prediction_rows: list[dict[str, object]] = []
        for spec in specs:
            features = list(spec["features"])
            df = self.load_model_df(str(spec["track"]), features)
            for col in features:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            y = df["is_goal"].astype(int).to_numpy()
            x = df[features]
            train_mask = df["split"].eq("train").to_numpy()
            pipe = Pipeline([
                ("prep", ColumnTransformer([("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), features)], remainder="drop")),
                ("model", LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")),
            ])
            pipe.fit(x.loc[train_mask], y[train_mask])
            pred = pipe.predict_proba(x)[:, 1]
            for split in SPLIT_ORDER:
                mask = df["split"].eq(split).to_numpy()
                result_rows.append({
                    "run_id": self.run_id,
                    "model_name": spec["model_name"],
                    "model_stage": spec["model_stage"],
                    "track": spec["track"],
                    "split": split,
                    "feature_count": len(features),
                    "features": ",".join(features),
                    "row_count": int(mask.sum()),
                    "goal_count": int(y[mask].sum()),
                    "goal_rate": float(y[mask].mean()) if mask.sum() else None,
                    "auc": self.safe_auc(y[mask], pred[mask]),
                    "log_loss": float(log_loss(y[mask], pred[mask], labels=[0, 1])) if mask.sum() else None,
                    "brier": float(brier_score_loss(y[mask], pred[mask])) if mask.sum() else None,
                    "trained_on_split": "train",
                    "evaluated_at": datetime.now(UTC).isoformat(),
                })
            # Save predictions only for baselines/candidates to keep table bounded.
            if spec["model_stage"] in {"true_baseline", "candidate_model"}:
                for event_id, split, actual, p in zip(df["event_id"], df["split"], y, pred, strict=False):
                    prediction_rows.append({
                        "run_id": self.run_id,
                        "model_name": spec["model_name"],
                        "track": spec["track"],
                        "split": split,
                        "event_id": str(event_id),
                        "is_goal": bool(actual),
                        "predicted_goal_probability": float(p),
                        "scored_at": datetime.now(UTC).isoformat(),
                    })
        self.load_json(result_rows, "cxg_split_multivariate_results_v1")
        self.load_json(prediction_rows, "cxg_split_model_predictions_v1")
        return {"rows": len(result_rows), "prediction_rows": len(prediction_rows)}

    @staticmethod
    def safe_auc(y: np.ndarray, p: np.ndarray) -> float | None:
        if len(y) == 0 or len(set(y.tolist())) < 2:
            return None
        return float(roc_auc_score(y, p))

    def write_report(self, balance: list[dict[str, object]], uni: dict[str, object], bi: dict[str, object], multi: dict[str, object]) -> None:
        top_models = self.query_rows(f"""
        SELECT model_name, model_stage, track, split, feature_count, row_count, goal_count, auc, log_loss, brier
        FROM {bq_table('cxg_split_multivariate_results_v1')}
        WHERE split IN ('validation','test')
        ORDER BY track, split, auc DESC
        """)
        lines = [
            "# CxG / CxG+ Split-Aware Analysis Report",
            "",
            f"Run: `{self.run_id}`",
            f"Seed: `{self.seed}`",
            "",
            "## Split Balance",
            "",
            "| Track | Split | Matches | Shots | Goals | Goal rate | 360 shots | 360 goals | 360 coverage |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for r in balance:
            lines.append(f"| {r['track']} | {r['split']} | {r['match_count']} | {r['shot_count']} | {r['goal_count']} | {r['goal_rate']:.4f} | {r['plus_shot_count']} | {r['plus_goal_count']} | {r['plus_coverage_rate']:.4f} |")
        lines.extend([
            "",
            "## Output Tables",
            "",
            f"- `oam_analysis.cxg_match_splits_v1`",
            f"- `oam_analysis.cxg_event_model_matrix_v1`",
            f"- `oam_analysis.cxg_plus_360_model_matrix_v1`",
            f"- `oam_analysis.cxg_split_univariate_v1`: {uni['rows']} rows",
            f"- `oam_analysis.cxg_split_bivariate_v1`: {bi['rows']} rows",
            f"- `oam_analysis.cxg_split_multivariate_results_v1`: {multi['rows']} rows",
            f"- `oam_analysis.cxg_split_model_predictions_v1`: {multi['prediction_rows']} rows",
            "",
            "## Validation/Test Model Ranking",
            "",
            "| Track | Split | Model | Stage | Features | Rows | Goals | AUC | Log loss | Brier |",
            "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
        ])
        for r in top_models:
            auc = "" if r["auc"] is None else f"{r['auc']:.4f}"
            lines.append(f"| {r['track']} | {r['split']} | `{r['model_name']}` | {r['model_stage']} | {r['feature_count']} | {r['row_count']} | {r['goal_count']} | {auc} | {r['log_loss']:.4f} | {r['brier']:.4f} |")
        lines.extend([
            "",
            "## Method",
            "",
            "Splits are by match_id. Univariate and bivariate outputs are split-tagged. Multivariate models are trained only on train and evaluated on train/validation/test. Test metrics are generated for audit visibility in this run, but future model-choice iterations should use validation until final freeze.",
        ])
        path = self.output_dir / "cxg_split_analysis_report.md"
        path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output-dir", default="audit_outputs/cxg_analysis")
    args = parser.parse_args()
    runner = SplitAnalysisRunner(args.run_id, Path(args.output_dir), args.seed)
    print(json.dumps(runner.run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

