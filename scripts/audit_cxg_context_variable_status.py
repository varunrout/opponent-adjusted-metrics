from google.cloud import bigquery

PROJECT = "oam-varun-260819"
DATASET = "oam_analysis"
LOCATION = "europe-west2"
NAMES = ("score_diff", "game_state", "match_minute")


def main() -> None:
    client = bigquery.Client(project=PROJECT)
    for table in [
        "cxg_feature_inventory_v1",
        "cxg_univariate_target_v1",
        "cxg_feature_decision_manifest_v1",
        "cxg_final_feature_selection_v1",
        "cxg_split_univariate_v1",
        "cxg_above_baseline_model_spec_v1",
    ]:
        print(f"==== {table} ====")
        rows = list(
            client.query(
                f"""
                SELECT *
                FROM `{PROJECT}.{DATASET}.{table}`
                WHERE {'column_name' if table != 'cxg_above_baseline_model_spec_v1' else 'feature_names'} LIKE '%score_diff%'
                   OR {'column_name' if table != 'cxg_above_baseline_model_spec_v1' else 'feature_names'} LIKE '%game_state%'
                   OR {'column_name' if table != 'cxg_above_baseline_model_spec_v1' else 'feature_names'} LIKE '%match_minute%'
                LIMIT 20
                """,
                location=LOCATION,
            ).result()
        )
        for row in rows:
            print(dict(row.items()))


if __name__ == "__main__":
    main()
