from google.cloud import bigquery, storage

PROJECT = "oam-varun-260819"
DATASET = "oam_analysis"
LOCATION = "europe-west2"
BUCKET = "oam-varun-260819-artifacts"
RUN = "cxg-analysis-20260820T201934Z"


def main() -> None:
    bq = bigquery.Client(project=PROJECT)
    tables = {
        row.table_name
        for row in bq.query(
            f"SELECT table_name FROM `{PROJECT}.{DATASET}.INFORMATION_SCHEMA.TABLES`",
            location=LOCATION,
        ).result()
    }
    print("BQ_REGISTRIES")
    for table in [
        "cxg_chart_registry_v1",
        "cxg_rendered_chart_registry_v1",
        "cxg_feature_eda_chart_registry_v1",
        "cxg_split_chart_registry_v1",
        "cxg_findings_chart_registry_v1",
        "cxg_findings_detail_chart_registry_v1",
    ]:
        if table in tables:
            n = list(
                bq.query(
                    f"SELECT COUNT(*) AS n FROM `{PROJECT}.{DATASET}.{table}`",
                    location=LOCATION,
                ).result()
            )[0]["n"]
            print(f"{table}\tpresent\t{n}")
        else:
            print(f"{table}\tmissing")

    print("GCS_PREFIXES")
    bucket = storage.Client(project=PROJECT).bucket(BUCKET)
    for prefix in [
        "rendered_charts/",
        "eda_appendix/",
        "split_analysis_charts/",
        "findings_charts/",
        "findings_detail_charts/",
        "baseline_multivariate_charts/",
    ]:
        full = f"analysis/cxg/{RUN}/{prefix}"
        count = sum(1 for _ in bucket.list_blobs(prefix=full))
        print(f"{prefix}\t{count}")


if __name__ == "__main__":
    main()
