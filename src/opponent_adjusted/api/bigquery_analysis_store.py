"""BigQuery-backed implementation of the oam_analysis serving store."""

from __future__ import annotations

from google.cloud import bigquery  # type: ignore[import-untyped]

from opponent_adjusted.api.analysis_interfaces import (
    BivariateCandidateRecord,
    BivariateInteractionRecord,
    BivariateStratifiedRecord,
    FeatureCorrelationRecord,
    FeatureInventoryRecord,
    PcaComponentRecord,
    PcaLoadingRecord,
    RenderedChartRecord,
    UnivariateTargetRecord,
)
from opponent_adjusted.api.bigquery_store import PROJECT, _client

ANALYSIS_DATASET = "oam_analysis"


class BigQueryAnalysisStore:
    """Read-only AnalysisStore backed by the oam_analysis BigQuery dataset."""

    def list_features(self, *, family: str | None = None) -> list[FeatureInventoryRecord]:
        client = _client()
        conditions: list[str] = []
        parameters: list[bigquery.ScalarQueryParameter] = []

        if family is not None:
            conditions.append("feature_family = @family")
            parameters.append(bigquery.ScalarQueryParameter("family", "STRING", family))

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        query = f"""
            SELECT
                feature_family,
                source_table,
                column_name,
                data_type,
                column_role,
                is_numeric,
                is_categorical
            FROM `{PROJECT}.{ANALYSIS_DATASET}.cxg_feature_inventory_v1`
            {where_clause}
        """
        job_config = bigquery.QueryJobConfig(query_parameters=parameters)
        rows = client.query(query, job_config=job_config).result()
        return [
            FeatureInventoryRecord(
                feature_family=row["feature_family"],
                source_table=row["source_table"],
                column_name=row["column_name"],
                data_type=row["data_type"],
                column_role=row["column_role"],
                is_numeric=row["is_numeric"],
                is_categorical=row["is_categorical"],
            )
            for row in rows
        ]

    def list_correlations(self, *, family: str | None = None) -> list[FeatureCorrelationRecord]:
        client = _client()
        parameters: list[bigquery.ScalarQueryParameter] = []

        if family is not None:
            parameters.append(bigquery.ScalarQueryParameter("family", "STRING", family))
            where_clause = f"""
                WHERE feature_a IN (
                    SELECT column_name
                    FROM `{PROJECT}.{ANALYSIS_DATASET}.cxg_feature_inventory_v1`
                    WHERE feature_family = @family
                )
                OR feature_b IN (
                    SELECT column_name
                    FROM `{PROJECT}.{ANALYSIS_DATASET}.cxg_feature_inventory_v1`
                    WHERE feature_family = @family
                )
            """
        else:
            where_clause = ""

        query = f"""
            SELECT
                track,
                feature_a,
                feature_b,
                r_train,
                n_train,
                is_redundant,
                resolution,
                resolution_reason
            FROM `{PROJECT}.{ANALYSIS_DATASET}.cxg_feature_correlation_v1`
            {where_clause}
        """
        job_config = bigquery.QueryJobConfig(query_parameters=parameters)
        rows = client.query(query, job_config=job_config).result()
        return [
            FeatureCorrelationRecord(
                track=row["track"],
                feature_a=row["feature_a"],
                feature_b=row["feature_b"],
                r_train=row["r_train"],
                n_train=row["n_train"],
                is_redundant=row["is_redundant"],
                resolution=row["resolution"],
                resolution_reason=row["resolution_reason"],
            )
            for row in rows
        ]

    def list_univariate(self, *, family: str | None = None) -> list[UnivariateTargetRecord]:
        client = _client()
        conditions: list[str] = []
        parameters: list[bigquery.ScalarQueryParameter] = []

        if family is not None:
            conditions.append("feature_family = @family")
            parameters.append(bigquery.ScalarQueryParameter("family", "STRING", family))

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        query = f"""
            SELECT
                feature_family,
                column_name,
                data_type,
                row_count,
                non_null_count,
                goal_rate,
                mean_when_available,
                mean_for_goals,
                mean_for_non_goals,
                point_biserial_corr
            FROM `{PROJECT}.{ANALYSIS_DATASET}.cxg_univariate_target_v1`
            {where_clause}
        """
        job_config = bigquery.QueryJobConfig(query_parameters=parameters)
        rows = client.query(query, job_config=job_config).result()
        return [
            UnivariateTargetRecord(
                feature_family=row["feature_family"],
                column_name=row["column_name"],
                data_type=row["data_type"],
                row_count=row["row_count"],
                non_null_count=row["non_null_count"],
                goal_rate=row["goal_rate"],
                mean_when_available=row["mean_when_available"],
                mean_for_goals=row["mean_for_goals"],
                mean_for_non_goals=row["mean_for_non_goals"],
                point_biserial_corr=row["point_biserial_corr"],
            )
            for row in rows
        ]

    def list_bivariate_candidates(self) -> list[BivariateCandidateRecord]:
        client = _client()
        query = f"""
            SELECT
                track,
                feature_family,
                column_name,
                data_type,
                qualification_reason
            FROM `{PROJECT}.{ANALYSIS_DATASET}.cxg_bivariate_candidate_pool_v1`
        """
        rows = client.query(query).result()
        return [
            BivariateCandidateRecord(
                track=row["track"],
                feature_family=row["feature_family"],
                column_name=row["column_name"],
                data_type=row["data_type"],
                qualification_reason=row["qualification_reason"],
            )
            for row in rows
        ]

    def list_bivariate_interactions(self) -> list[BivariateInteractionRecord]:
        client = _client()
        query = f"""
            SELECT
                track,
                tier,
                feature_a,
                feature_b,
                n_train,
                interaction_coef,
                interaction_se,
                interaction_p_raw,
                interaction_p_fdr,
                lr_stat,
                main_effect_a_coef,
                main_effect_b_coef,
                validated_on_val_split,
                fit_status
            FROM `{PROJECT}.{ANALYSIS_DATASET}.cxg_bivariate_interaction_v1`
        """
        rows = client.query(query).result()
        return [
            BivariateInteractionRecord(
                track=row["track"],
                tier=row["tier"],
                feature_a=row["feature_a"],
                feature_b=row["feature_b"],
                n_train=row["n_train"],
                interaction_coef=row["interaction_coef"],
                interaction_se=row["interaction_se"],
                interaction_p_raw=row["interaction_p_raw"],
                interaction_p_fdr=row["interaction_p_fdr"],
                lr_stat=row["lr_stat"],
                main_effect_a_coef=row["main_effect_a_coef"],
                main_effect_b_coef=row["main_effect_b_coef"],
                validated_on_val_split=row["validated_on_val_split"],
                fit_status=row["fit_status"],
            )
            for row in rows
        ]

    def list_bivariate_stratified(self) -> list[BivariateStratifiedRecord]:
        client = _client()
        query = f"""
            SELECT
                track,
                tier,
                feature_a,
                feature_b,
                stratum_a,
                stratum_b,
                n,
                goal_count,
                goal_rate
            FROM `{PROJECT}.{ANALYSIS_DATASET}.cxg_bivariate_stratified_v1`
        """
        rows = client.query(query).result()
        return [
            BivariateStratifiedRecord(
                track=row["track"],
                tier=row["tier"],
                feature_a=row["feature_a"],
                feature_b=row["feature_b"],
                stratum_a=row["stratum_a"],
                stratum_b=row["stratum_b"],
                n=row["n"],
                goal_count=row["goal_count"],
                goal_rate=row["goal_rate"],
            )
            for row in rows
        ]

    def list_pca_components(self) -> list[PcaComponentRecord]:
        client = _client()
        query = f"""
            SELECT
                track,
                component_number,
                explained_variance_ratio,
                cumulative_variance_ratio
            FROM `{PROJECT}.{ANALYSIS_DATASET}.cxg_pca_components_v1`
        """
        rows = client.query(query).result()
        return [
            PcaComponentRecord(
                track=row["track"],
                component_number=row["component_number"],
                explained_variance_ratio=row["explained_variance_ratio"],
                cumulative_variance_ratio=row["cumulative_variance_ratio"],
            )
            for row in rows
        ]

    def list_pca_loadings(self) -> list[PcaLoadingRecord]:
        client = _client()
        query = f"""
            SELECT
                track,
                component_number,
                feature_name,
                loading
            FROM `{PROJECT}.{ANALYSIS_DATASET}.cxg_pca_loadings_v1`
        """
        rows = client.query(query).result()
        return [
            PcaLoadingRecord(
                track=row["track"],
                component_number=row["component_number"],
                feature_name=row["feature_name"],
                loading=row["loading"],
            )
            for row in rows
        ]

    def get_latest_chart_run_id(self) -> str | None:
        client = _client()
        query = f"""
            SELECT run_id
            FROM `{PROJECT}.{ANALYSIS_DATASET}.cxg_rendered_chart_registry_v1`
            ORDER BY rendered_at DESC
            LIMIT 1
        """
        rows = list(client.query(query).result())
        if not rows:
            return None
        return rows[0]["run_id"]

    def list_charts(self, run_id: str) -> list[RenderedChartRecord]:
        client = _client()
        query = f"""
            SELECT
                run_id,
                chart_name,
                html_uri,
                png_uri,
                rendered_at
            FROM `{PROJECT}.{ANALYSIS_DATASET}.cxg_rendered_chart_registry_v1`
            WHERE run_id = @run_id
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("run_id", "STRING", run_id)]
        )
        rows = client.query(query, job_config=job_config).result()
        return [
            RenderedChartRecord(
                run_id=row["run_id"],
                chart_name=row["chart_name"],
                html_uri=row["html_uri"],
                png_uri=row["png_uri"],
                rendered_at=str(row["rendered_at"]) if row["rendered_at"] is not None else None,
            )
            for row in rows
        ]
