"""Fixtures for API router tests."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from opponent_adjusted.api.analysis_interfaces import (
    BivariateCandidateRecord,
    BivariateInteractionRecord,
    BivariateStratifiedRecord,
    CxgCoefficientRecord,
    CxgModelResultRecord,
    FeatureCorrelationRecord,
    FeatureInventoryRecord,
    PcaComponentRecord,
    PcaLoadingRecord,
    RenderedChartRecord,
    UnivariateTargetRecord,
)
from opponent_adjusted.api.dependencies import get_analysis_store, get_store
from opponent_adjusted.api.interfaces import (
    CompetitionRecord,
    LineupPlayerRecord,
    MatchRecord,
    PlayerSeasonRecord,
    ShotRecord,
    TeamSeasonRecord,
)
from opponent_adjusted.api.main import app


class FakeServingStore:
    """In-memory ServingStore fake for tests."""

    def __init__(
        self,
        competitions: list[CompetitionRecord],
        matches: list[MatchRecord],
        lineups: list[LineupPlayerRecord] | None = None,
        shots: list[ShotRecord] | None = None,
    ) -> None:
        self._competitions = competitions
        self._matches = matches
        self._lineups = lineups or []
        self._shots = shots or []

    def list_competitions(self) -> list[CompetitionRecord]:
        return list(self._competitions)

    def list_matches(
        self,
        *,
        competition_id: int | None = None,
        season_id: int | None = None,
        team_id: int | None = None,
    ) -> list[MatchRecord]:
        matches = self._matches
        if competition_id is not None:
            matches = [m for m in matches if m.competition_id == competition_id]
        if season_id is not None:
            matches = [m for m in matches if m.season_id == season_id]
        if team_id is not None:
            matches = [m for m in matches if m.home_team_id == team_id or m.away_team_id == team_id]
        return list(matches)

    def get_match(self, match_id: int) -> MatchRecord | None:
        for match in self._matches:
            if match.match_id == match_id:
                return match
        return None

    def list_lineups(self, match_id: int) -> list[LineupPlayerRecord]:
        return [row for row in self._lineups if row.match_id == match_id]

    def list_shots(self, match_id: int) -> list[ShotRecord]:
        return [row for row in self._shots if row.match_id == match_id]

    def _team_name(self, match: MatchRecord, team_id: int | None) -> str | None:
        if team_id == match.home_team_id:
            return match.home_team_name
        if team_id == match.away_team_id:
            return match.away_team_name
        return None

    def _filtered_shots(
        self,
        *,
        competition_id: int | None,
        season_id: int | None,
    ) -> list[ShotRecord]:
        matches_by_id = {m.match_id: m for m in self._matches}
        shots = self._shots
        if competition_id is not None:
            shots = [s for s in shots if matches_by_id[s.match_id].competition_id == competition_id]
        if season_id is not None:
            shots = [s for s in shots if matches_by_id[s.match_id].season_id == season_id]
        return shots

    def list_player_seasons(
        self,
        *,
        competition_id: int | None = None,
        season_id: int | None = None,
    ) -> list[PlayerSeasonRecord]:
        matches_by_id = {m.match_id: m for m in self._matches}
        shots = self._filtered_shots(competition_id=competition_id, season_id=season_id)

        aggregates: dict[int, dict] = {}
        for shot in shots:
            if shot.player_id is None:
                continue
            agg = aggregates.setdefault(
                shot.player_id,
                {
                    "player_name": shot.player_name,
                    "team_id": shot.team_id,
                    "team_name": self._team_name(matches_by_id[shot.match_id], shot.team_id),
                    "shots": 0,
                    "goals": 0,
                    "total_xg": 0.0,
                },
            )
            if shot.player_name is not None:
                agg["player_name"] = shot.player_name
            agg["shots"] += 1
            if shot.is_goal:
                agg["goals"] += 1
            agg["total_xg"] += shot.statsbomb_xg or 0.0

        records = [
            PlayerSeasonRecord(
                player_id=player_id,
                player_name=values["player_name"],
                team_id=values["team_id"],
                team_name=values["team_name"],
                shots=values["shots"],
                goals=values["goals"],
                total_xg=values["total_xg"],
            )
            for player_id, values in aggregates.items()
        ]
        records.sort(key=lambda r: r.total_xg, reverse=True)
        return records

    def list_player_shots(
        self,
        player_id: int,
        *,
        competition_id: int | None = None,
        season_id: int | None = None,
    ) -> list[ShotRecord]:
        shots = self._filtered_shots(competition_id=competition_id, season_id=season_id)
        return [s for s in shots if s.player_id == player_id]

    def list_team_seasons(
        self,
        *,
        competition_id: int | None = None,
        season_id: int | None = None,
    ) -> list[TeamSeasonRecord]:
        matches_by_id = {m.match_id: m for m in self._matches}
        shots = self._filtered_shots(competition_id=competition_id, season_id=season_id)

        aggregates: dict[int, dict] = {}
        for shot in shots:
            if shot.team_id is None:
                continue
            agg = aggregates.setdefault(
                shot.team_id,
                {
                    "team_name": self._team_name(matches_by_id[shot.match_id], shot.team_id),
                    "shots": 0,
                    "goals": 0,
                    "total_xg": 0.0,
                },
            )
            agg["shots"] += 1
            if shot.is_goal:
                agg["goals"] += 1
            agg["total_xg"] += shot.statsbomb_xg or 0.0

        records = [
            TeamSeasonRecord(
                team_id=team_id,
                team_name=values["team_name"],
                shots=values["shots"],
                goals=values["goals"],
                total_xg=values["total_xg"],
            )
            for team_id, values in aggregates.items()
        ]
        records.sort(key=lambda r: r.total_xg, reverse=True)
        return records

    def list_team_shots(
        self,
        team_id: int,
        *,
        competition_id: int | None = None,
        season_id: int | None = None,
    ) -> list[ShotRecord]:
        shots = self._filtered_shots(competition_id=competition_id, season_id=season_id)
        return [s for s in shots if s.team_id == team_id]


FAKE_COMPETITIONS = [
    CompetitionRecord(
        competition_id=43,
        season_id=3,
        competition_name="FIFA World Cup",
        competition_gender="male",
        country_name="International",
        season_name="2018",
        match_updated="2021-01-01T00:00:00",
        match_available="2021-01-01T00:00:00",
        match_updated_360=None,
        match_available_360=None,
    ),
    CompetitionRecord(
        competition_id=11,
        season_id=1,
        competition_name="La Liga",
        competition_gender="male",
        country_name="Spain",
        season_name="2020/2021",
        match_updated="2021-02-01T00:00:00",
        match_available="2021-02-01T00:00:00",
        match_updated_360="2021-02-02T00:00:00",
        match_available_360="2021-02-02T00:00:00",
    ),
    CompetitionRecord(
        competition_id=72,
        season_id=30,
        competition_name="Women's World Cup",
        competition_gender="female",
        country_name="International",
        season_name="2023",
        match_updated="2023-08-01T00:00:00",
        match_available="2023-08-01T00:00:00",
        match_updated_360=None,
        match_available_360=None,
    ),
]

FAKE_MATCHES = [
    MatchRecord(
        match_id=7,
        competition_id=43,
        season_id=3,
        match_date="2018-07-15",
        kick_off="17:00:00.000",
        home_team_id=771,
        home_team_name="France",
        away_team_id=772,
        away_team_name="Croatia",
        home_score=4,
        away_score=2,
        competition_stage="Final",
        stadium="Luzhniki Stadium",
        referee="Nestor Pitana",
        match_status="available",
        match_status_360=None,
        last_updated="2021-01-01T00:00:00",
        last_updated_360=None,
    ),
    MatchRecord(
        match_id=8,
        competition_id=11,
        season_id=1,
        match_date="2021-01-10",
        kick_off="20:00:00.000",
        home_team_id=217,
        home_team_name="Barcelona",
        away_team_id=220,
        away_team_name="Real Madrid",
        home_score=1,
        away_score=3,
        competition_stage="Regular Season",
        stadium="Camp Nou",
        referee="Antonio Mateu Lahoz",
        match_status="available",
        match_status_360="available",
        last_updated="2021-02-01T00:00:00",
        last_updated_360="2021-02-02T00:00:00",
    ),
    MatchRecord(
        match_id=9,
        competition_id=72,
        season_id=30,
        match_date="2023-08-20",
        kick_off="12:00:00.000",
        home_team_id=901,
        home_team_name="Spain",
        away_team_id=902,
        away_team_name="England",
        home_score=1,
        away_score=0,
        competition_stage="Final",
        stadium="Stadium Australia",
        referee="Tori Penso",
        match_status="available",
        match_status_360=None,
        last_updated="2023-08-21T00:00:00",
        last_updated_360=None,
    ),
]


FAKE_LINEUPS = [
    LineupPlayerRecord(
        match_id=7,
        team_id=771,
        team_name="France",
        formation=4231,
        player_id=3009,
        player_name="Hugo Lloris",
        position_name="Goalkeeper",
        jersey_number=1,
    ),
    LineupPlayerRecord(
        match_id=7,
        team_id=771,
        team_name="France",
        formation=4231,
        player_id=3010,
        player_name="Kylian Mbappé",
        position_name="Right Wing",
        jersey_number=10,
    ),
    LineupPlayerRecord(
        match_id=7,
        team_id=772,
        team_name="Croatia",
        formation=4141,
        player_id=3011,
        player_name="Luka Modrić",
        position_name="Center Attacking Midfield",
        jersey_number=10,
    ),
]

FAKE_SHOTS = [
    ShotRecord(
        event_id="shot-1",
        match_id=7,
        team_id=771,
        player_id=3010,
        player_name="Kylian Mbappé",
        minute=38,
        period=1,
        location_x=110.0,
        location_y=40.0,
        end_x=120.0,
        end_y=39.0,
        statsbomb_xg=0.32,
        outcome_name="Goal",
        body_part_name="Right Foot",
        is_goal=True,
    ),
    ShotRecord(
        event_id="shot-2",
        match_id=7,
        team_id=772,
        player_id=3011,
        player_name="Luka Modrić",
        minute=27,
        period=1,
        location_x=105.0,
        location_y=38.0,
        end_x=118.0,
        end_y=41.0,
        statsbomb_xg=0.11,
        outcome_name="Saved",
        body_part_name="Left Foot",
        is_goal=False,
    ),
    ShotRecord(
        event_id="shot-3",
        match_id=7,
        team_id=771,
        player_id=3009,
        player_name=None,
        minute=52,
        period=2,
        location_x=100.0,
        location_y=42.0,
        end_x=None,
        end_y=None,
        statsbomb_xg=0.05,
        outcome_name="Off T",
        body_part_name="Head",
        is_goal=False,
    ),
    ShotRecord(
        event_id="shot-4",
        match_id=7,
        team_id=771,
        player_id=3010,
        player_name="Kylian Mbappé",
        minute=61,
        period=2,
        location_x=108.0,
        location_y=44.0,
        end_x=119.0,
        end_y=40.0,
        statsbomb_xg=0.08,
        outcome_name="Saved",
        body_part_name="Right Foot",
        is_goal=False,
    ),
    ShotRecord(
        event_id="shot-5",
        match_id=8,
        team_id=217,
        player_id=4001,
        player_name="Lionel Messi",
        minute=15,
        period=1,
        location_x=112.0,
        location_y=39.0,
        end_x=120.0,
        end_y=40.0,
        statsbomb_xg=0.50,
        outcome_name="Goal",
        body_part_name="Left Foot",
        is_goal=True,
    ),
    ShotRecord(
        event_id="shot-6",
        match_id=9,
        team_id=901,
        player_id=5001,
        player_name="Jenni Hermoso",
        minute=29,
        period=1,
        location_x=109.0,
        location_y=41.0,
        end_x=119.0,
        end_y=39.0,
        statsbomb_xg=0.45,
        outcome_name="Goal",
        body_part_name="Right Foot",
        is_goal=True,
    ),
]


class FakeAnalysisStore:
    """In-memory AnalysisStore fake for tests."""

    def __init__(
        self,
        features: list[FeatureInventoryRecord],
        correlations: list[FeatureCorrelationRecord],
        univariate: list[UnivariateTargetRecord],
        bivariate_candidates: list[BivariateCandidateRecord],
        bivariate_interactions: list[BivariateInteractionRecord],
        bivariate_stratified: list[BivariateStratifiedRecord],
        pca_components: list[PcaComponentRecord],
        pca_loadings: list[PcaLoadingRecord],
        charts: list[RenderedChartRecord],
        cxg_model_results: list[CxgModelResultRecord] | None = None,
        cxg_coefficients: list[CxgCoefficientRecord] | None = None,
    ) -> None:
        self._features = features
        self._correlations = correlations
        self._univariate = univariate
        self._bivariate_candidates = bivariate_candidates
        self._bivariate_interactions = bivariate_interactions
        self._bivariate_stratified = bivariate_stratified
        self._pca_components = pca_components
        self._pca_loadings = pca_loadings
        self._charts = charts
        self._cxg_model_results = cxg_model_results or []
        self._cxg_coefficients = cxg_coefficients or []

    def list_features(self, *, family: str | None = None) -> list[FeatureInventoryRecord]:
        if family is None:
            return list(self._features)
        return [f for f in self._features if f.feature_family == family]

    def list_correlations(self, *, family: str | None = None) -> list[FeatureCorrelationRecord]:
        if family is None:
            return list(self._correlations)
        family_columns = {f.column_name for f in self._features if f.feature_family == family}
        return [
            c
            for c in self._correlations
            if c.feature_a in family_columns or c.feature_b in family_columns
        ]

    def list_univariate(self, *, family: str | None = None) -> list[UnivariateTargetRecord]:
        if family is None:
            return list(self._univariate)
        return [u for u in self._univariate if u.feature_family == family]

    def list_bivariate_candidates(self) -> list[BivariateCandidateRecord]:
        return list(self._bivariate_candidates)

    def list_bivariate_interactions(self) -> list[BivariateInteractionRecord]:
        return list(self._bivariate_interactions)

    def list_bivariate_stratified(self) -> list[BivariateStratifiedRecord]:
        return list(self._bivariate_stratified)

    def list_pca_components(self) -> list[PcaComponentRecord]:
        return list(self._pca_components)

    def list_pca_loadings(self) -> list[PcaLoadingRecord]:
        return list(self._pca_loadings)

    def get_latest_chart_run_id(self) -> str | None:
        if not self._charts:
            return None
        return max(self._charts, key=lambda c: c.rendered_at or "").run_id

    def list_charts(self, run_id: str) -> list[RenderedChartRecord]:
        return [c for c in self._charts if c.run_id == run_id]

    def list_cxg_model_results(self) -> list[CxgModelResultRecord]:
        return list(self._cxg_model_results)

    def list_cxg_coefficients(self, model_key: str) -> list[CxgCoefficientRecord]:
        known_keys = {r.model_key for r in self._cxg_model_results} | {
            c.model_key for c in self._cxg_coefficients
        }
        if model_key not in known_keys:
            raise ValueError(f"Unknown model_key: {model_key!r}")
        return [c for c in self._cxg_coefficients if c.model_key == model_key]


FAKE_FEATURES = [
    FeatureInventoryRecord(
        feature_family="shot_geometry",
        source_table="cxg_features_v1",
        column_name="shot_distance",
        data_type="FLOAT64",
        column_role="feature",
        is_numeric=True,
        is_categorical=False,
    ),
    FeatureInventoryRecord(
        feature_family="shot_geometry",
        source_table="cxg_features_v1",
        column_name="shot_angle",
        data_type="FLOAT64",
        column_role="feature",
        is_numeric=True,
        is_categorical=False,
    ),
    FeatureInventoryRecord(
        feature_family="opponent_adjusted",
        source_table="cxg_features_v1",
        column_name="nearest_defender_odi",
        data_type="FLOAT64",
        column_role="feature",
        is_numeric=True,
        is_categorical=False,
    ),
    FeatureInventoryRecord(
        feature_family="opponent_adjusted",
        source_table="cxg_features_v1",
        column_name="pressure_odi",
        data_type="FLOAT64",
        column_role="feature",
        is_numeric=True,
        is_categorical=False,
    ),
]

FAKE_CORRELATIONS = [
    FeatureCorrelationRecord(
        track="main",
        feature_a="shot_distance",
        feature_b="shot_angle",
        r_train=0.42,
        n_train=1000,
        is_redundant=False,
        resolution="keep_both",
        resolution_reason=None,
    ),
    FeatureCorrelationRecord(
        track="main",
        feature_a="nearest_defender_odi",
        feature_b="pressure_odi",
        r_train=0.91,
        n_train=1000,
        is_redundant=True,
        resolution="drop_b",
        resolution_reason="highly collinear",
    ),
    FeatureCorrelationRecord(
        track="main",
        feature_a="shot_distance",
        feature_b="nearest_defender_odi",
        r_train=0.15,
        n_train=1000,
        is_redundant=False,
        resolution="keep_both",
        resolution_reason=None,
    ),
    FeatureCorrelationRecord(
        track="main",
        feature_a="unrelated_column_a",
        feature_b="unrelated_column_b",
        r_train=0.05,
        n_train=1000,
        is_redundant=False,
        resolution="keep_both",
        resolution_reason=None,
    ),
]

FAKE_UNIVARIATE = [
    UnivariateTargetRecord(
        feature_family="shot_geometry",
        column_name="shot_distance",
        data_type="FLOAT64",
        row_count=1000,
        non_null_count=1000,
        goal_rate=0.1,
        mean_when_available=15.2,
        mean_for_goals=8.4,
        mean_for_non_goals=16.1,
        point_biserial_corr=-0.22,
    ),
    UnivariateTargetRecord(
        feature_family="shot_geometry",
        column_name="shot_angle",
        data_type="FLOAT64",
        row_count=1000,
        non_null_count=980,
        goal_rate=0.1,
        mean_when_available=0.6,
        mean_for_goals=0.8,
        mean_for_non_goals=0.55,
        point_biserial_corr=0.31,
    ),
    UnivariateTargetRecord(
        feature_family="opponent_adjusted",
        column_name="nearest_defender_odi",
        data_type="FLOAT64",
        row_count=1000,
        non_null_count=950,
        goal_rate=0.1,
        mean_when_available=2.1,
        mean_for_goals=3.0,
        mean_for_non_goals=2.0,
        point_biserial_corr=0.18,
    ),
]

FAKE_BIVARIATE_CANDIDATES = [
    BivariateCandidateRecord(
        track="main",
        feature_family="shot_geometry",
        column_name="shot_distance",
        data_type="FLOAT64",
        qualification_reason="high univariate signal",
    ),
    BivariateCandidateRecord(
        track="main",
        feature_family="opponent_adjusted",
        column_name="nearest_defender_odi",
        data_type="FLOAT64",
        qualification_reason="passed redundancy check",
    ),
]

FAKE_BIVARIATE_INTERACTIONS = [
    BivariateInteractionRecord(
        track="main",
        tier=1,
        feature_a="shot_distance",
        feature_b="nearest_defender_odi",
        n_train=1000,
        interaction_coef=0.12,
        interaction_se=0.03,
        interaction_p_raw=0.001,
        interaction_p_fdr=0.01,
        lr_stat=10.5,
        main_effect_a_coef=-0.5,
        main_effect_b_coef=0.3,
        validated_on_val_split=True,
        fit_status="converged",
    ),
]

FAKE_BIVARIATE_STRATIFIED = [
    BivariateStratifiedRecord(
        track="main",
        tier=1,
        feature_a="shot_distance",
        feature_b="nearest_defender_odi",
        stratum_a="near",
        stratum_b="pressured",
        n=200,
        goal_count=40,
        goal_rate=0.2,
    ),
]

FAKE_PCA_COMPONENTS = [
    PcaComponentRecord(
        track="main",
        component_number=1,
        explained_variance_ratio=0.35,
        cumulative_variance_ratio=0.35,
    ),
    PcaComponentRecord(
        track="main",
        component_number=2,
        explained_variance_ratio=0.2,
        cumulative_variance_ratio=0.55,
    ),
]

FAKE_PCA_LOADINGS = [
    PcaLoadingRecord(
        track="main",
        component_number=1,
        feature_name="shot_distance",
        loading=0.8,
    ),
    PcaLoadingRecord(
        track="main",
        component_number=1,
        feature_name="shot_angle",
        loading=-0.4,
    ),
]

FAKE_CHARTS = [
    RenderedChartRecord(
        run_id="cxg-analysis-20260810T010000Z",
        chart_name="feature_correlation_heatmap",
        html_uri="gs://bucket/old/heatmap.html",
        png_uri="gs://bucket/old/heatmap.png",
        rendered_at="2026-08-10T01:00:00Z",
    ),
    RenderedChartRecord(
        run_id="cxg-analysis-20260810T010000Z",
        chart_name="pca_scree_plot",
        html_uri="gs://bucket/old/scree.html",
        png_uri="gs://bucket/old/scree.png",
        rendered_at="2026-08-10T01:00:00Z",
    ),
    RenderedChartRecord(
        run_id="cxg-analysis-20260822T014705Z",
        chart_name="feature_correlation_heatmap",
        html_uri="gs://bucket/new/heatmap.html",
        png_uri="gs://bucket/new/heatmap.png",
        rendered_at="2026-08-22T01:47:05Z",
    ),
    RenderedChartRecord(
        run_id="cxg-analysis-20260822T014705Z",
        chart_name="pca_scree_plot",
        html_uri="gs://bucket/new/scree.html",
        png_uri="gs://bucket/new/scree.png",
        rendered_at="2026-08-22T01:47:05Z",
    ),
]

# Mirrors the real oam_ml shape (verified against live BigQuery): each
# model_key's _metrics table can hold multiple `model` values (the CxG
# version itself plus statsbomb_xg/dumb_baseline comparators) at the same
# track/split. is_current is True only for event_v3/plus_v3.
FAKE_CXG_MODEL_RESULTS = [
    CxgModelResultRecord(
        model_key="event_v3",
        track="cxg_event",
        split="test",
        model="v3",
        n=2427,
        log_loss=0.3003,
        brier_score=0.0852,
        roc_auc=0.7148,
        is_frozen=True,
        is_current=True,
    ),
    CxgModelResultRecord(
        model_key="event_v3",
        track="cxg_event",
        split="test",
        model="statsbomb_xg",
        n=2427,
        log_loss=0.2597,
        brier_score=0.0718,
        roc_auc=0.7972,
        is_frozen=True,
        is_current=True,
    ),
    CxgModelResultRecord(
        model_key="plus_v3",
        track="cxg_plus",
        split="test",
        model="v3",
        n=590,
        log_loss=0.2555,
        brier_score=0.0713,
        roc_auc=0.8313,
        is_frozen=True,
        is_current=True,
    ),
    CxgModelResultRecord(
        model_key="plus_v3",
        track="cxg_plus",
        split="test",
        model="statsbomb_xg",
        n=590,
        log_loss=0.2430,
        brier_score=0.0665,
        roc_auc=0.8476,
        is_frozen=True,
        is_current=True,
    ),
    CxgModelResultRecord(
        model_key="baseline_v1",
        track="cxg_event",
        split="test",
        model="v1",
        n=2427,
        log_loss=0.3058,
        brier_score=0.0872,
        roc_auc=0.6939,
        is_frozen=True,
        is_current=False,
    ),
    CxgModelResultRecord(
        model_key="plus_v2",
        track="cxg_plus",
        split="test",
        model="v2",
        n=590,
        log_loss=0.2566,
        brier_score=0.0716,
        roc_auc=0.8302,
        is_frozen=True,
        is_current=False,
    ),
]

FAKE_CXG_COEFFICIENTS = [
    CxgCoefficientRecord(
        model_key="event_v3",
        track="cxg_event",
        feature="const",
        coefficient=-2.4634,
        std_error=None,
        p_value=None,
    ),
    CxgCoefficientRecord(
        model_key="event_v3",
        track="cxg_event",
        feature="shot_x_sb",
        coefficient=0.7701,
        std_error=None,
        p_value=None,
    ),
    CxgCoefficientRecord(
        model_key="plus_v3",
        track="cxg_plus",
        feature="const",
        coefficient=-2.1,
        std_error=None,
        p_value=None,
    ),
    CxgCoefficientRecord(
        model_key="plus_v3",
        track="cxg_plus",
        feature="nearest_defender_zone_displacement",
        coefficient=0.15,
        std_error=0.05,
        p_value=0.01,
    ),
]


@pytest.fixture
def client() -> TestClient:
    app.dependency_overrides[get_store] = lambda: FakeServingStore(
        competitions=FAKE_COMPETITIONS,
        matches=FAKE_MATCHES,
        lineups=FAKE_LINEUPS,
        shots=FAKE_SHOTS,
    )
    app.dependency_overrides[get_analysis_store] = lambda: FakeAnalysisStore(
        features=FAKE_FEATURES,
        correlations=FAKE_CORRELATIONS,
        univariate=FAKE_UNIVARIATE,
        bivariate_candidates=FAKE_BIVARIATE_CANDIDATES,
        bivariate_interactions=FAKE_BIVARIATE_INTERACTIONS,
        bivariate_stratified=FAKE_BIVARIATE_STRATIFIED,
        pca_components=FAKE_PCA_COMPONENTS,
        pca_loadings=FAKE_PCA_LOADINGS,
        charts=FAKE_CHARTS,
        cxg_model_results=FAKE_CXG_MODEL_RESULTS,
        cxg_coefficients=FAKE_CXG_COEFFICIENTS,
    )
    test_client = TestClient(app)
    yield test_client
    app.dependency_overrides.clear()
