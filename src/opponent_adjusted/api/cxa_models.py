"""CxA combined-scorer read path: model comparison (P_create vs. P_convert test
metrics + coverage stats) for the public Models page, and per-pass coverage lookups
for a future per-pass display. Backed by `oam_ml.cxa_{track}_test_v1_metrics` /
`oam_ml.cxconvert_{track}_test_v1_metrics` / `oam_serving.cxa_{track}_combined_v1`.

Mirrors `cxg_coverage.py`'s structure and "no placeholder" discipline exactly, per
docs/analysis/cxa_combined_scorer_design_v1.md's own API sketch (section 4).

**Decision 4 (binding, from the design doc / the dashboard-build task): every query
here filters to `split='test'`** -- the underlying `oam_serving` tables carry all
three splits by design (train+validation+test), matching CxG's own real precedent
(`oam_ml.cxg_event_v3_predictions` also carries every split, with `test`-only
filtering applied at this exact layer, not baked into the table) -- but nothing
public-facing in this module ever reads a `train` or `validation` row.

**Decision 5 (binding): this module never computes or exposes `cxa_combined_score`'s
own log_loss/AUC as if it were "the CxA model's performance."** P_create's and
P_convert's own already-reported test metrics are read separately from their own
`*_test_v1_metrics` tables and returned as clearly stage-labelled rows, never unioned
into one fake combined number. `docs/analysis/cxa_combined_scorer_v1.md` section 3
already established that the product is demonstrably a *worse* predictor of `y_goal`
than `p_convert` alone -- surfacing it as a headline metric here would misrepresent
that finding.

**Decision 1 (binding): the ~2% chance-creating coverage caveat is reused verbatim**
from the design doc's section 1c wording wherever the combined score is described,
not rephrased looser.
"""

from __future__ import annotations

import threading
from typing import Protocol

from cachetools import TTLCache, cached
from cachetools.keys import hashkey
from pydantic import BaseModel, ConfigDict

from opponent_adjusted.api.bigquery_store import CACHE_TTL_SECONDS, PROJECT, _client

ML_DATASET = "oam_ml"
SERVING_DATASET = "oam_serving"

# Real values, matching the *_test_v1_metrics / *_combined_v1 tables' own track
# naming exactly -- NOT "cxa_event"/"cxa_plus" (that prefix belongs to the table
# *names*, not the `track` column's own values).
TRACKS = ("event", "plus")

COVERAGE_SPLIT = "test"

# Exact wording from docs/analysis/cxa_combined_scorer_design_v1.md section 1c,
# reused verbatim per decision 1 -- do not rephrase.
COMBINED_SCORE_CAVEAT = (
    "Combined CxA (create x convert) -- chance-creating passes only, ~2% of all "
    "passes; undefined, not zero, elsewhere."
)

_FROZEN_MODEL_NAMES = {"frozen_tree", "frozen_candidate"}

# Module-level cache + lock, same style as cxg_coverage.py's own
# _coverage_cache/_coverage_lock -- created once at import time, shared across
# every BigQueryCxaModelStore instance.
_coverage_cache: TTLCache = TTLCache(maxsize=8, ttl=CACHE_TTL_SECONDS)
_coverage_lock = threading.Lock()


def _track_cache_key(self, track: str) -> tuple:  # noqa: ANN001
    return hashkey(track)


class CxaStageMetric(BaseModel):
    """One row from a P_create or P_convert `oam_ml.*_test_v1_metrics` table."""

    model_config = ConfigDict(from_attributes=True)

    stage: str  # "p_create" | "p_convert"
    model: str  # "dumb_baseline" | "v1" | "frozen_tree" | "frozen_candidate"
    split: str
    n: int
    log_loss: float | None
    brier_score: float | None
    roc_auc: float | None
    is_frozen: bool  # True only for the row that is this stage's actual frozen model


class CxaCoverage(BaseModel):
    """Coverage stats for one track, read from `oam_serving.cxa_{track}_combined_v1`
    WHERE split='test'."""

    model_config = ConfigDict(from_attributes=True)

    split: str
    population_n: int
    chance_creating_n: int
    coverage_pct: float


class CxaModelSummary(BaseModel):
    """One track's full comparison payload for the public Models page."""

    model_config = ConfigDict(from_attributes=True)

    track: str  # "event" | "plus"
    stage_metrics: list[CxaStageMetric]
    coverage: CxaCoverage
    combined_score_caveat: str


class CxaCoverageValues(BaseModel):
    """One pass's covered values. `p_convert_predicted_prob`/`cxa_combined_score`
    are None (never a placeholder like 0) for a pass that never created a chance."""

    p_create_predicted_prob: float | None
    p_convert_predicted_prob: float | None
    cxa_combined_score: float | None


class CxaCoverageResponse(BaseModel):
    """API response shape for a per-pass CxA coverage lookup."""

    model_config = ConfigDict(from_attributes=True)

    track: str
    values: dict[str, CxaCoverageValues]


class CxaModelStore(Protocol):
    """Read-only contract for CxA model-comparison and coverage lookups."""

    def list_model_summaries(self) -> list[CxaModelSummary]:
        """Return one CxaModelSummary per track (event, plus)."""

    def get_cxa_for_passes(self, pass_event_ids: list[str], *, track: str) -> dict[str, CxaCoverageValues]:
        """Return {pass_event_id: values} for whichever of the given ids fall inside
        this track's test-split combined table. ids with no row at all are simply
        absent from the result -- never a placeholder, same discipline as
        CxgCoverageStore.get_cxg_for_events."""


class BigQueryCxaModelStore:
    """CxaModelStore backed by `oam_ml`'s test-metrics tables and `oam_serving`'s
    combined tables."""

    def list_model_summaries(self) -> list[CxaModelSummary]:
        client = _client()
        summaries: list[CxaModelSummary] = []
        for track in TRACKS:
            stage_rows: list[CxaStageMetric] = []
            for stage, table in (
                ("p_create", f"cxa_{track}_test_v1_metrics"),
                ("p_convert", f"cxconvert_{track}_test_v1_metrics"),
            ):
                query = f"""
                    SELECT model, split, n, log_loss, brier_score, roc_auc
                    FROM `{PROJECT}.{ML_DATASET}.{table}`
                    WHERE split = '{COVERAGE_SPLIT}'
                    ORDER BY model
                """
                for row in client.query(query).result():
                    stage_rows.append(
                        CxaStageMetric(
                            stage=stage,
                            model=row["model"],
                            split=row["split"],
                            n=row["n"],
                            log_loss=row["log_loss"],
                            brier_score=row["brier_score"],
                            roc_auc=row["roc_auc"],
                            is_frozen=row["model"] in _FROZEN_MODEL_NAMES,
                        )
                    )

            coverage_query = f"""
                SELECT
                    COUNT(*) AS population_n,
                    COUNTIF(cxa_combined_score IS NOT NULL) AS chance_creating_n
                FROM `{PROJECT}.{SERVING_DATASET}.cxa_{track}_combined_v1`
                WHERE split = '{COVERAGE_SPLIT}'
            """
            cov_row = list(client.query(coverage_query).result())[0]
            population_n = cov_row["population_n"]
            chance_creating_n = cov_row["chance_creating_n"]
            coverage = CxaCoverage(
                split=COVERAGE_SPLIT,
                population_n=population_n,
                chance_creating_n=chance_creating_n,
                coverage_pct=round(chance_creating_n / population_n * 100, 3) if population_n else 0.0,
            )

            summaries.append(
                CxaModelSummary(
                    track=track,
                    stage_metrics=stage_rows,
                    coverage=coverage,
                    combined_score_caveat=COMBINED_SCORE_CAVEAT,
                )
            )
        return summaries

    # Cached per-track, same reasoning as CxgCoverageStore: each track's full
    # test-split combined table is small (event ~92.8k rows, plus ~18.6k rows,
    # same order of magnitude as CxG's own cached test-split predictions) and
    # identical for every visitor, so fetching it once per TTL window and doing
    # the pass_event_id lookup in memory beats caching per distinct request shape.
    @cached(cache=_coverage_cache, key=_track_cache_key, lock=_coverage_lock)
    def _get_track_coverage(self, track: str) -> dict[str, CxaCoverageValues]:
        if track not in TRACKS:
            raise ValueError(f"Unknown track: {track!r}")
        client = _client()
        query = f"""
            SELECT pass_event_id, p_create_predicted_prob, p_convert_predicted_prob, cxa_combined_score
            FROM `{PROJECT}.{SERVING_DATASET}.cxa_{track}_combined_v1`
            WHERE split = '{COVERAGE_SPLIT}'
        """
        rows = client.query(query).result()
        return {
            row["pass_event_id"]: CxaCoverageValues(
                p_create_predicted_prob=row["p_create_predicted_prob"],
                p_convert_predicted_prob=row["p_convert_predicted_prob"],
                cxa_combined_score=row["cxa_combined_score"],
            )
            for row in rows
        }

    def get_cxa_for_passes(self, pass_event_ids: list[str], *, track: str) -> dict[str, CxaCoverageValues]:
        coverage = self._get_track_coverage(track)
        return {pid: coverage[pid] for pid in pass_event_ids if pid in coverage}
