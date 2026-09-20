import type {
  BivariateResponse,
  ChartsResponse,
  CompetitionResponse,
  CxaCoverageResponse,
  CxaExplainability,
  CxaModelSummary,
  CxgCoefficientResponse,
  CxgCoverageResponse,
  CxgModelResultResponse,
  FeatureCorrelationResponse,
  FeatureInventoryResponse,
  MatchDetailResponse,
  MatchResponse,
  MeResponse,
  OpponentContextResponse,
  PcaResponse,
  PlayerSeasonResponse,
  ShotFreezeFrameResponse,
  ShotResponse,
  TeamSeasonResponse,
  UnivariateTargetResponse,
} from "@/lib/types";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";

class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE_URL}${path}`, init);
  if (!res.ok) {
    throw new ApiError(res.status, `Request to ${path} failed with status ${res.status}`);
  }
  return (await res.json()) as T;
}

export function getCompetitions(): Promise<CompetitionResponse[]> {
  return apiFetch<CompetitionResponse[]>("/v1/competitions");
}

export function getMatches(filters: {
  competition_id?: number | null;
  season_id?: number | null;
  team_id?: number | null;
}): Promise<MatchResponse[]> {
  const params = new URLSearchParams();
  if (filters.competition_id != null) {
    params.set("competition_id", String(filters.competition_id));
  }
  if (filters.season_id != null) {
    params.set("season_id", String(filters.season_id));
  }
  if (filters.team_id != null) {
    params.set("team_id", String(filters.team_id));
  }
  const qs = params.toString();
  return apiFetch<MatchResponse[]>(`/v1/matches${qs ? `?${qs}` : ""}`);
}

export function getMatch(matchId: number | string): Promise<MatchDetailResponse> {
  return apiFetch<MatchDetailResponse>(`/v1/matches/${matchId}`);
}

export function getMatchShots(matchId: number | string): Promise<ShotResponse[]> {
  return apiFetch<ShotResponse[]>(`/v1/matches/${matchId}/shots`);
}

// Real StatsBomb 360 positions (teammates/opponents/GK) for one shot, from
// oam_core.three_sixty_frames/three_sixty_players — 404 when the shot has
// no 360 frame (most shots; only 166 of 610 matches carry one), which this
// resolves to null rather than letting it surface as a page-level error.
export function getShotFreezeFrame(
  matchId: number | string,
  eventId: string
): Promise<ShotFreezeFrameResponse | null> {
  return apiFetch<ShotFreezeFrameResponse>(`/v1/matches/${matchId}/shots/${eventId}/freeze-frame`).catch(
    (err) => {
      if (err instanceof ApiError && err.status === 404) return null;
      throw err;
    }
  );
}

export function getPlayers(filters: {
  competition_id?: number | null;
  season_id?: number | null;
}): Promise<PlayerSeasonResponse[]> {
  const params = new URLSearchParams();
  if (filters.competition_id != null) {
    params.set("competition_id", String(filters.competition_id));
  }
  if (filters.season_id != null) {
    params.set("season_id", String(filters.season_id));
  }
  const qs = params.toString();
  return apiFetch<PlayerSeasonResponse[]>(`/v1/players${qs ? `?${qs}` : ""}`);
}

export function getPlayerShots(
  playerId: number | string,
  filters: { competition_id?: number | null; season_id?: number | null }
): Promise<ShotResponse[]> {
  const params = new URLSearchParams();
  if (filters.competition_id != null) {
    params.set("competition_id", String(filters.competition_id));
  }
  if (filters.season_id != null) {
    params.set("season_id", String(filters.season_id));
  }
  const qs = params.toString();
  return apiFetch<ShotResponse[]>(`/v1/players/${playerId}/shots${qs ? `?${qs}` : ""}`);
}

export function getTeams(filters: {
  competition_id?: number | null;
  season_id?: number | null;
}): Promise<TeamSeasonResponse[]> {
  const params = new URLSearchParams();
  if (filters.competition_id != null) {
    params.set("competition_id", String(filters.competition_id));
  }
  if (filters.season_id != null) {
    params.set("season_id", String(filters.season_id));
  }
  const qs = params.toString();
  return apiFetch<TeamSeasonResponse[]>(`/v1/teams${qs ? `?${qs}` : ""}`);
}

export function getTeamShots(
  teamId: number | string,
  filters: { competition_id?: number | null; season_id?: number | null }
): Promise<ShotResponse[]> {
  const params = new URLSearchParams();
  if (filters.competition_id != null) {
    params.set("competition_id", String(filters.competition_id));
  }
  if (filters.season_id != null) {
    params.set("season_id", String(filters.season_id));
  }
  const qs = params.toString();
  return apiFetch<ShotResponse[]>(`/v1/teams/${teamId}/shots${qs ? `?${qs}` : ""}`);
}

export function getTeamShotsFaced(
  teamId: number | string,
  filters: { competition_id?: number | null; season_id?: number | null }
): Promise<ShotResponse[]> {
  const params = new URLSearchParams();
  if (filters.competition_id != null) {
    params.set("competition_id", String(filters.competition_id));
  }
  if (filters.season_id != null) {
    params.set("season_id", String(filters.season_id));
  }
  const qs = params.toString();
  return apiFetch<ShotResponse[]>(`/v1/teams/${teamId}/shots-faced${qs ? `?${qs}` : ""}`);
}

export function getMe(idToken?: string | null): Promise<MeResponse> {
  const headers: HeadersInit | undefined = idToken ? { Authorization: `Bearer ${idToken}` } : undefined;
  return apiFetch<MeResponse>("/v1/me", headers ? { headers } : undefined);
}

// --- /v1/analysis/* (admin-gated: requires a Firebase ID token with role
// "admin"; missing/non-admin tokens surface as a 403 ApiError) -----------

function authHeaders(idToken?: string | null): RequestInit | undefined {
  return idToken ? { headers: { Authorization: `Bearer ${idToken}` } } : undefined;
}

export function getAnalysisFeatures(
  idToken: string | null | undefined,
  family?: string
): Promise<FeatureInventoryResponse[]> {
  const qs = family ? `?family=${encodeURIComponent(family)}` : "";
  return apiFetch<FeatureInventoryResponse[]>(`/v1/analysis/features${qs}`, authHeaders(idToken));
}

export function getAnalysisCorrelation(
  idToken: string | null | undefined,
  family?: string
): Promise<FeatureCorrelationResponse[]> {
  const qs = family ? `?family=${encodeURIComponent(family)}` : "";
  return apiFetch<FeatureCorrelationResponse[]>(`/v1/analysis/correlation${qs}`, authHeaders(idToken));
}

export function getAnalysisUnivariate(
  idToken: string | null | undefined,
  family?: string
): Promise<UnivariateTargetResponse[]> {
  const qs = family ? `?family=${encodeURIComponent(family)}` : "";
  return apiFetch<UnivariateTargetResponse[]>(`/v1/analysis/univariate${qs}`, authHeaders(idToken));
}

export function getAnalysisBivariate(idToken: string | null | undefined): Promise<BivariateResponse> {
  return apiFetch<BivariateResponse>("/v1/analysis/bivariate", authHeaders(idToken));
}

export function getAnalysisPca(idToken: string | null | undefined): Promise<PcaResponse> {
  return apiFetch<PcaResponse>("/v1/analysis/pca", authHeaders(idToken));
}

export function getAnalysisCharts(
  idToken: string | null | undefined,
  runId?: string
): Promise<ChartsResponse> {
  const qs = runId ? `?run_id=${encodeURIComponent(runId)}` : "";
  return apiFetch<ChartsResponse>(`/v1/analysis/charts${qs}`, authHeaders(idToken));
}

export function getCxgModelResults(
  idToken: string | null | undefined
): Promise<CxgModelResultResponse[]> {
  return apiFetch<CxgModelResultResponse[]>("/v1/analysis/cxg-models", authHeaders(idToken));
}

export function getCxgModelCoefficients(
  idToken: string | null | undefined,
  modelKey: string
): Promise<CxgCoefficientResponse[]> {
  return apiFetch<CxgCoefficientResponse[]>(
    `/v1/analysis/cxg-models/${encodeURIComponent(modelKey)}/coefficients`,
    authHeaders(idToken)
  );
}

// --- /v1/models/* (public mirror of /v1/analysis/cxg-models*; no admin
// gate, no auth header — powers the public Models page. Deliberate
// divergence from the admin-only Analysis tab's identical-shaped data:
// see routers/models.py's own docstring for why. Same response types as
// the admin versions above; don't conflate the two endpoints.) ----------

export function getPublicCxgModelResults(): Promise<CxgModelResultResponse[]> {
  return apiFetch<CxgModelResultResponse[]>("/v1/models/cxg-models");
}

export function getPublicCxgModelCoefficients(modelKey: string): Promise<CxgCoefficientResponse[]> {
  return apiFetch<CxgCoefficientResponse[]>(
    `/v1/models/cxg-models/${encodeURIComponent(modelKey)}/coefficients`
  );
}

// --- /v1/cxg/coverage (guest-accessible Explore-zone endpoint; no auth) --

export function getCxgCoverage(eventIds: string[], track: string): Promise<CxgCoverageResponse> {
  const qs = `?track=${encodeURIComponent(track)}&event_ids=${encodeURIComponent(eventIds.join(","))}`;
  return apiFetch<CxgCoverageResponse>(`/v1/cxg/coverage${qs}`);
}

// --- /v1/cxg/opponent-context (guest-accessible Explore-zone endpoint) --

export function getShotOpponentContext(eventIds: string[]): Promise<OpponentContextResponse[]> {
  if (eventIds.length === 0) return Promise.resolve([]);
  const qs = `?event_ids=${encodeURIComponent(eventIds.join(","))}`;
  return apiFetch<OpponentContextResponse[]>(`/v1/cxg/opponent-context${qs}`);
}

// --- /v1/models/cxa-models (public, no auth) — powers the public Models
// page's CxA (event-only) / CxA+ cards. Not yet called by any page in this
// task (the Models page stays static per models-data.ts, same as CxG) — this
// function exists so a future live/detail view has it ready, mirroring how
// getPublicCxgModelResults already exists for that same reason. -----------

export function getPublicCxaModelSummaries(): Promise<CxaModelSummary[]> {
  return apiFetch<CxaModelSummary[]>("/v1/models/cxa-models");
}

export function getPublicCxaExplainability(track: string): Promise<CxaExplainability> {
  return apiFetch<CxaExplainability>(`/v1/models/cxa-models/${encodeURIComponent(track)}/explainability`);
}

// --- /v1/cxa/coverage (guest-accessible; not yet called anywhere — see
// docs/analysis/cxa_dashboard_models_page_v1.md's "what's next": a per-pass
// display is a separate, deferred future task) -----------------------------

export function getCxaCoverage(passEventIds: string[], track: string): Promise<CxaCoverageResponse> {
  const qs = `?track=${encodeURIComponent(track)}&pass_event_ids=${encodeURIComponent(passEventIds.join(","))}`;
  return apiFetch<CxaCoverageResponse>(`/v1/cxa/coverage${qs}`);
}

export { ApiError };
