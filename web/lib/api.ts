import type {
  CompetitionResponse,
  MatchDetailResponse,
  MatchResponse,
  PlayerSeasonResponse,
  ShotResponse,
  TeamSeasonResponse,
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

async function apiFetch<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE_URL}${path}`);
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

export { ApiError };
