import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import type { MatchResponse, PlayerSeasonResponse, ShotResponse } from "@/lib/types";

vi.mock("next/navigation", () => ({
  useParams: () => ({ playerId: "100" }),
  useRouter: () => ({ push: vi.fn(), replace: vi.fn() }),
}));

vi.mock("@/components/shell/MatchFilterProvider", () => ({
  useMatchFilter: vi.fn(),
}));

vi.mock("@/lib/api", () => ({
  getPlayerShots: vi.fn(),
  getCxgCoverage: vi.fn(),
  getShotOpponentContext: vi.fn(),
  getPlayers: vi.fn(),
  getMatches: vi.fn(),
}));

import {
  getPlayerShots,
  getCxgCoverage,
  getShotOpponentContext,
  getPlayers,
  getMatches,
} from "@/lib/api";
import { useMatchFilter } from "@/components/shell/MatchFilterProvider";
import PlayerDetailPage from "@/app/players/[playerId]/page";

function mockFilter(overrides: Partial<{ cxgScopeOnly: boolean }> = {}) {
  vi.mocked(useMatchFilter).mockReturnValue({
    competitionId: null,
    seasonId: null,
    metricMode: "xg",
    cxgScopeOnly: false,
    ...overrides,
  } as ReturnType<typeof useMatchFilter>);
}

function makeShot(overrides: Partial<ShotResponse>): ShotResponse {
  return {
    event_id: "e1",
    match_id: 1,
    team_id: 10,
    player_id: 100,
    player_name: "Test Player",
    minute: 10,
    period: 1,
    location_x: 100,
    location_y: 40,
    end_x: null,
    end_y: null,
    statsbomb_xg: 0.3,
    outcome_name: "Goal",
    body_part_name: null,
    is_goal: true,
    ...overrides,
  };
}

function makePlayerSeason(overrides: Partial<PlayerSeasonResponse>): PlayerSeasonResponse {
  return {
    player_id: 100,
    player_name: "Test Player",
    team_id: 10,
    team_name: "Test FC",
    shots: 10,
    goals: 2,
    total_xg: 1.5,
    ...overrides,
  };
}

function makeMatch(overrides: Partial<MatchResponse>): MatchResponse {
  return {
    match_id: 1,
    competition_id: 2,
    season_id: 27,
    match_date: "2016-01-01",
    kick_off: null,
    home_team_id: 10,
    home_team_name: "Home FC",
    away_team_id: 20,
    away_team_name: "Away FC",
    home_score: 1,
    away_score: 0,
    home_xg: null,
    away_xg: null,
    competition_stage: null,
    stadium: null,
    referee: null,
    match_status: "available",
    match_status_360: "available",
    last_updated: null,
    last_updated_360: null,
    ...overrides,
  };
}

describe("PlayerDetailPage", () => {
  beforeEach(() => {
    vi.mocked(getPlayerShots).mockReset();
    vi.mocked(getCxgCoverage).mockReset().mockResolvedValue({ track: "cxg_event", values: {} });
    vi.mocked(getShotOpponentContext).mockReset().mockResolvedValue([]);
    vi.mocked(getPlayers).mockReset().mockResolvedValue([]);
    vi.mocked(getMatches).mockReset().mockResolvedValue([]);
    mockFilter();
  });

  it("renders the shot map and tiles from all shots when cxgScopeOnly is false", async () => {
    vi.mocked(getPlayerShots).mockResolvedValue([
      makeShot({ event_id: "e1", match_id: 1 }),
      makeShot({ event_id: "e2", match_id: 2 }),
    ]);

    render(<PlayerDetailPage />);

    await waitFor(() => expect(screen.getAllByTestId("shot-marker").length).toBe(2));
    expect(getMatches).not.toHaveBeenCalled();
  });

  describe("CxG matches only scope", () => {
    it("excludes shots from non-360 matches (match_status_360 !== available) when cxgScopeOnly is true", async () => {
      mockFilter({ cxgScopeOnly: true });
      vi.mocked(getPlayerShots).mockResolvedValue([
        makeShot({ event_id: "covered-shot", match_id: 1, is_goal: true, statsbomb_xg: 0.3 }),
        makeShot({ event_id: "uncovered-shot", match_id: 2, is_goal: true, statsbomb_xg: 0.5 }),
      ]);
      vi.mocked(getMatches).mockResolvedValue([
        makeMatch({ match_id: 1, match_status_360: "available" }),
        makeMatch({ match_id: 2, match_status_360: "scheduled" }),
      ]);

      render(<PlayerDetailPage />);

      await waitFor(() => expect(screen.getAllByTestId("shot-marker").length).toBe(1));
      expect(getMatches).toHaveBeenCalledWith({});
    });

    it("shows every shot, regardless of match_status_360, when cxgScopeOnly is false", async () => {
      mockFilter({ cxgScopeOnly: false });
      vi.mocked(getPlayerShots).mockResolvedValue([
        makeShot({ event_id: "e1", match_id: 1 }),
        makeShot({ event_id: "e2", match_id: 2 }),
      ]);

      render(<PlayerDetailPage />);

      await waitFor(() => expect(screen.getAllByTestId("shot-marker").length).toBe(2));
      expect(getMatches).not.toHaveBeenCalled();
    });

    it("computes percentile from the full players list, unaffected by scope", async () => {
      mockFilter({ cxgScopeOnly: true });
      vi.mocked(getPlayerShots).mockResolvedValue([makeShot({ event_id: "e1", match_id: 1 })]);
      vi.mocked(getMatches).mockResolvedValue([makeMatch({ match_id: 1, match_status_360: "available" })]);
      vi.mocked(getPlayers).mockResolvedValue([
        makePlayerSeason({ player_id: 100, shots: 10, goals: 5, total_xg: 2 }),
        makePlayerSeason({ player_id: 200, shots: 10, goals: 1, total_xg: 5 }),
      ]);

      render(<PlayerDetailPage />);

      await waitFor(() => expect(screen.getByText("Percentile vs the full players list")).toBeInTheDocument());
    });
  });
});
