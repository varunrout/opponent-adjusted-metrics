import { describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import type { MatchResponse, ShotResponse } from "@/lib/types";

vi.mock("next/navigation", () => ({
  useParams: () => ({ teamId: "10" }),
  useRouter: () => ({ push: vi.fn(), replace: vi.fn() }),
}));

vi.mock("@/components/shell/MatchFilterProvider", () => ({
  useMatchFilter: () => ({ competitionId: null, seasonId: null }),
}));

vi.mock("@/lib/api", () => ({
  getTeamShots: vi.fn(),
  getMatches: vi.fn(),
  getCxgCoverage: vi.fn(),
}));

import { getTeamShots, getMatches, getCxgCoverage } from "@/lib/api";
import TeamDetailPage from "@/app/teams/[teamId]/page";

function makeShot(overrides: Partial<ShotResponse>): ShotResponse {
  return {
    event_id: "e1",
    match_id: 1,
    team_id: 10,
    player_id: 100,
    player_name: "Top Scorer",
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
    competition_stage: null,
    stadium: null,
    referee: null,
    match_status: null,
    match_status_360: null,
    last_updated: null,
    last_updated_360: null,
    ...overrides,
  };
}

describe("TeamDetailPage", () => {
  it("renders a shot map, CxG coverage caption, derived tiles, and top scorers", async () => {
    vi.mocked(getTeamShots).mockResolvedValue([
      makeShot({ event_id: "e1", player_id: 100, player_name: "Scorer A", is_goal: true, statsbomb_xg: 0.3 }),
      makeShot({ event_id: "e2", player_id: 100, player_name: "Scorer A", is_goal: false, statsbomb_xg: 0.1 }),
    ]);
    vi.mocked(getMatches).mockResolvedValue([makeMatch({})]);
    vi.mocked(getCxgCoverage).mockImplementation((_ids, track) =>
      Promise.resolve({ track, values: track === "cxg_event" ? { e1: 0.25 } : ({} as Record<string, number>) })
    );

    render(<TeamDetailPage />);

    await waitFor(() => expect(screen.getAllByText("Home FC").length).toBeGreaterThan(0));
    // xG/shot = 0.4 / 2 = 0.20; G-xG = 1 - 0.4 = +0.60
    expect(screen.getByText("0.20")).toBeInTheDocument();
    expect(screen.getByText("+0.60")).toBeInTheDocument();
    // CxG coverage caption (1 of 2 shots covered)
    expect(screen.getByText(/1 of 2 shots have CxG coverage/)).toBeInTheDocument();
    // Visible disclosure badge next to it, not just a hover title (v3 §8.1 —
    // hover-only disclosure fails entirely on touch devices).
    expect(screen.getByText("Experimental")).toBeInTheDocument();
    // Top scorer
    expect(screen.getByText("Scorer A")).toBeInTheDocument();
    // Shot map renders markers
    expect(screen.getAllByTestId("shot-marker").length).toBe(2);
  });

  it("degrades cxg_event and cxg_plus coverage independently on failure", async () => {
    vi.mocked(getTeamShots).mockResolvedValue([makeShot({ event_id: "e1" })]);
    vi.mocked(getMatches).mockResolvedValue([]);
    vi.mocked(getCxgCoverage).mockImplementation((_ids, track) =>
      track === "cxg_event" ? Promise.reject(new Error("boom")) : Promise.resolve({ track, values: {} })
    );

    render(<TeamDetailPage />);

    await waitFor(() => expect(screen.getAllByTestId("shot-marker").length).toBe(1));
    // Neither track has coverage now, so no caption — page still renders fine.
    expect(screen.queryByText(/CxG coverage/)).not.toBeInTheDocument();
  });
});
