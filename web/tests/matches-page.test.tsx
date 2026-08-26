import { describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import type { MatchResponse } from "@/lib/types";

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn(), replace: vi.fn() }),
}));

vi.mock("@/components/shell/MatchFilterProvider", () => ({
  useMatchFilter: () => ({ competitionId: null, seasonId: null }),
}));

vi.mock("@/lib/api", () => ({
  getMatches: vi.fn(),
}));

import { getMatches } from "@/lib/api";
import MatchesPage from "@/app/matches/page";

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
    home_score: 2,
    away_score: 0,
    competition_stage: "Regular Season",
    stadium: "Some Stadium",
    referee: null,
    match_status: null,
    match_status_360: null,
    last_updated: null,
    last_updated_360: null,
    ...overrides,
  };
}

describe("MatchesPage", () => {
  it("renders Stage and Venue columns from previously-unused fields", async () => {
    vi.mocked(getMatches).mockResolvedValue([makeMatch({})]);
    render(<MatchesPage />);

    await waitFor(() => expect(screen.getByText("Regular Season")).toBeInTheDocument());
    expect(screen.getByText("Some Stadium")).toBeInTheDocument();
  });

  it("colours the winning side's name at --text and the losing side at --text2", async () => {
    vi.mocked(getMatches).mockResolvedValue([
      makeMatch({ home_team_name: "Winner", away_team_name: "Loser", home_score: 2, away_score: 0 }),
    ]);
    render(<MatchesPage />);

    await waitFor(() => expect(screen.getByText("Winner")).toBeInTheDocument());
    expect(screen.getByText("Winner")).toHaveClass("text-text");
    expect(screen.getByText("Loser")).toHaveClass("text-text2");
  });

  it("leaves both team names at --text for a draw", async () => {
    vi.mocked(getMatches).mockResolvedValue([
      makeMatch({ home_team_name: "Side A", away_team_name: "Side B", home_score: 1, away_score: 1 }),
    ]);
    render(<MatchesPage />);

    await waitFor(() => expect(screen.getByText("Side A")).toBeInTheDocument());
    expect(screen.getByText("Side A")).toHaveClass("text-text");
    expect(screen.getByText("Side B")).toHaveClass("text-text");
  });
});
