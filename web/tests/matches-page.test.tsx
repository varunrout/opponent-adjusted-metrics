import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import type { MatchResponse } from "@/lib/types";

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn(), replace: vi.fn() }),
}));

vi.mock("@/components/shell/MatchFilterProvider", () => ({
  useMatchFilter: vi.fn(),
}));

vi.mock("@/lib/api", () => ({
  getMatches: vi.fn(),
}));

import { getMatches } from "@/lib/api";
import { useMatchFilter } from "@/components/shell/MatchFilterProvider";
import MatchesPage from "@/app/matches/page";

function mockFilter(overrides: Partial<{ cxgScopeOnly: boolean }> = {}) {
  vi.mocked(useMatchFilter).mockReturnValue({
    competitionId: null,
    seasonId: null,
    teamId: null,
    cxgScopeOnly: false,
    ...overrides,
  } as ReturnType<typeof useMatchFilter>);
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
    home_score: 2,
    away_score: 0,
    home_xg: 1.4,
    away_xg: 0.6,
    competition_stage: "Regular Season",
    stadium: "Some Stadium",
    referee: null,
    match_status: "available",
    match_status_360: "available",
    last_updated: null,
    last_updated_360: null,
    ...overrides,
  };
}

describe("MatchesPage", () => {
  beforeEach(() => {
    vi.mocked(getMatches).mockReset();
    mockFilter();
  });

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

  it("shows a real dataset-wide coverage strip computed from match_status, not hardcoded", async () => {
    vi.mocked(getMatches).mockResolvedValue([
      makeMatch({ match_status: "available" }),
      makeMatch({ match_id: 2, match_status: "processing" }),
    ]);
    render(<MatchesPage />);

    await waitFor(() => expect(screen.getByTestId("cxg-coverage-strip")).toBeInTheDocument());
    expect(screen.getByTestId("cxg-coverage-strip")).toHaveTextContent("1 of 2 matches");
  });

  it("renders the score column as a diverging bar when xG is available", async () => {
    vi.mocked(getMatches).mockResolvedValue([makeMatch({ home_xg: 1.8, away_xg: 0.4 })]);
    render(<MatchesPage />);

    await waitFor(() => expect(screen.getAllByTestId("diverging-bar").length).toBeGreaterThan(0));
  });

  describe("CxG matches only scope", () => {
    it("filters out matches with no 360 coverage (match_status_360 !== available) when cxgScopeOnly is true", async () => {
      mockFilter({ cxgScopeOnly: true });
      vi.mocked(getMatches).mockResolvedValue([
        makeMatch({
          match_id: 7532,
          home_team_name: "Peru",
          away_team_name: "Denmark",
          match_status_360: "scheduled",
        }),
        makeMatch({
          match_id: 99,
          home_team_name: "Covered A",
          away_team_name: "Covered B",
          match_status_360: "available",
        }),
      ]);

      render(<MatchesPage />);

      await waitFor(() => expect(screen.getByText("Covered A")).toBeInTheDocument());
      expect(screen.queryByText("Peru")).not.toBeInTheDocument();
    });

    it("shows every match, regardless of match_status_360, when cxgScopeOnly is false", async () => {
      mockFilter({ cxgScopeOnly: false });
      vi.mocked(getMatches).mockResolvedValue([
        makeMatch({
          match_id: 7532,
          home_team_name: "Peru",
          away_team_name: "Denmark",
          match_status_360: "scheduled",
        }),
      ]);

      render(<MatchesPage />);

      await waitFor(() => expect(screen.getByText("Peru")).toBeInTheDocument());
    });

    it("keeps the coverage strip unaffected by the scope toggle", async () => {
      mockFilter({ cxgScopeOnly: true });
      vi.mocked(getMatches).mockResolvedValue([
        makeMatch({ match_id: 1, match_status: "available", match_status_360: "scheduled" }),
        makeMatch({ match_id: 2, match_status: "available", match_status_360: "scheduled" }),
      ]);

      render(<MatchesPage />);

      // Coverage strip reflects match_status (event-wide precondition) over
      // the full unfiltered getMatches({}) result — both matches here are
      // match_status "available" despite match_status_360 not being, and
      // despite the scope toggle being on for the list itself.
      await waitFor(() => expect(screen.getByTestId("cxg-coverage-strip")).toBeInTheDocument());
      expect(screen.getByTestId("cxg-coverage-strip")).toHaveTextContent("2 of 2 matches");
    });
  });
});
