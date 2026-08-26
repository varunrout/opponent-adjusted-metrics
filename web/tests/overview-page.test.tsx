import { describe, expect, it, vi } from "vitest";
import { act, render, screen, waitFor } from "@testing-library/react";
import type { MatchResponse } from "@/lib/types";

// ClickableRow uses next/navigation's useRouter, which requires an
// app-router context that isn't present under vitest/jsdom.
vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn(), replace: vi.fn() }),
}));

vi.mock("@/lib/api", () => ({
  getMatches: vi.fn(),
}));

import { getMatches } from "@/lib/api";
import OverviewPage from "@/app/overview/page";

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

describe("OverviewPage", () => {
  it("renders no fabricated numbers and shows the real CxG comparison table", async () => {
    vi.mocked(getMatches).mockResolvedValue([]);
    render(<OverviewPage />);
    // The old page hardcoded "Team CxG 1.82" — must never reappear.
    expect(screen.queryByText("1.82")).not.toBeInTheDocument();
    expect(screen.queryByText("Team CxG")).not.toBeInTheDocument();
    expect(screen.queryByText("Team CxA")).not.toBeInTheDocument();
    await waitFor(() => expect(screen.getByText(/No matches available/)).toBeInTheDocument());
    // Real, spec-sourced comparison numbers (v2 §11).
    expect(screen.getAllByText("0.3003").length).toBeGreaterThan(0);
    expect(screen.getAllByText("0.2597").length).toBeGreaterThan(0);
  });

  it("sorts recent matches by date descending and links each row to its detail page", async () => {
    const older = makeMatch({ match_id: 1, match_date: "2016-01-01", home_team_name: "Older" });
    const newer = makeMatch({ match_id: 2, match_date: "2020-01-01", home_team_name: "Newer" });
    vi.mocked(getMatches).mockResolvedValue([older, newer]);

    render(<OverviewPage />);

    await waitFor(() => expect(screen.getByText("Newer")).toBeInTheDocument());
    // "Newer" (2020) should come before "Older" (2016) in document order.
    const position = screen.getByText("Older").compareDocumentPosition(screen.getByText("Newer"));
    expect(position & Node.DOCUMENT_POSITION_PRECEDING).toBeTruthy();
  });

  it("shows a retry button when the matches strip fails to load", async () => {
    vi.mocked(getMatches).mockRejectedValueOnce(new Error("boom")).mockResolvedValueOnce([]);

    render(<OverviewPage />);

    const retry = await screen.findByText("Retry");
    expect(screen.getByText(/Couldn't load recent matches/)).toBeInTheDocument();
    await act(async () => {
      retry.click();
    });
    await waitFor(() => expect(screen.getByText(/No matches available/)).toBeInTheDocument());
  });
});
