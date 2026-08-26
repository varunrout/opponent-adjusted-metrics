import { describe, expect, it, vi } from "vitest";
import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import { Sidebar } from "@/components/shell/Sidebar";
import { MatchFilterProvider, useMatchFilter } from "@/components/shell/MatchFilterProvider";
import type { CompetitionResponse, TeamSeasonResponse } from "@/lib/types";

vi.mock("@/lib/api", () => ({
  getCompetitions: vi.fn(),
  getTeams: vi.fn(),
}));

import { getCompetitions, getTeams } from "@/lib/api";

const COMPETITIONS: CompetitionResponse[] = [
  {
    competition_id: 2,
    season_id: 27,
    competition_name: "Premier League",
    competition_gender: "male",
    country_name: "England",
    season_name: "2015/2016",
    match_updated: null,
    match_available: null,
    match_updated_360: null,
    match_available_360: null,
  },
];

const TEAMS: TeamSeasonResponse[] = [
  { team_id: 10, team_name: "Leicester City", shots: 100, goals: 20, total_xg: 15 },
];

function ModeProbe() {
  const { metricMode } = useMatchFilter();
  return <div data-testid="metric-mode">{metricMode}</div>;
}

describe("Sidebar", () => {
  it("populates the Team dropdown from getTeams instead of a hardcoded single option", async () => {
    vi.mocked(getCompetitions).mockResolvedValue(COMPETITIONS);
    vi.mocked(getTeams).mockResolvedValue(TEAMS);

    render(
      <MatchFilterProvider>
        <Sidebar />
      </MatchFilterProvider>
    );

    await waitFor(() => expect(screen.getByText("Leicester City")).toBeInTheDocument());
  });

  it('the xG/CxG pills toggle metricMode without triggering a new data fetch', async () => {
    vi.mocked(getCompetitions).mockResolvedValue(COMPETITIONS);
    vi.mocked(getTeams).mockResolvedValue(TEAMS);

    render(
      <MatchFilterProvider>
        <Sidebar />
        <ModeProbe />
      </MatchFilterProvider>
    );

    await waitFor(() => expect(screen.getByText("Leicester City")).toBeInTheDocument());
    expect(screen.getByTestId("metric-mode")).toHaveTextContent("xg");

    const getTeamsCallsBefore = vi.mocked(getTeams).mock.calls.length;
    fireEvent.click(screen.getByText("CxG"));

    expect(screen.getByTestId("metric-mode")).toHaveTextContent("cxg");
    expect(vi.mocked(getTeams).mock.calls.length).toBe(getTeamsCallsBefore);
  });

  it("keeps CxA and CxT disabled with a soon badge", async () => {
    vi.mocked(getCompetitions).mockResolvedValue(COMPETITIONS);
    vi.mocked(getTeams).mockResolvedValue(TEAMS);

    render(
      <MatchFilterProvider>
        <Sidebar />
      </MatchFilterProvider>
    );

    await waitFor(() => expect(screen.getByText("Leicester City")).toBeInTheDocument());
    expect(screen.getAllByText("soon")).toHaveLength(2);
  });
});
