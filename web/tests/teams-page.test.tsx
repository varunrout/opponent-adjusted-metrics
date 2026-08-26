import { describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import type { TeamSeasonResponse } from "@/lib/types";

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn(), replace: vi.fn() }),
}));

vi.mock("@/components/shell/MatchFilterProvider", () => ({
  useMatchFilter: () => ({ competitionId: null, seasonId: null }),
}));

vi.mock("@/lib/api", () => ({
  getTeams: vi.fn(),
}));

import { getTeams } from "@/lib/api";
import TeamsPage from "@/app/teams/page";

function makeTeam(overrides: Partial<TeamSeasonResponse>): TeamSeasonResponse {
  return { team_id: 1, team_name: "Test FC", shots: 20, goals: 5, total_xg: 4, ...overrides };
}

describe("TeamsPage", () => {
  it("computes xG/shot and signed G-xG, and links rows via ClickableRow (not a plain <a>)", async () => {
    vi.mocked(getTeams).mockResolvedValue([
      makeTeam({ team_id: 1, team_name: "Overperforming FC", shots: 20, goals: 8, total_xg: 4 }),
    ]);
    render(<TeamsPage />);

    await waitFor(() => expect(screen.getByText("Overperforming FC")).toBeInTheDocument());
    expect(screen.getByText("0.20")).toBeInTheDocument();
    expect(screen.getByText("+4.00")).toBeInTheDocument();

    const row = screen.getByRole("link");
    expect(row.tagName).toBe("DIV");
    expect(row).toHaveAttribute("tabIndex", "0");
  });

  it("filters below the default minimum-shots threshold", async () => {
    vi.mocked(getTeams).mockResolvedValue([
      makeTeam({ team_id: 2, team_name: "Small Sample FC", shots: 3, goals: 1, total_xg: 0.2 }),
    ]);
    render(<TeamsPage />);

    await waitFor(() => expect(screen.getByText(/No teams found/)).toBeInTheDocument());
  });
});
