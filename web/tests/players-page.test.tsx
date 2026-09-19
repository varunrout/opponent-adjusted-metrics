import { describe, expect, it, vi } from "vitest";
import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import type { PlayerSeasonResponse } from "@/lib/types";

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn(), replace: vi.fn() }),
}));

vi.mock("@/components/shell/MatchFilterProvider", () => ({
  useMatchFilter: () => ({ competitionId: null, seasonId: null }),
}));

vi.mock("@/lib/api", () => ({
  getPlayers: vi.fn(),
}));

import { getPlayers } from "@/lib/api";
import PlayersPage from "@/app/players/page";

function makePlayer(overrides: Partial<PlayerSeasonResponse>): PlayerSeasonResponse {
  return {
    player_id: 1,
    player_name: "Test Player",
    team_id: 1,
    team_name: "Test FC",
    shots: 20,
    goals: 5,
    total_xg: 4,
    ...overrides,
  };
}

describe("PlayersPage", () => {
  it("computes xG/shot and signed G-xG correctly", async () => {
    vi.mocked(getPlayers).mockResolvedValue([
      makePlayer({ player_id: 1, player_name: "Overperformer", shots: 20, goals: 8, total_xg: 4 }),
    ]);
    render(<PlayersPage />);

    await waitFor(() => expect(screen.getByText("Overperformer")).toBeInTheDocument());
    // xG/shot = 4 / 20 = 0.20
    expect(screen.getByText("0.20")).toBeInTheDocument();
    // G-xG = 8 - 4 = +4.00, signed and positive
    expect(screen.getByText("+4.00")).toBeInTheDocument();
  });

  it("signs a negative G-xG with a minus and colours it red", async () => {
    vi.mocked(getPlayers).mockResolvedValue([
      makePlayer({ player_id: 2, player_name: "Underperformer", shots: 20, goals: 1, total_xg: 4 }),
    ]);
    render(<PlayersPage />);

    await waitFor(() => expect(screen.getByText("Underperformer")).toBeInTheDocument());
    const cell = screen.getByText("-3.00");
    expect(cell).toBeInTheDocument();
    expect(cell.style.color).toBe("var(--red)");
  });

  it("filters out players below the minimum-shots threshold, default 10", async () => {
    vi.mocked(getPlayers).mockResolvedValue([
      makePlayer({ player_id: 3, player_name: "One Shot Wonder", shots: 1, goals: 1, total_xg: 0.05 }),
      makePlayer({ player_id: 4, player_name: "Regular Starter", shots: 15, goals: 3, total_xg: 2 }),
    ]);
    render(<PlayersPage />);

    await waitFor(() => expect(screen.getByText("Regular Starter")).toBeInTheDocument());
    expect(screen.queryByText("One Shot Wonder")).not.toBeInTheDocument();
    expect(screen.getByText(/min\. 10 shots/)).toBeInTheDocument();
  });

  it("raising the minimum-shots input re-filters the table", async () => {
    vi.mocked(getPlayers).mockResolvedValue([
      makePlayer({ player_id: 5, player_name: "Twelve Shots", shots: 12, goals: 2, total_xg: 1 }),
    ]);
    render(<PlayersPage />);

    await waitFor(() => expect(screen.getByText("Twelve Shots")).toBeInTheDocument());
    const input = screen.getByLabelText("Minimum shots");
    fireEvent.change(input, { target: { value: "15" } });
    expect(screen.queryByText("Twelve Shots")).not.toBeInTheDocument();
  });
});
