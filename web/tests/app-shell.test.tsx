import { describe, expect, it, vi } from "vitest";
import { render, screen, cleanup } from "@testing-library/react";

// AppShell decides whether to render the filter Sidebar based on the
// current route. Mock next/navigation's usePathname per-test.
let mockPathname = "/overview";
vi.mock("next/navigation", () => ({
  usePathname: () => mockPathname,
}));

// Sidebar calls useMatchFilter() directly; mock it so this test doesn't
// need a real MatchFilterProvider (which fetches competitions on mount).
vi.mock("@/components/shell/MatchFilterProvider", () => ({
  useMatchFilter: () => ({
    competitions: [],
    competitionsLoading: false,
    competitionId: null,
    seasonId: null,
    setCompetitionId: vi.fn(),
    setSeasonId: vi.fn(),
  }),
}));

import { AppShell } from "@/components/shell/AppShell";

describe("AppShell sidebar scoping", () => {
  it.each([
    ["/matches", true],
    ["/matches/7", true],
    ["/players", true],
    ["/players/42", true],
    ["/teams", true],
    ["/teams/9", true],
    ["/overview", false],
    ["/stories", false],
    ["/models", false],
    ["/about", false],
    ["/analysis", false],
  ])("renders sidebar=%s for %s", (pathname, expected) => {
    mockPathname = pathname;
    render(
      <AppShell>
        <div>content</div>
      </AppShell>
    );
    const sidebar = screen.queryByText("Competition");
    if (expected) {
      expect(sidebar).toBeInTheDocument();
    } else {
      expect(sidebar).not.toBeInTheDocument();
    }
    cleanup();
  });
});
