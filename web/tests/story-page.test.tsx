import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import StoryPage from "@/app/stories/[slug]/page";

describe("StoryPage", () => {
  it("renders the full CxG v3 article body for its slug", () => {
    render(<StoryPage params={{ slug: "cxg-v3-honest-comparison" }} />);
    expect(screen.getByText("CxG v3 — an honest comparison against StatsBomb xG")).toBeInTheDocument();
    expect(screen.getByText(/CxG v3 trails/)).toBeInTheDocument();
    expect(screen.getByText(/0.3003/)).toBeInTheDocument();
    expect(screen.getAllByText(/zone_displacement/).length).toBeGreaterThan(0);
  });

  it("shows an honest 'writeup in progress' state for a story with no body yet", () => {
    render(<StoryPage params={{ slug: "cxg-late-game-context" }} />);
    expect(screen.getByText(/Writeup in progress/)).toBeInTheDocument();
  });

  it("links back to the Stories index", () => {
    render(<StoryPage params={{ slug: "cxg-v1-release-notes" }} />);
    const back = screen.getByRole("link", { name: /Back to Stories/ });
    expect(back).toHaveAttribute("href", "/stories");
  });
});
