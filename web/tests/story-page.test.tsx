import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import StoryPage from "@/app/stories/[slug]/page";
import { STORIES } from "@/lib/stories-data";

describe("StoryPage", () => {
  it("renders the full CxG v3 article body for its slug", () => {
    render(<StoryPage params={{ slug: "cxg-v3-honest-comparison" }} />);
    expect(
      screen.getByText("CxG v3 against StatsBomb xG: an honest comparison")
    ).toBeInTheDocument();
    expect(screen.getByText(/0.3003/)).toBeInTheDocument();
    // Mentioned in two separate paragraphs (the caveat, then what it would
    // take to close the gap), so this must be an All-variant query.
    expect(screen.getAllByText(/zone displacement/).length).toBeGreaterThan(0);
  });

  it("renders a body for every story, with no stubs left", () => {
    // Every headline on /stories now leads to a real article. If a new
    // story is added without a body, this fails rather than silently
    // shipping a dead-end card.
    for (const story of STORIES) {
      expect(story.body, `story "${story.slug}" has no body`).toBeDefined();
      expect(story.body!.length).toBeGreaterThan(0);
    }
  });

  it("reports the honest negative result rather than claiming CxG uses late-game context", () => {
    render(<StoryPage params={{ slug: "late-game-features-that-failed" }} />);
    expect(screen.getByText(/None of the fifteen validated/)).toBeInTheDocument();
  });

  it("links back to the Stories index", () => {
    render(<StoryPage params={{ slug: "cxg-v1-to-v3" }} />);
    const back = screen.getByRole("link", { name: /Back to Stories/ });
    expect(back).toHaveAttribute("href", "/stories");
  });
});
