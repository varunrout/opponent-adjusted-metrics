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

  it("renders the new same-gap-two-percentages story with its split-calc figure", () => {
    render(<StoryPage params={{ slug: "same-gap-two-percentages" }} />);
    expect(
      screen.getByText("The gap to StatsBomb was 15%. It was also 17.7%. Both were right.")
    ).toBeInTheDocument();
    expect(screen.getByText("15.08%")).toBeInTheDocument();
    expect(screen.getByText("17.72%")).toBeInTheDocument();
  });

  it("renders publish-marker-bug's gate table and claimed-vs-actual timeline", () => {
    render(<StoryPage params={{ slug: "publish-marker-bug" }} />);
    expect(screen.getByText("Closure gates")).toBeInTheDocument();
    expect(screen.getByText("PASS — 198/198")).toBeInTheDocument();
    expect(screen.getByTestId("upload-timeline-actual")).toHaveTextContent("_SUCCESS uploaded last");
  });

  it("renders everything-was-3x-too-big's copies diagram and real-vs-shown bars", () => {
    render(<StoryPage params={{ slug: "everything-was-3x-too-big" }} />);
    expect(screen.getByText("Real vs. shown on site")).toBeInTheDocument();
    expect(screen.getAllByTestId("grouped-bars-row")).toHaveLength(3);
  });

  it("renders late-game-features-that-failed's univariate and bivariate figures", () => {
    render(<StoryPage params={{ slug: "late-game-features-that-failed" }} />);
    expect(screen.getByTestId("pass-fail-list")).toHaveTextContent("Match minute");
    expect(screen.getByText("Did not clear")).toBeInTheDocument();
    expect(screen.getByText("GK index × manpower difference")).toBeInTheDocument();
  });

  it("renders no figures on the two locked stories", () => {
    const { unmount } = render(<StoryPage params={{ slug: "cxg-v1-to-v3" }} />);
    expect(screen.queryByTestId("story-figure")).not.toBeInTheDocument();
    unmount();

    render(<StoryPage params={{ slug: "cxg-v3-honest-comparison" }} />);
    expect(screen.queryByTestId("story-figure")).not.toBeInTheDocument();
  });
});
