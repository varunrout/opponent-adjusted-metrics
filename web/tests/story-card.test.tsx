import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { StoryCard } from "@/components/ui/StoryCard";
import type { StoryInfo } from "@/lib/stories-data";

const STORY: StoryInfo = {
  slug: "example-story",
  category: "Methodology",
  headline: "An example story",
};

describe("StoryCard", () => {
  it("links to its own /stories/[slug] route (confirmed live defect: used to render a bare div)", () => {
    render(<StoryCard story={STORY} />);
    const link = screen.getByRole("link");
    expect(link).toHaveAttribute("href", "/stories/example-story");
  });

  it("still renders category, headline, and optional takeaway inside the link", () => {
    render(<StoryCard story={{ ...STORY, takeaway: "A one-line takeaway" }} />);
    expect(screen.getByText("Methodology")).toBeInTheDocument();
    expect(screen.getByText("An example story")).toBeInTheDocument();
    expect(screen.getByText("A one-line takeaway")).toBeInTheDocument();
  });
});
