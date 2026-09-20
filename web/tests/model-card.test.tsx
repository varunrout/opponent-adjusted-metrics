import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { ModelCard } from "@/components/ui/ModelCard";
import type { ModelInfo } from "@/lib/models-data";

const EVALUATED_MODEL: ModelInfo = {
  name: "CxG",
  status: "evaluated",
  statusLabel: "Evaluated",
  tier: "Core",
  validationMetrics: [{ label: "Test log_loss", value: "0.3003" }],
  featureFamilyCount: "8 features",
  comparisonNote: "Trails the StatsBomb xG baseline (log_loss 0.2597).",
  comparisonStoryHref: "/stories/cxg-v3-honest-comparison",
};

describe("ModelCard", () => {
  it("renders the comparisonNote when present", () => {
    render(<ModelCard model={EVALUATED_MODEL} />);
    expect(screen.getByText("Trails the StatsBomb xG baseline (log_loss 0.2597).")).toBeInTheDocument();
  });

  it("does not render a comparisonNote paragraph when absent", () => {
    const withoutNote: ModelInfo = { ...EVALUATED_MODEL, comparisonNote: undefined };
    render(<ModelCard model={withoutNote} />);
    expect(screen.queryByText(/Trails the StatsBomb/)).not.toBeInTheDocument();
    expect(screen.queryByRole("link", { name: /See Stories/ })).not.toBeInTheDocument();
  });

  it("links to the specific CxG v3 comparison story next to the comparisonNote", () => {
    render(<ModelCard model={EVALUATED_MODEL} />);
    const link = screen.getByRole("link", { name: /See Stories for the full comparison/ });
    expect(link).toHaveAttribute("href", "/stories/cxg-v3-honest-comparison");
  });

  it("renders comparisonNote as plain text, with no Stories link, when comparisonStoryHref is unset", () => {
    // The CxA/CxA+ case: a real comparisonNote (the coverage caveat) with no
    // corresponding story to link to yet. Must not fall back to CxG's link.
    const withoutStory: ModelInfo = { ...EVALUATED_MODEL, comparisonStoryHref: undefined };
    render(<ModelCard model={withoutStory} />);
    expect(screen.getByText("Trails the StatsBomb xG baseline (log_loss 0.2597).")).toBeInTheDocument();
    expect(screen.queryByRole("link", { name: /See Stories/ })).not.toBeInTheDocument();
  });

  it("renders the experimental badge and note when experimentalNote is set", () => {
    const withExperimental: ModelInfo = {
      ...EVALUATED_MODEL,
      experimentalNote: "2,830 total chance-creating passes, 419 in test, zero Premier League rows.",
    };
    render(<ModelCard model={withExperimental} />);
    expect(screen.getByText("Experimental")).toBeInTheDocument();
    expect(
      screen.getByText("2,830 total chance-creating passes, 419 in test, zero Premier League rows.")
    ).toBeInTheDocument();
  });

  it("does not render the experimental badge when experimentalNote is absent", () => {
    render(<ModelCard model={EVALUATED_MODEL} />);
    expect(screen.queryByText("Experimental")).not.toBeInTheDocument();
  });

  it("links to the public model detail page when detailModelKey is set", () => {
    render(<ModelCard model={{ ...EVALUATED_MODEL, detailModelKey: "event_v3" }} />);
    const link = screen.getByRole("link", { name: /View full results & coefficients/ });
    expect(link).toHaveAttribute("href", "/models/event_v3");
  });

  it("does not render the detail link when detailModelKey is absent", () => {
    render(<ModelCard model={EVALUATED_MODEL} />);
    expect(screen.queryByRole("link", { name: /View full results/ })).not.toBeInTheDocument();
  });

  it("renders the evaluated status label distinctly from promoted/training/planned", () => {
    render(<ModelCard model={EVALUATED_MODEL} />);
    const badge = screen.getByText("Evaluated");
    expect(badge).toBeInTheDocument();
    // Evaluated must not reuse promoted's teal styling (that would visually
    // overclaim "shipped to production") or training's amber styling
    // (v3 training is done, this isn't "in progress" either).
    expect(badge.style.color).not.toBe("var(--teal)");
    expect(badge.style.color).not.toBe("var(--amber)");
  });
});
