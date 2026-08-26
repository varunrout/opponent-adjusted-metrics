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
