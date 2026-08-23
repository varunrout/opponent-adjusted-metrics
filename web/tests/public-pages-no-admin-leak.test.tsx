import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import ModelsPage from "@/app/models/page";
import StoriesPage from "@/app/stories/page";

// Models and Stories are guest-visible (see nav-config.ts and
// docs/dashboard_content_ideation.md). Both pages currently render only
// the public registry layer — status badges, tier chips, validation
// metrics, feature-family counts, and story teasers — with no admin
// internals (raw validation logs, version pin/inspect controls,
// promote/retire actions, unpublished drafts) anywhere in the tree, and
// neither page does any role check because there's nothing to gate yet.
// This test is a regression guard: if admin-only controls are ever added
// to either page, they must be gated behind a real role check, not just
// rendered because the nav tab happens to be public.
// Note: deliberately more specific than bare "promote"/"draft" — the
// public "Promoted" status badge is expected content, not a leak.
const FORBIDDEN_SUBSTRINGS = [
  "validation log",
  "promote model",
  "promote to",
  "retire model",
  "confirm promotion",
  "pin version",
  "pin model",
  "unpublished",
  "draft story",
  "internal only",
  "admin only",
];

function assertNoForbiddenContent(container: HTMLElement) {
  const text = container.textContent?.toLowerCase() ?? "";
  for (const phrase of FORBIDDEN_SUBSTRINGS) {
    expect(text).not.toContain(phrase);
  }
}

describe("public pages don't leak admin internals", () => {
  it("Models page renders only the public registry layer", () => {
    const { container } = render(<ModelsPage />);
    assertNoForbiddenContent(container);
    // Sanity check the public layer actually renders as intended. CxG/CxG+
    // are "Evaluated" (real v3 test-set results, honestly compared to the
    // StatsBomb baseline), not "Promoted" — no serving layer exists yet.
    expect(screen.getByText("CxG")).toBeInTheDocument();
    expect(screen.getAllByText("Evaluated").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Test log_loss").length).toBeGreaterThan(0);
    expect(screen.getAllByText(/Trails the StatsBomb xG baseline/).length).toBeGreaterThan(0);
  });

  it("Stories page renders only public teasers, including the new dev-log category", () => {
    const { container } = render(<StoriesPage />);
    assertNoForbiddenContent(container);
    expect(screen.getByText("Dev log")).toBeInTheDocument();
  });
});
