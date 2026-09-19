import { describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import ModelsPage from "@/app/models/page";
import StoriesPage from "@/app/stories/page";

vi.mock("next/navigation", () => ({
  useParams: () => ({ modelKey: "event_v3" }),
}));

vi.mock("@/lib/api", () => ({
  getPublicCxgModelResults: vi.fn().mockResolvedValue([
    {
      model_key: "event_v3",
      track: "cxg_event",
      split: "test",
      model: "v3",
      n: 2427,
      log_loss: 0.3003,
      brier_score: 0.0852,
      roc_auc: 0.7148,
      is_frozen: true,
      is_current: true,
    },
  ]),
  getPublicCxgModelCoefficients: vi.fn().mockResolvedValue([
    { model_key: "event_v3", track: "cxg_event", feature: "const", coefficient: -2.46, std_error: null, p_value: null },
  ]),
}));

import ModelDetailPage from "@/app/models/[modelKey]/page";

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

  it("Model detail page renders no admin gate and no sign-in requirement", async () => {
    const { container } = render(<ModelDetailPage />);
    await waitFor(() => expect(screen.getByText("Model: event_v3")).toBeInTheDocument());
    assertNoForbiddenContent(container);
    expect(container.textContent?.toLowerCase()).not.toContain("sign in");
    expect(container.textContent?.toLowerCase()).not.toContain("admin account");
  });

  it("Stories page renders only public teasers, including the new dev-log category", () => {
    const { container } = render(<StoriesPage />);
    assertNoForbiddenContent(container);
    expect(screen.getAllByText("Dev log").length).toBeGreaterThan(0);
    // CxG v3's honest-comparison entry: headline plus its takeaway both
    // render, and the qualitative disclosure text doesn't trip the
    // forbidden-substring check above. It's intentionally public-facing
    // honest content, not an admin internal.
    expect(
      screen.getByText("CxG v3 against StatsBomb xG: an honest comparison")
    ).toBeInTheDocument();
    expect(
      screen.getByText(
        "CxG v3 beat every version of itself and still lost to StatsBomb on all six metrics. I'm publishing the numbers anyway."
      )
    ).toBeInTheDocument();
    // The negative-result story must stay on the public index: a rejected
    // hypothesis is disclosure, not an internal.
    expect(
      screen.getByText("I built late-game context features and they failed")
    ).toBeInTheDocument();
  });
});
