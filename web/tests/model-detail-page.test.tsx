import { describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import type { CxgCoefficientResponse, CxgModelResultResponse } from "@/lib/types";

vi.mock("next/navigation", () => ({
  useParams: () => ({ modelKey: "event_v3" }),
}));

vi.mock("@/lib/api", () => ({
  getPublicCxgModelResults: vi.fn(),
  getPublicCxgModelCoefficients: vi.fn(),
}));

import { getPublicCxgModelResults, getPublicCxgModelCoefficients } from "@/lib/api";
import ModelDetailPage from "@/app/models/[modelKey]/page";

function makeResult(overrides: Partial<CxgModelResultResponse>): CxgModelResultResponse {
  return {
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
    ...overrides,
  };
}

function makeCoef(overrides: Partial<CxgCoefficientResponse>): CxgCoefficientResponse {
  return {
    model_key: "event_v3",
    track: "cxg_event",
    feature: "shot_x_sb",
    coefficient: 0.77,
    std_error: null,
    p_value: null,
    ...overrides,
  };
}

describe("ModelDetailPage", () => {
  it("renders no admin/sign-in gate — no RoleProvider dependency, no gated content", async () => {
    vi.mocked(getPublicCxgModelResults).mockResolvedValue([
      makeResult({}),
      makeResult({ model: "statsbomb_xg", is_current: true }),
    ]);
    vi.mocked(getPublicCxgModelCoefficients).mockResolvedValue([makeCoef({})]);

    render(<ModelDetailPage />);

    await waitFor(() => expect(screen.getByText("Model: event_v3")).toBeInTheDocument());
    expect(screen.queryByText(/sign in/i)).not.toBeInTheDocument();
  });

  it("renders the results table grouped by track", async () => {
    vi.mocked(getPublicCxgModelResults).mockResolvedValue([
      makeResult({ track: "cxg_event" }),
      makeResult({ track: "cxg_plus", model_key: "plus_v3" }),
    ]);
    vi.mocked(getPublicCxgModelCoefficients).mockResolvedValue([]);

    render(<ModelDetailPage />);

    await waitFor(() => expect(screen.getByText("cxg_event")).toBeInTheDocument());
    expect(screen.getByText("cxg_plus")).toBeInTheDocument();
  });

  it("renders version pills for all 4 real model_keys, linking to their own detail route", async () => {
    vi.mocked(getPublicCxgModelResults).mockResolvedValue([makeResult({})]);
    vi.mocked(getPublicCxgModelCoefficients).mockResolvedValue([]);

    render(<ModelDetailPage />);

    await waitFor(() => expect(screen.getByRole("link", { name: /baseline_v1/ })).toBeInTheDocument());
    expect(screen.getByRole("link", { name: /baseline_v1/ })).toHaveAttribute("href", "/models/baseline_v1");
    expect(screen.getByRole("link", { name: /plus_v2/ })).toHaveAttribute("href", "/models/plus_v2");
    expect(screen.getByRole("link", { name: /plus_v3/ })).toHaveAttribute("href", "/models/plus_v3");
  });

  it("renders coefficient rows and flags when std_error/p_value are entirely unavailable", async () => {
    vi.mocked(getPublicCxgModelResults).mockResolvedValue([makeResult({})]);
    vi.mocked(getPublicCxgModelCoefficients).mockResolvedValue([
      makeCoef({ feature: "const", coefficient: -2.46, std_error: null, p_value: null }),
      makeCoef({ feature: "shot_x_sb", coefficient: 0.77, std_error: null, p_value: null }),
    ]);

    render(<ModelDetailPage />);

    // "shot_x_sb" renders twice — once in the forest plot label, once in
    // the coefficients table's first column.
    await waitFor(() => expect(screen.getAllByText("shot_x_sb").length).toBeGreaterThan(0));
    expect(screen.getByText(/Standard errors and p-values are unavailable/)).toBeInTheDocument();
    // "—" appears for the null std_error/p_value cells rather than 0 or blank.
    expect(screen.getAllByText("—").length).toBeGreaterThan(0);
  });

  it("does not show the unavailable-stats note when std_error is present (baseline_v1-style)", async () => {
    vi.mocked(getPublicCxgModelResults).mockResolvedValue([makeResult({})]);
    vi.mocked(getPublicCxgModelCoefficients).mockResolvedValue([
      makeCoef({ feature: "const", coefficient: -2.4, std_error: 0.04, p_value: 0.0 }),
    ]);

    render(<ModelDetailPage />);

    await waitFor(() => expect(screen.getByText(/1 coefficient row/)).toBeInTheDocument());
    expect(screen.queryByText(/Standard errors and p-values are unavailable/)).not.toBeInTheDocument();
  });

  it("renders the known-caveats card", async () => {
    vi.mocked(getPublicCxgModelResults).mockResolvedValue([makeResult({})]);
    vi.mocked(getPublicCxgModelCoefficients).mockResolvedValue([]);

    render(<ModelDetailPage />);

    await waitFor(() => expect(screen.getByText("Known caveats")).toBeInTheDocument());
    expect(screen.getByText(/Feature-pool asymmetry/)).toBeInTheDocument();
    expect(screen.getByText(/unexplained bimodality/)).toBeInTheDocument();
  });

  it("shows an error state, not a crash, when results fail to load", async () => {
    vi.mocked(getPublicCxgModelResults).mockRejectedValue(new Error("boom"));
    vi.mocked(getPublicCxgModelCoefficients).mockResolvedValue([]);

    render(<ModelDetailPage />);

    await waitFor(() => expect(screen.getByText(/Couldn't load model results/)).toBeInTheDocument());
  });
});
