import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { CoefficientForestPlot } from "@/components/models/CoefficientForestPlot";
import type { CxgCoefficientResponse } from "@/lib/types";

function makeCoef(overrides: Partial<CxgCoefficientResponse>): CxgCoefficientResponse {
  return {
    model_key: "event_v3",
    track: "cxg_event",
    feature: "some_feature",
    coefficient: 0.5,
    std_error: null,
    p_value: null,
    ...overrides,
  };
}

describe("CoefficientForestPlot", () => {
  it("excludes the intercept (const) from the plotted rows", () => {
    render(
      <CoefficientForestPlot
        coefficients={[makeCoef({ feature: "const", coefficient: -2.4 }), makeCoef({ feature: "shot_distance" })]}
      />
    );
    expect(screen.queryByText("const")).not.toBeInTheDocument();
    expect(screen.getByText("shot_distance")).toBeInTheDocument();
  });

  it("excludes rows with a null coefficient", () => {
    render(
      <CoefficientForestPlot
        coefficients={[makeCoef({ feature: "null_one", coefficient: null }), makeCoef({ feature: "real_one" })]}
      />
    );
    expect(screen.queryByText("null_one")).not.toBeInTheDocument();
    expect(screen.getByText("real_one")).toBeInTheDocument();
  });

  it("shows a fallback message when nothing is plottable", () => {
    render(<CoefficientForestPlot coefficients={[makeCoef({ feature: "const", coefficient: -2.4 })]} />);
    expect(screen.getByText(/No plottable coefficients/)).toBeInTheDocument();
    expect(screen.queryByTestId("coefficient-forest-plot")).not.toBeInTheDocument();
  });

  it("renders both positive and negative coefficients with their formatted value", () => {
    render(
      <CoefficientForestPlot
        coefficients={[
          makeCoef({ feature: "positive_feature", coefficient: 0.777 }),
          makeCoef({ feature: "negative_feature", coefficient: -0.333 }),
        ]}
      />
    );
    expect(screen.getByText("0.777")).toBeInTheDocument();
    expect(screen.getByText("-0.333")).toBeInTheDocument();
  });
});
