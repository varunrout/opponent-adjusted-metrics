import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";

import { StoryFigureView } from "@/components/story/StoryFigureView";
import type { StoryFigure } from "@/lib/stories-data";

describe("StoryFigureView", () => {
  it("renders a split-calc figure with both branches", () => {
    const figure: StoryFigure = {
      kind: "split-calc",
      afterParagraph: 0,
      title: "Two denominators",
      shared: [{ label: "A", value: "0.30" }],
      branches: [
        { label: "÷ A", result: "15%" },
        { label: "÷ B", result: "17%" },
      ],
    };
    render(<StoryFigureView figure={figure} />);
    expect(screen.getByText("Two denominators")).toBeInTheDocument();
    expect(screen.getAllByTestId("split-calc-branch")).toHaveLength(2);
    expect(screen.getByText("15%")).toBeInTheDocument();
    expect(screen.getByText("17%")).toBeInTheDocument();
  });

  it("renders grouped-bars rows and values", () => {
    const figure: StoryFigure = {
      kind: "grouped-bars",
      afterParagraph: 0,
      title: "Real vs shown",
      rows: [
        {
          label: "Matches",
          bars: [
            { label: "Real", value: 610, color: "var(--teal)" },
            { label: "Shown", value: 1830, color: "var(--red)" },
          ],
        },
      ],
    };
    render(<StoryFigureView figure={figure} />);
    expect(screen.getAllByTestId("grouped-bars-row")).toHaveLength(1);
    expect(screen.getByText("610")).toBeInTheDocument();
    expect(screen.getByText("1830")).toBeInTheDocument();
  });

  it("renders a copies-diagram with per-copy and total values", () => {
    const figure: StoryFigure = {
      kind: "copies-diagram",
      afterParagraph: 0,
      title: "3 copies",
      copyLabels: ["v1", "v1_1", "v1_2"],
      perCopyValue: 610,
      entityLabel: "matches",
      totalLabel: "shown",
      note: "0 of 9 queries filtered",
    };
    render(<StoryFigureView figure={figure} />);
    expect(screen.getAllByText("610")).toHaveLength(3);
    expect(screen.getByText("1,830")).toBeInTheDocument();
    expect(screen.getByText(/0 of 9 queries filtered/)).toBeInTheDocument();
  });

  it("renders a table figure with all rows", () => {
    const figure: StoryFigure = {
      kind: "table",
      afterParagraph: 0,
      title: "Gates",
      columns: ["Gate", "Result"],
      rows: [
        ["Full test suite", "PASS — 198/198"],
        ["Reconciliation", "PASS — 18/18 tables"],
      ],
    };
    render(<StoryFigureView figure={figure} />);
    expect(screen.getByText("Full test suite")).toBeInTheDocument();
    expect(screen.getByText("PASS — 198/198")).toBeInTheDocument();
    expect(screen.getByText("Reconciliation")).toBeInTheDocument();
  });

  it("renders a pass-fail-list with cleared/did-not-clear labels", () => {
    const figure: StoryFigure = {
      kind: "pass-fail-list",
      afterParagraph: 0,
      title: "Univariate screen",
      items: [
        { label: "Score difference", passed: true },
        { label: "Match minute", passed: false },
      ],
    };
    render(<StoryFigureView figure={figure} />);
    expect(screen.getByText("Score difference")).toBeInTheDocument();
    expect(screen.getByText("Cleared")).toBeInTheDocument();
    expect(screen.getByText("Match minute")).toBeInTheDocument();
    expect(screen.getByText("Did not clear")).toBeInTheDocument();
  });

  it("renders an upload-timeline with claimed vs actual columns", () => {
    const figure: StoryFigure = {
      kind: "upload-timeline",
      afterParagraph: 0,
      title: "Claimed vs actual",
      claimed: [{ label: "Marker written first" }],
      actual: [
        { label: "manifest.json uploaded", detail: "12:01:50.152 UTC" },
        { label: "_SUCCESS uploaded last", detail: "12:01:50.215 UTC" },
      ],
    };
    render(<StoryFigureView figure={figure} />);
    expect(screen.getByTestId("upload-timeline-claimed")).toHaveTextContent("Marker written first");
    expect(screen.getByTestId("upload-timeline-actual")).toHaveTextContent("manifest.json uploaded");
    expect(screen.getByTestId("upload-timeline-actual")).toHaveTextContent("12:01:50.215 UTC");
  });
});
