import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import AboutPage from "@/app/about/page";

describe("AboutPage", () => {
  it('no longer renders the "Not yet built" placeholder', () => {
    render(<AboutPage />);
    expect(screen.queryByText("Not yet built")).not.toBeInTheDocument();
  });

  it("covers all five required sections", () => {
    render(<AboutPage />);
    expect(screen.getByText("What this is")).toBeInTheDocument();
    expect(screen.getByText("Data source")).toBeInTheDocument();
    expect(screen.getByText("Glossary")).toBeInTheDocument();
    expect(screen.getByText("How it's built")).toBeInTheDocument();
    expect(screen.getByText("Honest limitations")).toBeInTheDocument();
  });

  it("defines every metric shown elsewhere in the app", () => {
    render(<AboutPage />);
    expect(screen.getByText(/^xG \(expected goals\)/)).toBeInTheDocument();
    expect(screen.getByText(/^CxG \(context-adjusted expected goals\)/)).toBeInTheDocument();
    expect(screen.getByText("CxG+")).toBeInTheDocument();
    expect(screen.getByText("xG/shot")).toBeInTheDocument();
    expect(screen.getByText(/G−xG/)).toBeInTheDocument();
    expect(screen.getByText("Opponent-adjusted")).toBeInTheDocument();
  });

  it("states limitations plainly, without hedging them away", () => {
    render(<AboutPage />);
    expect(screen.getByText(/not scored live/)).toBeInTheDocument();
    expect(screen.getByText(/trails the StatsBomb xG baseline/)).toBeInTheDocument();
    expect(screen.getAllByText(/oam_serving/).length).toBeGreaterThan(0);
  });
});
