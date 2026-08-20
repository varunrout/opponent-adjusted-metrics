import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { TeamLink, PlayerLink } from "@/components/ui/EntityLink";

describe("cross-linking helpers", () => {
  it("TeamLink renders a link to /teams/[id] when teamId is known", () => {
    render(<TeamLink teamId={217} name="Barcelona" />);
    const link = screen.getByRole("link", { name: "Barcelona" });
    expect(link).toHaveAttribute("href", "/teams/217");
  });

  it("TeamLink falls back to plain text when teamId is null", () => {
    render(<TeamLink teamId={null} name="Unknown team" />);
    expect(screen.queryByRole("link")).not.toBeInTheDocument();
    expect(screen.getByText("Unknown team")).toBeInTheDocument();
  });

  it("PlayerLink renders a link to /players/[id] when playerId is known", () => {
    render(<PlayerLink playerId={3010} name="Kylian Mbappé" />);
    const link = screen.getByRole("link", { name: "Kylian Mbappé" });
    expect(link).toHaveAttribute("href", "/players/3010");
  });

  it("PlayerLink falls back to plain text when playerId is null", () => {
    render(<PlayerLink playerId={null} name="Unknown" />);
    expect(screen.queryByRole("link")).not.toBeInTheDocument();
  });
});
