import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { Badge } from "@/components/ui/Badge";

describe("Badge", () => {
  it('renders the "experimental" variant in --amber', () => {
    render(<Badge status="experimental" label="Experimental" />);
    const badge = screen.getByText("Experimental");
    expect(badge.style.color).toBe("var(--amber)");
  });

  it("still supports the existing model-status variants unchanged", () => {
    render(<Badge status="evaluated" label="Evaluated" />);
    const badge = screen.getByText("Evaluated");
    expect(badge.style.color).toBe("var(--text2)");
  });
});
