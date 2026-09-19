import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";

import { DivergingBar } from "@/components/ui/DivergingBar";

describe("DivergingBar", () => {
  it("renders both labels and formatted values", () => {
    render(<DivergingBar left={{ label: "Goals", value: 5 }} right={{ label: "xG", value: 3.42 }} />);
    expect(screen.getByText("Goals")).toBeInTheDocument();
    expect(screen.getByText("xG")).toBeInTheDocument();
    expect(screen.getByText("5.00")).toBeInTheDocument();
    expect(screen.getByText("3.42")).toBeInTheDocument();
  });

  it("scales bar widths relative to the larger value", () => {
    const { container } = render(
      <DivergingBar left={{ label: "Goals", value: 10 }} right={{ label: "xG", value: 5 }} />
    );
    const leftBar = container.querySelector('[data-testid="diverging-bar-left"]') as HTMLElement;
    const rightBar = container.querySelector('[data-testid="diverging-bar-right"]') as HTMLElement;
    expect(leftBar.style.width).toBe("100%");
    expect(rightBar.style.width).toBe("50%");
  });

  it("renders a reference line only when provided", () => {
    const { container: withRef } = render(
      <DivergingBar
        left={{ label: "Goals", value: 5 }}
        right={{ label: "xG", value: 3 }}
        referenceLine={5}
      />
    );
    expect(withRef.querySelector('[data-testid="diverging-bar-reference"]')).toBeInTheDocument();

    const { container: withoutRef } = render(
      <DivergingBar left={{ label: "Goals", value: 5 }} right={{ label: "xG", value: 3 }} />
    );
    expect(withoutRef.querySelector('[data-testid="diverging-bar-reference"]')).not.toBeInTheDocument();
  });

  it("supports a custom value formatter", () => {
    render(
      <DivergingBar
        left={{ label: "Goals", value: 5 }}
        right={{ label: "xG", value: 3 }}
        formatValue={(v) => `${v} shots`}
      />
    );
    expect(screen.getByText("5 shots")).toBeInTheDocument();
  });
});
