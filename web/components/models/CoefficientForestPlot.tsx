import type { CxgCoefficientResponse } from "@/lib/types";

// Signed horizontal bar per coefficient, centered at zero — a forest plot
// without the confidence-interval whiskers (std_error/p_value are null for
// 3 of the 4 real model versions; whiskers would misrepresent those rows as
// having a computed interval when none exists). Bar direction/color follows
// the same green="increases" / red="decreases" outcome-direction convention
// used elsewhere in the app (e.g. Players/Teams G-xG), since a coefficient's
// sign literally is the direction of its effect on goal probability.
export function CoefficientForestPlot({ coefficients }: { coefficients: CxgCoefficientResponse[] }) {
  const plottable = coefficients.filter((c) => c.coefficient != null && c.feature !== "const");
  if (plottable.length === 0) {
    return <p className="text-[12px] text-muted m-0">No plottable coefficients (excluding the intercept).</p>;
  }
  const maxAbs = Math.max(...plottable.map((c) => Math.abs(c.coefficient as number)), 0.0001);

  return (
    <div className="flex flex-col gap-2" data-testid="coefficient-forest-plot">
      {plottable.map((c) => {
        const value = c.coefficient as number;
        const pct = (Math.abs(value) / maxAbs) * 50;
        const positive = value >= 0;
        return (
          <div key={c.feature} className="flex items-center gap-2 text-[11px]">
            <span className="w-40 shrink-0 truncate text-text2 font-data" title={c.feature}>
              {c.feature}
            </span>
            <div className="flex-1 h-3 relative" style={{ background: "var(--card-hi)" }}>
              <div className="absolute top-0 bottom-0 left-1/2 w-px" style={{ background: "var(--border)" }} />
              <div
                className="absolute top-0 bottom-0"
                style={{
                  left: positive ? "50%" : `${50 - pct}%`,
                  width: `${pct}%`,
                  background: positive ? "var(--green)" : "var(--red)",
                }}
              />
            </div>
            <span className="w-14 shrink-0 text-right font-data text-text">{value.toFixed(3)}</span>
          </div>
        );
      })}
    </div>
  );
}
