// Shared "two values compared side by side" primitive — a labelled pair of
// bars growing from a shared centre, optionally with a reference line
// marking a third value (e.g. actual goals against an xG/CxG bar). Used by
// the Matches list score column, Players list/detail G−xG and covered-shot
// comparisons, and Teams detail's attack/defence rows — one implementation
// instead of five ad-hoc ones.
export function DivergingBar({
  left,
  right,
  referenceLine,
  formatValue = (v) => v.toFixed(2),
}: {
  left: { label: string; value: number; color?: string };
  right: { label: string; value: number; color?: string };
  // A value (in the same units as left/right) marked on both sides as a
  // fixed reference point — e.g. actual goals scored, plotted against xG
  // and CxG bars so the viewer can see over/under-performance at a glance.
  referenceLine?: number;
  formatValue?: (value: number) => string;
}) {
  const leftColor = left.color ?? "var(--home-team)";
  const rightColor = right.color ?? "var(--away-team)";
  const max = Math.max(left.value, right.value, referenceLine ?? 0, 0.0001);
  const leftPct = Math.min(100, (left.value / max) * 100);
  const rightPct = Math.min(100, (right.value / max) * 100);
  const referencePct = referenceLine != null ? Math.min(100, (referenceLine / max) * 100) : null;

  return (
    <div className="flex flex-col gap-1" data-testid="diverging-bar">
      <div className="flex items-center justify-between text-[11px] text-text2">
        <span>{left.label}</span>
        <span className="font-data text-text">{formatValue(left.value)}</span>
      </div>
      <div className="relative h-2 rounded-full overflow-hidden bg-card-hi" style={{ background: "var(--card-hi)" }}>
        <div
          className="h-full rounded-full"
          style={{ width: `${leftPct}%`, background: leftColor }}
          data-testid="diverging-bar-left"
        />
        {referencePct != null && (
          <div
            className="absolute top-0 bottom-0 w-[2px]"
            style={{ left: `${referencePct}%`, background: "var(--green)" }}
            data-testid="diverging-bar-reference"
            title={`Reference: ${formatValue(referenceLine as number)}`}
          />
        )}
      </div>
      <div className="relative h-2 rounded-full overflow-hidden" style={{ background: "var(--card-hi)" }}>
        <div
          className="h-full rounded-full"
          style={{ width: `${rightPct}%`, background: rightColor }}
          data-testid="diverging-bar-right"
        />
        {referencePct != null && (
          <div
            className="absolute top-0 bottom-0 w-[2px]"
            style={{ left: `${referencePct}%`, background: "var(--green)" }}
          />
        )}
      </div>
      <div className="flex items-center justify-between text-[11px] text-text2">
        <span>{right.label}</span>
        <span className="font-data text-text">{formatValue(right.value)}</span>
      </div>
    </div>
  );
}
