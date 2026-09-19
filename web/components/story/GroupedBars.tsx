// Rows of grouped horizontal bars — e.g. "real" vs "shown on site" pairs.
// Bar widths scale relative to the largest value across the whole figure,
// so magnitude is comparable across rows, not just within one.
export function GroupedBars({
  rows,
}: {
  rows: { label: string; bars: { label: string; value: number; color: string }[] }[];
}) {
  const max = Math.max(...rows.flatMap((r) => r.bars.map((b) => b.value)), 1);

  return (
    <div className="flex flex-col gap-4">
      {rows.map((row) => (
        <div key={row.label} data-testid="grouped-bars-row">
          <div className="text-[11.5px] text-text2 mb-1.5">{row.label}</div>
          <div className="flex flex-col gap-1">
            {row.bars.map((bar) => (
              <div key={bar.label} className="flex items-center gap-2">
                <span className="text-[10.5px] text-muted w-24 shrink-0">{bar.label}</span>
                <div className="flex-1 h-2.5 rounded-full overflow-hidden" style={{ background: "var(--card-hi)" }}>
                  <div
                    className="h-full rounded-full"
                    style={{ width: `${(bar.value / max) * 100}%`, background: bar.color }}
                  />
                </div>
                <span className="font-data text-[11px] text-text w-14 text-right shrink-0">{bar.value}</span>
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}
