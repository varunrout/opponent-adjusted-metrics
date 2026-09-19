// N identical-value "copies" stacking into one inflated total — the
// silver-schema-version duplication root cause. Each copy renders as an
// equal-width segment; the segments visually sum to the total bar below.
export function CopiesDiagram({
  copyLabels,
  perCopyValue,
  entityLabel,
  totalLabel,
  note,
}: {
  copyLabels: string[];
  perCopyValue: number;
  entityLabel: string;
  totalLabel: string;
  note: string;
}) {
  const total = perCopyValue * copyLabels.length;
  const segmentColors = ["var(--teal)", "var(--violet)", "var(--amber)", "var(--home-team)", "var(--away-team)"];

  return (
    <div className="flex flex-col gap-3">
      <div className="flex gap-1" data-testid="copies-diagram-segments">
        {copyLabels.map((label, i) => (
          <div key={label} className="flex-1 flex flex-col items-center gap-1">
            <div
              className="w-full h-8 rounded flex items-center justify-center font-data text-[11px] text-[#0b0e12]"
              style={{ background: segmentColors[i % segmentColors.length] }}
            >
              {perCopyValue.toLocaleString()}
            </div>
            <span className="text-[10px] text-muted text-center leading-snug">{label}</span>
          </div>
        ))}
      </div>

      <div className="text-center text-text2 text-[13px]">+</div>

      <div className="text-center">
        <div className="font-data text-[18px]" style={{ color: "var(--red)" }}>
          {total.toLocaleString()}
        </div>
        <div className="text-[10.5px] text-muted mt-0.5">
          {entityLabel} — {totalLabel}
        </div>
      </div>

      <p className="text-[11px] text-amber m-0" style={{ color: "var(--amber)" }}>
        ⚠ {note}
      </p>
    </div>
  );
}
