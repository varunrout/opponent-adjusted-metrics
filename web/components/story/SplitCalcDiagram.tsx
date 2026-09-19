// Two shared input numbers, branching through two different calculations to
// two different (both correct) results — e.g. the same log-loss gap divided
// by two different denominators.
export function SplitCalcDiagram({
  shared,
  branches,
}: {
  shared: { label: string; value: string }[];
  branches: { label: string; result: string }[];
}) {
  return (
    <div className="flex flex-col items-center gap-3">
      <div className="flex items-center gap-4 flex-wrap justify-center">
        {shared.map((s) => (
          <div key={s.label} className="text-center">
            <div className="font-data text-[16px] text-text">{s.value}</div>
            <div className="text-[10.5px] text-muted mt-0.5">{s.label}</div>
          </div>
        ))}
      </div>

      <div className="text-text2 text-[14px]">↓</div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 w-full">
        {branches.map((b) => (
          <div
            key={b.label}
            className="flex flex-col items-center gap-1.5 bg-card-hi rounded p-3 border border-border"
            data-testid="split-calc-branch"
          >
            <div className="text-[11px] text-text2 text-center leading-snug">{b.label}</div>
            <div className="font-data text-[20px]" style={{ color: "var(--teal)" }}>
              {b.result}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
