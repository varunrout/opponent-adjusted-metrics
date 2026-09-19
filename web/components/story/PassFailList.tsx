// A list of candidates with a pass/fail pill each — e.g. a univariate
// screening pass. Uses the site's green/red outcome-semantic tokens,
// matching Players/Teams' G-xG coloring rather than inventing a new scheme.
export function PassFailList({ items }: { items: { label: string; passed: boolean }[] }) {
  return (
    <div className="flex flex-col gap-1.5" data-testid="pass-fail-list">
      {items.map((item) => (
        <div
          key={item.label}
          className="flex items-center justify-between gap-3 py-1.5 border-b border-border last:border-b-0"
        >
          <span className="text-[12px] text-text2">{item.label}</span>
          <span
            className="text-[10px] px-2 py-0.5 rounded font-medium shrink-0"
            style={{
              background: item.passed ? "rgba(34,197,94,.15)" : "rgba(239,68,68,.15)",
              color: item.passed ? "var(--green)" : "var(--red)",
            }}
          >
            {item.passed ? "Cleared" : "Did not clear"}
          </span>
        </div>
      ))}
    </div>
  );
}
