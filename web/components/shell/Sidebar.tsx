const metricPills = [
  { label: "xG", state: "on" as const },
  { label: "CxG", state: "default" as const },
  { label: "CxA", state: "disabled" as const },
  { label: "CxT", state: "disabled" as const },
];

export function Sidebar() {
  return (
    <aside className="w-[220px] flex-shrink-0 bg-surface border-r border-border px-4 py-[18px]">
      <FilterGroup label="Competition">
        <select className="w-full bg-card border border-border text-text rounded-lg px-2.5 py-[7px] text-[12.5px]">
          <option>Premier League</option>
          <option>FIFA World Cup</option>
          <option>UEFA Euro</option>
        </select>
      </FilterGroup>

      <FilterGroup label="Season">
        <select className="w-full bg-card border border-border text-text rounded-lg px-2.5 py-[7px] text-[12.5px]">
          <option>2015/2016</option>
        </select>
      </FilterGroup>

      <FilterGroup label="Team">
        <select className="w-full bg-card border border-border text-text rounded-lg px-2.5 py-[7px] text-[12.5px]">
          <option>All teams</option>
        </select>
      </FilterGroup>

      <FilterGroup label="Metric">
        <div className="flex flex-col gap-1.5">
          {metricPills.map((pill) => (
            <div
              key={pill.label}
              className={[
                "flex items-center justify-between px-2.5 py-[7px] rounded-lg border border-border text-[12.5px] text-text2",
                pill.state === "on" ? "border-teal text-text bg-teal/[0.08]" : "",
                pill.state === "disabled" ? "opacity-45 cursor-not-allowed" : "cursor-pointer",
              ].join(" ")}
            >
              <span>{pill.label}</span>
              {pill.state === "disabled" && (
                <span className="text-[9.5px] bg-card-hi text-muted px-1.5 py-px rounded">soon</span>
              )}
            </div>
          ))}
        </div>
      </FilterGroup>
    </aside>
  );
}

function FilterGroup({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="mb-5">
      <label className="block text-[11px] text-muted mb-1.5 tracking-wide">{label}</label>
      {children}
    </div>
  );
}
