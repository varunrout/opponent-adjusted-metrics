"use client";

import { useMemo } from "react";
import { useMatchFilter } from "@/components/shell/MatchFilterProvider";

const metricPills = [
  { label: "xG", state: "on" as const },
  { label: "CxG", state: "default" as const },
  { label: "CxA", state: "disabled" as const },
  { label: "CxT", state: "disabled" as const },
];

export function Sidebar() {
  const { competitions, competitionId, seasonId, setCompetitionId, setSeasonId } = useMatchFilter();

  // Unique competitions by competition_id (a competition can appear once per season row).
  const uniqueCompetitions = useMemo(() => {
    const seen = new Map<number, string>();
    for (const c of competitions) {
      if (!seen.has(c.competition_id)) {
        seen.set(c.competition_id, c.competition_name ?? `Competition ${c.competition_id}`);
      }
    }
    return Array.from(seen.entries()).map(([competition_id, competition_name]) => ({
      competition_id,
      competition_name,
    }));
  }, [competitions]);

  const seasonsForCompetition = useMemo(() => {
    if (competitionId == null) return competitions;
    return competitions.filter((c) => c.competition_id === competitionId);
  }, [competitions, competitionId]);

  return (
    <aside className="w-[220px] flex-shrink-0 bg-surface border-r border-border px-4 py-[18px]">
      <FilterGroup label="Competition">
        <select
          className="w-full bg-card border border-border text-text rounded-lg px-2.5 py-[7px] text-[12.5px]"
          value={competitionId ?? ""}
          onChange={(e) => setCompetitionId(e.target.value === "" ? null : Number(e.target.value))}
        >
          <option value="">All competitions</option>
          {uniqueCompetitions.map((c) => (
            <option key={c.competition_id} value={c.competition_id}>
              {c.competition_name}
            </option>
          ))}
        </select>
      </FilterGroup>

      <FilterGroup label="Season">
        <select
          className="w-full bg-card border border-border text-text rounded-lg px-2.5 py-[7px] text-[12.5px]"
          value={seasonId ?? ""}
          onChange={(e) => setSeasonId(e.target.value === "" ? null : Number(e.target.value))}
          disabled={seasonsForCompetition.length === 0}
        >
          <option value="">All seasons</option>
          {seasonsForCompetition.map((c) => (
            <option key={c.season_id} value={c.season_id}>
              {c.season_name}
            </option>
          ))}
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
