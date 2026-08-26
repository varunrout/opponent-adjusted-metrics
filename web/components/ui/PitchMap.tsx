import type { ShotResponse } from "@/lib/types";

const HOME_COLOR = "#3B82F6";
const AWAY_COLOR = "#F97316";

function radiusForXg(xg: number | null): number {
  const value = xg ?? 0;
  return Math.min(5.5, Math.max(2, 2 + value * 6));
}

// Ported from docs/dashboard_layout_skeleton.html's .pitch-wrap svg
// (StatsBomb 120x80 coordinate space per the design spec's PitchMap primitive).
// Shot markers are now data-driven: pass real ShotResponse[] instead of the
// old hardcoded mock circles.
export function PitchMap({
  shots = [],
  homeTeamId = null,
  cxgByEventId,
  cxgPlusByEventId,
  sizeBy = "xg",
  showLegend = false,
}: {
  shots?: ShotResponse[];
  homeTeamId?: number | null;
  cxgByEventId?: Record<string, number>;
  cxgPlusByEventId?: Record<string, number>;
  // Display mode only (docs/dashboard_content_spec_v3.md §2.2) — never a
  // refetch. "cxg" sizes dots by the cxg_event track where coverage exists;
  // uncovered shots render at reduced opacity with a dashed stroke rather
  // than falling back to xG (never substitute xG and label it CxG).
  sizeBy?: "xg" | "cxg";
  showLegend?: boolean;
}) {
  return (
    <div>
      <svg viewBox="0 0 120 80" className="w-full h-auto block rounded-md">
      <rect x="0" y="0" width="120" height="80" fill="#123b25" />
      <g stroke="rgba(255,255,255,0.35)" strokeWidth="0.4" fill="none">
        <rect x="0.4" y="0.4" width="119.2" height="79.2" />
        <line x1="60" y1="0" x2="60" y2="80" />
        <circle cx="60" cy="40" r="9.15" />
        <rect x="0" y="18" width="18" height="44" />
        <rect x="102" y="18" width="18" height="44" />
        <rect x="0" y="30" width="6" height="20" />
        <rect x="114" y="30" width="6" height="20" />
      </g>
      {shots
        .filter((shot) => shot.location_x != null && shot.location_y != null)
        .map((shot) => {
          const color = shot.team_id === homeTeamId ? HOME_COLOR : AWAY_COLOR;
          const cxg = cxgByEventId?.[shot.event_id];
          const cxgPlus = cxgPlusByEventId?.[shot.event_id];
          const hasCxg = typeof cxg === "number";
          const hasCxgPlus = typeof cxgPlus === "number";
          const cxgParts: string[] = [];
          if (hasCxg) cxgParts.push(`CxG ${cxg!.toFixed(2)} (8 features)`);
          if (hasCxgPlus) cxgParts.push(`CxG+ ${cxgPlus!.toFixed(2)} (24 features)`);

          const cxgUncovered = sizeBy === "cxg" && !hasCxg;
          const radius = radiusForXg(sizeBy === "cxg" && hasCxg ? cxg! : shot.statsbomb_xg);

          return (
            <circle
              key={shot.event_id}
              data-testid="shot-marker"
              data-cxg-uncovered={cxgUncovered ? "true" : undefined}
              cx={shot.location_x as number}
              cy={shot.location_y as number}
              r={radius}
              fill={color}
              fillOpacity={cxgUncovered ? 0.12 : shot.is_goal ? 0.9 : 0.2}
              stroke={shot.is_goal ? "#0b0e12" : color}
              strokeWidth={cxgParts.length > 0 ? "0.9" : "0.5"}
              strokeDasharray={cxgUncovered ? "1,0.8" : undefined}
            >
              {cxgParts.length > 0 ? (
                <title>
                  {`xG ${(shot.statsbomb_xg ?? 0).toFixed(2)} · ${cxgParts.join(
                    " · "
                  )} — Experimental · limited data & features (v3 test-set only, not scored live)`}
                </title>
              ) : null}
            </circle>
          );
        })}
      </svg>
      {showLegend && (
        <div
          data-testid="pitch-map-legend"
          className="flex flex-wrap items-center gap-x-4 gap-y-1.5 mt-2 text-[11px] text-muted"
        >
          <LegendSwatch color={HOME_COLOR} label="Home" />
          <LegendSwatch color={AWAY_COLOR} label="Away" />
          <span className="flex items-center gap-1.5">
            <svg width="10" height="10" viewBox="0 0 10 10" aria-hidden>
              <circle cx="5" cy="5" r="4" fill="var(--muted)" fillOpacity={0.9} />
            </svg>
            Filled = goal
          </span>
          <span>Dot size = {sizeBy === "cxg" ? "CxG" : "xG"}</span>
        </div>
      )}
    </div>
  );
}

function LegendSwatch({ color, label }: { color: string; label: string }) {
  return (
    <span className="flex items-center gap-1.5">
      <svg width="10" height="10" viewBox="0 0 10 10" aria-hidden>
        <circle cx="5" cy="5" r="4" fill={color} />
      </svg>
      {label}
    </span>
  );
}
