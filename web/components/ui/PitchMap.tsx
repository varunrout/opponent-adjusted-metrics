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
}: {
  shots?: ShotResponse[];
  homeTeamId?: number | null;
}) {
  return (
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
          const radius = radiusForXg(shot.statsbomb_xg);
          return (
            <circle
              key={shot.event_id}
              data-testid="shot-marker"
              cx={shot.location_x as number}
              cy={shot.location_y as number}
              r={radius}
              fill={color}
              fillOpacity={shot.is_goal ? 0.9 : 0.2}
              stroke={shot.is_goal ? "#0b0e12" : color}
              strokeWidth="0.5"
            />
          );
        })}
    </svg>
  );
}
