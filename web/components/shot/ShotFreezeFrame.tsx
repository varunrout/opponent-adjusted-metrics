import { useEffect, useState } from "react";
import type { OpponentContextResponse, ShotFreezeFrameResponse, ShotResponse } from "@/lib/types";
import { getShotFreezeFrame } from "@/lib/api";
import { PitchBackground } from "@/components/ui/PitchBackground";

// StatsBomb 120x80 pitch space — goal mouth posts, matching the six-yard
// box drawn in PitchBackground (x=114-120, y=30-50).
const GOAL_TOP: [number, number] = [120, 36];
const GOAL_BOTTOM: [number, number] = [120, 44];

function polygonPoints(area: number[]): string {
  const points: string[] = [];
  for (let i = 0; i + 1 < area.length; i += 2) {
    points.push(`${area[i]},${area[i + 1]}`);
  }
  return points.join(" ");
}

/**
 * The shot's freeze-frame visual. Real StatsBomb 360 positional data —
 * teammates, opponents, GK — is fetched lazily on open from
 * oam_core.three_sixty_frames/three_sixty_players (166 of 610 matches carry
 * it; a 404 for the rest just means no dots, not an error). Aggregate
 * distances/roles/archetypes still come from `context`
 * (oam_analysis.cxg_analysis_opponent_adjusted_v1) unchanged — this is
 * purely additive, real dots alongside the existing chips. No assist-arrow:
 * that needs the assisting pass event's own x/y, which this table doesn't
 * carry (it's the shot's own frame, not the prior event's).
 */
export function ShotFreezeFrame({
  shot,
  context,
}: {
  shot: ShotResponse;
  context: OpponentContextResponse | null;
}) {
  const x = shot.location_x;
  const y = shot.location_y;
  const hasPosition = x != null && y != null;

  const [frame, setFrame] = useState<ShotFreezeFrameResponse | null>(null);

  useEffect(() => {
    let cancelled = false;
    setFrame(null);
    getShotFreezeFrame(shot.match_id, shot.event_id)
      .then((data) => {
        if (!cancelled) setFrame(data);
      })
      .catch(() => {
        if (!cancelled) setFrame(null);
      });
    return () => {
      cancelled = true;
    };
  }, [shot.match_id, shot.event_id]);

  const players = (frame?.players ?? []).filter((p) => p.x != null && p.y != null);
  const teammates = players.filter((p) => p.teammate === true && p.actor !== true);
  const opponents = players.filter((p) => p.teammate === false && p.keeper !== true);
  const opposingGk = players.filter((p) => p.teammate === false && p.keeper === true);

  // No geometrically-nearest-opponent line: checked against real data (8
  // shots with both a frame and a cxg_analysis_opponent_adjusted_v1 row) —
  // the native-distance-to-nearest-frame-opponent and the model's own
  // nearest_defender_gap routinely disagree (sometimes 3-10x apart, in
  // either direction), so drawing a line and implying it's "the" nearest
  // defender the model used would misrepresent that number. The text chips
  // below stay the only source of truth for it.

  return (
    <div>
      <svg viewBox="0 0 120 80" className="w-full h-auto block rounded-md" data-testid="shot-freeze-frame">
        <PitchBackground />
        {frame && frame.visible_area.length >= 6 && (
          <polygon
            points={polygonPoints(frame.visible_area)}
            fill="none"
            stroke="#fff"
            strokeOpacity={0.18}
            strokeWidth={0.4}
            strokeDasharray="1.2,1"
            data-testid="visible-area"
          />
        )}
        {hasPosition && (
          <>
            <polygon
              points={`${x},${y} ${GOAL_TOP[0]},${GOAL_TOP[1]} ${GOAL_BOTTOM[0]},${GOAL_BOTTOM[1]}`}
              fill="var(--amber)"
              fillOpacity={0.22}
              stroke="var(--amber)"
              strokeOpacity={0.4}
              strokeWidth={0.3}
              data-testid="goal-angle-wedge"
            />

            {teammates.map((p) => (
              <circle
                key={`teammate-${p.ordinal}`}
                cx={p.x as number}
                cy={p.y as number}
                r={1.8}
                fill="var(--home-team)"
                fillOpacity={0.45}
                data-testid="freeze-frame-teammate"
              />
            ))}

            {opponents.map((p) => (
              <circle
                key={`opponent-${p.ordinal}`}
                cx={p.x as number}
                cy={p.y as number}
                r={1.8}
                fill="var(--away-team)"
                fillOpacity={0.45}
                data-testid="freeze-frame-opponent"
              />
            ))}

            {opposingGk.map((p) => (
              <g key={`gk-${p.ordinal}`}>
                <line
                  x1={x as number}
                  y1={y as number}
                  x2={p.x as number}
                  y2={p.y as number}
                  stroke="var(--violet)"
                  strokeOpacity={0.5}
                  strokeWidth={0.3}
                  data-testid="gk-line"
                />
                <circle
                  cx={p.x as number}
                  cy={p.y as number}
                  r={2}
                  fill="var(--violet)"
                  fillOpacity={0.7}
                  stroke="#fff"
                  strokeWidth={0.3}
                  data-testid="freeze-frame-gk"
                />
              </g>
            ))}

            <circle
              cx={x as number}
              cy={y as number}
              r={2.6}
              fill={shot.is_goal ? "var(--green)" : "var(--home-team)"}
              stroke="none"
            />
            <circle
              cx={x as number}
              cy={y as number}
              r={3.6}
              fill="none"
              stroke="#fff"
              strokeWidth={0.5}
              data-testid="shot-selection-ring"
            />
          </>
        )}
      </svg>
      {context && (
        <div className="flex flex-wrap gap-2 mt-2 text-[11px]" data-testid="defender-chips">
          {context.nearest_defender_role && (
            <span className="px-2 py-1 rounded bg-card-hi" style={{ color: "var(--red)" }}>
              Nearest defender: {context.nearest_defender_role}
              {context.nearest_defender_gap != null ? ` · ${context.nearest_defender_gap.toFixed(1)}m` : ""}
            </span>
          )}
          {context.gk_odi != null && (
            <span className="px-2 py-1 rounded bg-card-hi" style={{ color: "var(--violet)" }}>
              GK distance index: {context.gk_odi.toFixed(2)}
            </span>
          )}
        </div>
      )}
    </div>
  );
}
