import type { ShotResponse } from "@/lib/types";

const WIDTH = 320;
const HEIGHT = 100;
const PAD_LEFT = 28;
const PAD_BOTTOM = 16;
const MAX_MINUTE = 100; // covers regular + stoppage time for both halves

function cumulativePoints(shots: ShotResponse[], teamId: number | null): [number, number][] {
  const teamShots = shots
    .filter((s) => s.team_id === teamId && s.minute != null)
    .sort((a, b) => (a.minute ?? 0) - (b.minute ?? 0));

  const points: [number, number][] = [[0, 0]];
  let cumulative = 0;
  for (const shot of teamShots) {
    cumulative += shot.statsbomb_xg ?? 0;
    points.push([shot.minute ?? 0, cumulative]);
  }
  points.push([MAX_MINUTE, cumulative]);
  return points;
}

function toPath(points: [number, number][], maxY: number): string {
  const plotWidth = WIDTH - PAD_LEFT;
  const plotHeight = HEIGHT - PAD_BOTTOM;
  const scaleX = (minute: number) => PAD_LEFT + (minute / MAX_MINUTE) * plotWidth;
  const scaleY = (value: number) => plotHeight - (value / (maxY || 1)) * plotHeight;

  // Step function: hold the previous value until the next shot's minute.
  let d = `M ${scaleX(points[0][0])} ${scaleY(points[0][1])}`;
  for (let i = 1; i < points.length; i++) {
    const [prevMinute, prevValue] = points[i - 1];
    const [minute, value] = points[i];
    d += ` L ${scaleX(minute)} ${scaleY(prevValue)}`;
    if (value !== prevValue) {
      d += ` L ${scaleX(minute)} ${scaleY(value)}`;
    }
  }
  return d;
}

/**
 * Cumulative xG step-function timeline for both teams over the match.
 * Real per-shot minute + xG, no smoothing or interpolation.
 */
export function XgTimeline({
  shots,
  homeTeamId,
  awayTeamId,
  homeLabel = "Home",
  awayLabel = "Away",
}: {
  shots: ShotResponse[];
  homeTeamId: number | null;
  awayTeamId: number | null;
  homeLabel?: string;
  awayLabel?: string;
}) {
  const homePoints = cumulativePoints(shots, homeTeamId);
  const awayPoints = cumulativePoints(shots, awayTeamId);
  const maxY = Math.max(homePoints[homePoints.length - 1][1], awayPoints[awayPoints.length - 1][1], 0.1);

  return (
    <div>
      <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} className="w-full h-auto block" data-testid="xg-timeline">
        <line
          x1={PAD_LEFT}
          y1={HEIGHT - PAD_BOTTOM}
          x2={WIDTH}
          y2={HEIGHT - PAD_BOTTOM}
          stroke="var(--border)"
          strokeWidth={1}
        />
        <line
          x1={PAD_LEFT}
          y1={0}
          x2={PAD_LEFT}
          y2={HEIGHT - PAD_BOTTOM}
          stroke="var(--border)"
          strokeWidth={1}
        />
        <text x={0} y={6} fontSize={7} fill="var(--muted)">
          {maxY.toFixed(1)}
        </text>
        <text x={0} y={HEIGHT - PAD_BOTTOM} fontSize={7} fill="var(--muted)">
          0
        </text>
        <path d={toPath(homePoints, maxY)} fill="none" stroke="var(--home-team)" strokeWidth={1.5} />
        <path d={toPath(awayPoints, maxY)} fill="none" stroke="var(--away-team)" strokeWidth={1.5} />
      </svg>
      <div className="flex items-center gap-4 mt-1.5 text-[11px] text-text2">
        <span className="flex items-center gap-1.5">
          <span className="inline-block w-2.5 h-2.5 rounded-full" style={{ background: "var(--home-team)" }} />
          {homeLabel}
        </span>
        <span className="flex items-center gap-1.5">
          <span className="inline-block w-2.5 h-2.5 rounded-full" style={{ background: "var(--away-team)" }} />
          {awayLabel}
        </span>
      </div>
    </div>
  );
}
