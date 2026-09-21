import { median, quadrantMetricN, quadrantMetricLabel, type QuadrantMetricKey } from "@/lib/analysis-helpers";
import type { PlayerSeasonQuadrantRow } from "@/lib/types";

const WIDTH = 520;
const HEIGHT = 360;
const PAD = 40;

// A player-season's dot is dimmed below this combined (min of the two
// metrics' own) sample size — a flag, not a filter: the point still plots
// and its exact n is always in the tooltip, per this task's own "show n
// alongside" instruction (the CxA+ "Experimental" badge precedent flags
// rather than silently drops low-n data, and there's no existing per-point
// low-n rule in this codebase to reuse, so this dimming threshold is this
// component's own new, explicit decision — not inherited from elsewhere).
const LOW_N_THRESHOLD = 5;

/**
 * Track B's quadrant scatter (Hard gate 2): one dot per player-season,
 * axes crossed at the league median of whichever two metrics are selected
 * (never a fixed 0/0 origin — the design spec's own "axes crossed at league
 * median" instruction). Caller owns X/Y metric selection (the "build-your-
 * own" part) and passes the already-fetched row set.
 *
 * A row is plotted only when BOTH selected metrics are non-null for it —
 * absent, not zero, for a player-season with zero coverage on one axis (see
 * PlayerSeasonQuadrantRow's own null-vs-zero contract). Rows with only one
 * axis covered are silently left off THIS chart, not shown at a fabricated
 * zero position; they're still real, visible elsewhere via their own `_n`
 * counts if a caller wants to surface them separately.
 */
export function QuadrantScatter({
  rows,
  xMetric,
  yMetric,
}: {
  rows: PlayerSeasonQuadrantRow[];
  xMetric: QuadrantMetricKey;
  yMetric: QuadrantMetricKey;
}) {
  const points = rows
    .map((row) => ({
      row,
      x: row[xMetric],
      y: row[yMetric],
      nX: quadrantMetricN(row, xMetric),
      nY: quadrantMetricN(row, yMetric),
    }))
    .filter((p): p is typeof p & { x: number; y: number } => p.x != null && p.y != null);

  if (points.length === 0) {
    return (
      <p className="text-[12.5px] text-muted m-0">
        No player-seasons have both {quadrantMetricLabel(xMetric)} and {quadrantMetricLabel(yMetric)} coverage in
        this scope.
      </p>
    );
  }

  const xValues = points.map((p) => p.x);
  const yValues = points.map((p) => p.y);
  const xMedian = median(xValues);
  const yMedian = median(yValues);

  const xMin = Math.min(...xValues);
  const xMax = Math.max(...xValues);
  const yMin = Math.min(...yValues);
  const yMax = Math.max(...yValues);
  // Pad the domain 8% each side so median-adjacent points aren't stuck on
  // the plot edge, and guard against a degenerate single-value domain.
  const xPad = (xMax - xMin) * 0.08 || Math.abs(xMax) * 0.08 || 0.01;
  const yPad = (yMax - yMin) * 0.08 || Math.abs(yMax) * 0.08 || 0.01;
  const xDomain: [number, number] = [xMin - xPad, xMax + xPad];
  const yDomain: [number, number] = [yMin - yPad, yMax + yPad];

  const plotW = WIDTH - PAD * 2;
  const plotH = HEIGHT - PAD * 2;
  const scaleX = (v: number) => PAD + ((v - xDomain[0]) / (xDomain[1] - xDomain[0] || 1)) * plotW;
  const scaleY = (v: number) => HEIGHT - PAD - ((v - yDomain[0]) / (yDomain[1] - yDomain[0] || 1)) * plotH;

  const lowNCount = points.filter((p) => Math.min(p.nX, p.nY) < LOW_N_THRESHOLD).length;

  return (
    <div>
      <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} className="w-full h-auto block" data-testid="quadrant-scatter">
        {/* Plot border */}
        <rect x={PAD} y={PAD} width={plotW} height={plotH} fill="none" stroke="var(--border)" strokeWidth={1} />

        {/* Median cross-hairs, dashed to read as reference lines, not axes */}
        <line
          x1={scaleX(xMedian)}
          y1={PAD}
          x2={scaleX(xMedian)}
          y2={HEIGHT - PAD}
          stroke="var(--muted)"
          strokeWidth={1}
          strokeDasharray="3,3"
        />
        <line
          x1={PAD}
          y1={scaleY(yMedian)}
          x2={WIDTH - PAD}
          y2={scaleY(yMedian)}
          stroke="var(--muted)"
          strokeWidth={1}
          strokeDasharray="3,3"
        />

        {points.map((p) => {
          const isLowN = Math.min(p.nX, p.nY) < LOW_N_THRESHOLD;
          return (
            <circle
              key={`${p.row.player_id}-${p.row.competition_id}-${p.row.season_id}`}
              cx={scaleX(p.x)}
              cy={scaleY(p.y)}
              r={3.5}
              fill="var(--teal)"
              fillOpacity={isLowN ? 0.28 : 0.85}
              data-testid="quadrant-scatter-point"
            >
              <title>
                {(p.row.player_name ?? "Unknown player") +
                  ` — ${quadrantMetricLabel(xMetric)}: ${p.x.toFixed(3)} (n=${p.nX}), ` +
                  `${quadrantMetricLabel(yMetric)}: ${p.y.toFixed(3)} (n=${p.nY})`}
              </title>
            </circle>
          );
        })}

        <text x={PAD} y={HEIGHT - PAD + 14} fontSize={9} fill="var(--text2)">
          {quadrantMetricLabel(xMetric)} →
        </text>
        <text
          x={PAD - 8}
          y={PAD - 8}
          fontSize={9}
          fill="var(--text2)"
          textAnchor="start"
        >
          ↑ {quadrantMetricLabel(yMetric)}
        </text>
      </svg>
      <p className="text-[11px] text-muted mt-1.5 mb-0">
        {points.length} player-season{points.length === 1 ? "" : "s"} plotted (both metrics covered, test split
        only). Dashed lines: league median ({xMedian.toFixed(3)} / {yMedian.toFixed(3)}).{" "}
        {lowNCount > 0 &&
          `${lowNCount} point${lowNCount === 1 ? "" : "s"} shown faded — under ${LOW_N_THRESHOLD} covered events on at least one axis; hover any point for its exact n.`}
      </p>
    </div>
  );
}
