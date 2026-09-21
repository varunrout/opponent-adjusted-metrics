"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import { Card } from "@/components/ui/Card";
import { MetricTile } from "@/components/ui/MetricTile";
import { PitchMap } from "@/components/ui/PitchMap";
import { Badge } from "@/components/ui/Badge";
import { Skeleton } from "@/components/ui/Skeleton";
import { DivergingBar } from "@/components/ui/DivergingBar";
import { ShotDetailModal } from "@/components/shot/ShotDetailModal";
import { useMatchFilter } from "@/components/shell/MatchFilterProvider";
import {
  getPlayerShots,
  getCxgCoverage,
  getCxaCoverageByShot,
  getShotOpponentContext,
  getPlayers,
  getMatches,
} from "@/lib/api";
import { summarizeShots } from "@/lib/shot-summary";
import { describeCxgCoverage } from "@/lib/analysis-helpers";
import type {
  CxaShotCoverageValues,
  OpponentContextResponse,
  PlayerSeasonResponse,
  ShotResponse,
} from "@/lib/types";

export default function PlayerDetailPage() {
  const params = useParams<{ playerId: string }>();
  const playerId = params.playerId;
  const { competitionId, seasonId, metricMode, cxgScopeOnly } = useMatchFilter();

  const [shots, setShots] = useState<ShotResponse[]>([]);
  const [cxgByEventId, setCxgByEventId] = useState<Record<string, number>>({});
  const [cxgPlusByEventId, setCxgPlusByEventId] = useState<Record<string, number>>({});
  const [cxaByEventId, setCxaByEventId] = useState<Record<string, CxaShotCoverageValues>>({});
  const [cxaPlusByEventId, setCxaPlusByEventId] = useState<Record<string, CxaShotCoverageValues>>({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);
  const [selectedShot, setSelectedShot] = useState<ShotResponse | null>(null);
  const [opponentContext, setOpponentContext] = useState<OpponentContextResponse[]>([]);
  const [allPlayers, setAllPlayers] = useState<PlayerSeasonResponse[]>([]);
  const [coveredMatchIds, setCoveredMatchIds] = useState<Set<number> | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(false);
    setCxgByEventId({});
    setCxgPlusByEventId({});
    setCxaByEventId({});
    setCxaPlusByEventId({});

    getPlayerShots(playerId, { competition_id: competitionId, season_id: seasonId })
      .then((data) => {
        if (cancelled) return;
        setShots(data);

        const eventIds = data.map((s) => s.event_id);

        getCxgCoverage(eventIds, "cxg_event")
          .then((coverage) => {
            if (!cancelled) setCxgByEventId(coverage.values);
          })
          .catch(() => {
            if (!cancelled) setCxgByEventId({});
          });

        getCxgCoverage(eventIds, "cxg_plus")
          .then((coverage) => {
            if (!cancelled) setCxgPlusByEventId(coverage.values);
          })
          .catch(() => {
            if (!cancelled) setCxgPlusByEventId({});
          });

        getShotOpponentContext(eventIds)
          .then((rows) => {
            if (!cancelled) setOpponentContext(rows);
          })
          .catch(() => {
            if (!cancelled) setOpponentContext([]);
          });

        getCxaCoverageByShot(eventIds, "event")
          .then((coverage) => {
            if (!cancelled) setCxaByEventId(coverage.values);
          })
          .catch(() => {
            if (!cancelled) setCxaByEventId({});
          });

        getCxaCoverageByShot(eventIds, "plus")
          .then((coverage) => {
            if (!cancelled) setCxaPlusByEventId(coverage.values);
          })
          .catch(() => {
            if (!cancelled) setCxaPlusByEventId({});
          });
      })
      .catch(() => {
        if (!cancelled) setError(true);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [playerId, competitionId, seasonId]);

  useEffect(() => {
    let cancelled = false;
    getPlayers({ competition_id: competitionId, season_id: seasonId })
      .then((data) => {
        if (!cancelled) setAllPlayers(data);
      })
      .catch(() => {
        if (!cancelled) setAllPlayers([]);
      });
    return () => {
      cancelled = true;
    };
  }, [competitionId, seasonId]);

  // Same "CxG matches only" scope as the Matches page — match_status_360
  // === "available" is the real per-match signal for 360 coverage, already
  // returned by /v1/matches. No separate endpoint (there isn't one).
  useEffect(() => {
    let cancelled = false;
    if (!cxgScopeOnly) {
      setCoveredMatchIds(null);
      return;
    }
    getMatches({})
      .then((rows) => {
        if (!cancelled) {
          setCoveredMatchIds(
            new Set(rows.filter((r) => r.match_status_360 === "available").map((r) => r.match_id))
          );
        }
      })
      .catch(() => {
        if (!cancelled) setCoveredMatchIds(new Set());
      });
    return () => {
      cancelled = true;
    };
  }, [cxgScopeOnly]);

  if (loading) {
    return (
      <section>
        <Skeleton style={{ height: 32, width: "50%", marginBottom: 18 }} />
        <Skeleton style={{ height: 240 }} />
      </section>
    );
  }

  if (error) {
    return (
      <section>
        <Card>
          <p className="text-[12.5px] text-muted m-0">Couldn&apos;t load this player. Try again shortly.</p>
        </Card>
      </section>
    );
  }

  const firstShot = shots[0];
  const playerName = firstShot?.player_name ?? "Unknown player";
  const teamId = firstShot?.team_id ?? null;

  // "CxG matches only" scopes every shot-driven number on this page to
  // matches carrying 360 coverage — the shot map, tiles, covered-shots
  // comparison, and archetype breakdown all derive from this, not raw `shots`.
  const scopedShots = coveredMatchIds ? shots.filter((s) => coveredMatchIds.has(s.match_id)) : shots;

  const summary = summarizeShots(scopedShots);

  const coveredShots = scopedShots.filter((s) => cxgByEventId[s.event_id] != null);
  const coveredGoals = coveredShots.filter((s) => s.is_goal).length;
  const coveredXg = coveredShots.reduce((sum, s) => sum + (s.statsbomb_xg ?? 0), 0);
  const coveredCxg = coveredShots.reduce((sum, s) => sum + (cxgByEventId[s.event_id] ?? 0), 0);

  const archetypeBreakdown = deriveArchetypeBreakdown(scopedShots, opponentContext);

  const percentiles = derivePercentiles(allPlayers, Number(playerId));

  return (
    <section>
      <div className="mb-[18px]">
        <h1 className="text-lg font-semibold m-0">{playerName}</h1>
        <div className="text-[12.5px] text-muted mt-1">Season shot record</div>
      </div>

      <div className="grid grid-cols-3 gap-3 mb-[18px]">
        <MetricTile label="Shots" value={String(summary.shots)} />
        <MetricTile label="Goals" value={String(summary.goals)} />
        <MetricTile label="Total xG" value={summary.totalXg.toFixed(2)} />
      </div>

      <Card title="Shot map">
        {scopedShots.length === 0 ? (
          <p className="text-[12.5px] text-muted m-0">No shots recorded for this player in the current filters.</p>
        ) : (
          <>
            <PitchMap
              shots={scopedShots}
              homeTeamId={teamId}
              cxgByEventId={cxgByEventId}
              cxgPlusByEventId={cxgPlusByEventId}
              sizeBy={metricMode}
              showLegend
              onShotClick={setSelectedShot}
            />
            {(() => {
              const captions = [
                describeCxgCoverage(scopedShots.length, Object.keys(cxgByEventId).length, "CxG"),
                describeCxgCoverage(scopedShots.length, Object.keys(cxgPlusByEventId).length, "CxG+"),
              ].filter(Boolean);
              return captions.length > 0 ? (
                <div className="flex items-center gap-2 mt-2">
                  <Badge status="experimental" label="Experimental" />
                  <p className="text-[11.5px] text-muted m-0">{captions.join(" · ")}</p>
                </div>
              ) : null;
            })()}
          </>
        )}
      </Card>

      {coveredShots.length > 0 && (
        <Card title="Is he actually good, or just facing soft defenses?" className="mt-4">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <DivergingBar
              left={{ label: "Goals", value: coveredGoals, color: "var(--green)" }}
              right={{ label: "Total xG", value: coveredXg, color: "var(--muted)" }}
              referenceLine={coveredGoals}
            />
            <DivergingBar
              left={{ label: "Goals", value: coveredGoals, color: "var(--green)" }}
              right={{ label: "Total CxG", value: coveredCxg, color: "var(--teal)" }}
              referenceLine={coveredGoals}
            />
          </div>
          <p className="text-[11px] text-muted mt-2 mb-0">
            Scoped to this player&apos;s {coveredShots.length} CxG-covered shot
            {coveredShots.length === 1 ? "" : "s"} only.
          </p>
        </Card>
      )}

      {archetypeBreakdown.length > 0 && (
        <Card title="Shots by defender-style archetype" className="mt-4">
          <ArchetypeBreakdownList breakdown={archetypeBreakdown} />
        </Card>
      )}

      {percentiles && (
        <Card title="Percentile vs the full players list" className="mt-4">
          <div className="grid grid-cols-2 gap-4 text-center">
            <div>
              <div className="text-[24px] font-data" style={{ color: "var(--teal)" }}>
                {percentiles.xgPerShot}
                <span className="text-[13px] text-muted">th</span>
              </div>
              <div className="text-[11px] text-text2 mt-1">xG/shot percentile</div>
            </div>
            <div>
              <div className="text-[24px] font-data" style={{ color: "var(--teal)" }}>
                {percentiles.goalsMinusXg}
                <span className="text-[13px] text-muted">th</span>
              </div>
              <div className="text-[11px] text-text2 mt-1">G−xG percentile</div>
            </div>
          </div>
        </Card>
      )}

      <ShotDetailModal
        shot={selectedShot}
        open={selectedShot != null}
        onClose={() => setSelectedShot(null)}
        cxg={selectedShot ? cxgByEventId[selectedShot.event_id] : undefined}
        cxgPlus={selectedShot ? cxgPlusByEventId[selectedShot.event_id] : undefined}
        cxaEvent={selectedShot ? cxaByEventId[selectedShot.event_id] : undefined}
        cxaPlus={selectedShot ? cxaPlusByEventId[selectedShot.event_id] : undefined}
      />
    </section>
  );
}

type ArchetypeRow = { archetype: string; shots: number; goals: number };

function deriveArchetypeBreakdown(
  shots: ShotResponse[],
  context: OpponentContextResponse[]
): ArchetypeRow[] {
  const contextByEventId = new Map(context.map((c) => [c.event_id, c]));
  const byArchetype = new Map<string, ArchetypeRow>();
  for (const shot of shots) {
    const archetype = contextByEventId.get(shot.event_id)?.nearest_defender_style_archetype;
    if (!archetype) continue;
    const row = byArchetype.get(archetype) ?? { archetype, shots: 0, goals: 0 };
    row.shots += 1;
    if (shot.is_goal) row.goals += 1;
    byArchetype.set(archetype, row);
  }
  return Array.from(byArchetype.values()).sort((a, b) => b.shots - a.shots);
}

function ArchetypeBreakdownList({ breakdown }: { breakdown: ArchetypeRow[] }) {
  const maxShots = Math.max(...breakdown.map((row) => row.shots), 1);
  return (
    <div className="flex flex-col gap-2">
      {breakdown.map((row) => (
        <div key={row.archetype} className="flex flex-col gap-1">
          <div className="flex items-center justify-between text-[11.5px]">
            <span className="text-text">{row.archetype}</span>
            <span className="font-data text-text2">
              {row.shots} shot{row.shots === 1 ? "" : "s"} · {row.goals} goal{row.goals === 1 ? "" : "s"}
            </span>
          </div>
          <div className="h-2 rounded-full overflow-hidden" style={{ background: "var(--card-hi)" }}>
            <div
              className="h-full rounded-full"
              style={{ width: `${(row.shots / maxShots) * 100}%`, background: "var(--violet)" }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}

function percentileRank(values: number[], value: number): number {
  if (values.length === 0) return 0;
  const below = values.filter((v) => v < value).length;
  return Math.round((below / values.length) * 100);
}

function derivePercentiles(
  players: PlayerSeasonResponse[],
  playerId: number
): { xgPerShot: number; goalsMinusXg: number } | null {
  const target = players.find((p) => p.player_id === playerId);
  if (!target || players.length === 0) return null;

  const xgPerShotValues = players.map((p) => (p.shots > 0 ? p.total_xg / p.shots : 0));
  const goalsMinusXgValues = players.map((p) => p.goals - p.total_xg);

  const targetXgPerShot = target.shots > 0 ? target.total_xg / target.shots : 0;
  const targetGoalsMinusXg = target.goals - target.total_xg;

  return {
    xgPerShot: percentileRank(xgPerShotValues, targetXgPerShot),
    goalsMinusXg: percentileRank(goalsMinusXgValues, targetGoalsMinusXg),
  };
}
