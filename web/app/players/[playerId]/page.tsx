"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import { Card } from "@/components/ui/Card";
import { MetricTile } from "@/components/ui/MetricTile";
import { PitchMap } from "@/components/ui/PitchMap";
import { Skeleton } from "@/components/ui/Skeleton";
import { useMatchFilter } from "@/components/shell/MatchFilterProvider";
import { getPlayerShots, getCxgCoverage } from "@/lib/api";
import { summarizeShots } from "@/lib/shot-summary";
import { describeCxgCoverage } from "@/lib/analysis-helpers";
import type { ShotResponse } from "@/lib/types";

export default function PlayerDetailPage() {
  const params = useParams<{ playerId: string }>();
  const playerId = params.playerId;
  const { competitionId, seasonId, metricMode } = useMatchFilter();

  const [shots, setShots] = useState<ShotResponse[]>([]);
  const [cxgByEventId, setCxgByEventId] = useState<Record<string, number>>({});
  const [cxgPlusByEventId, setCxgPlusByEventId] = useState<Record<string, number>>({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(false);
    setCxgByEventId({});
    setCxgPlusByEventId({});

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
  const summary = summarizeShots(shots);

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
        {shots.length === 0 ? (
          <p className="text-[12.5px] text-muted m-0">No shots recorded for this player in the current filters.</p>
        ) : (
          <>
            <PitchMap
              shots={shots}
              homeTeamId={teamId}
              cxgByEventId={cxgByEventId}
              cxgPlusByEventId={cxgPlusByEventId}
              sizeBy={metricMode}
            />
            {(() => {
              const captions = [
                describeCxgCoverage(shots.length, Object.keys(cxgByEventId).length, "CxG"),
                describeCxgCoverage(shots.length, Object.keys(cxgPlusByEventId).length, "CxG+"),
              ].filter(Boolean);
              return captions.length > 0 ? (
                <p className="text-[11.5px] text-muted mt-2 mb-0">{captions.join(" · ")}</p>
              ) : null;
            })()}
          </>
        )}
      </Card>
    </section>
  );
}
