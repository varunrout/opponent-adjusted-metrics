"use client";

import { useEffect, useState } from "react";
import { useParams, notFound } from "next/navigation";
import { Card } from "@/components/ui/Card";
import { MetricTile } from "@/components/ui/MetricTile";
import { PitchMap } from "@/components/ui/PitchMap";
import { Skeleton } from "@/components/ui/Skeleton";
import { TeamLink, PlayerLink } from "@/components/ui/EntityLink";
import { getMatch, getMatchShots, getCxgCoverage, ApiError } from "@/lib/api";
import { describeCxgCoverage } from "@/lib/analysis-helpers";
import { useMatchFilter } from "@/components/shell/MatchFilterProvider";
import type { MatchDetailResponse, ShotResponse, LineupPlayerResponse } from "@/lib/types";

export default function MatchDetailPage() {
  const params = useParams<{ matchId: string }>();
  const matchId = params.matchId;
  const { metricMode } = useMatchFilter();

  const [match, setMatch] = useState<MatchDetailResponse | null>(null);
  const [shots, setShots] = useState<ShotResponse[]>([]);
  const [cxgByEventId, setCxgByEventId] = useState<Record<string, number>>({});
  const [cxgPlusByEventId, setCxgPlusByEventId] = useState<Record<string, number>>({});
  const [loading, setLoading] = useState(true);
  const [missing, setMissing] = useState(false);
  const [error, setError] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setMissing(false);
    setError(false);
    setCxgByEventId({});
    setCxgPlusByEventId({});

    Promise.all([getMatch(matchId), getMatchShots(matchId)])
      .then(([matchData, shotsData]) => {
        if (cancelled) return;
        setMatch(matchData);
        setShots(shotsData);

        const eventIds = shotsData.map((s) => s.event_id);

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
      .catch((err) => {
        if (cancelled) return;
        if (err instanceof ApiError && err.status === 404) {
          setMissing(true);
        } else {
          setError(true);
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [matchId]);

  if (missing) {
    notFound();
  }

  if (loading) {
    return (
      <section>
        <Skeleton style={{ height: 32, width: "50%", marginBottom: 18 }} />
        <Skeleton style={{ height: 240 }} />
      </section>
    );
  }

  if (error || !match) {
    return (
      <section>
        <Card>
          <p className="text-[12.5px] text-muted m-0">Couldn&apos;t load this match. Try again shortly.</p>
        </Card>
      </section>
    );
  }

  const homeTeamId = match.home_team_id;
  const awayTeamId = match.away_team_id;

  const homeXg = shots
    .filter((s) => s.team_id === homeTeamId)
    .reduce((sum, s) => sum + (s.statsbomb_xg ?? 0), 0);
  const awayXg = shots
    .filter((s) => s.team_id === awayTeamId)
    .reduce((sum, s) => sum + (s.statsbomb_xg ?? 0), 0);

  const homeLineup = match.lineups.filter((p) => p.team_id === homeTeamId);
  const awayLineup = match.lineups.filter((p) => p.team_id === awayTeamId);

  return (
    <section>
      <div className="mb-[18px]">
        <h1 className="text-lg font-semibold m-0 flex items-baseline gap-1.5 flex-wrap">
          <TeamLink teamId={homeTeamId} name={match.home_team_name ?? "TBD"} className="text-text" />
          <span>
            {match.home_score ?? "-"} — {match.away_score ?? "-"}
          </span>
          <TeamLink teamId={awayTeamId} name={match.away_team_name ?? "TBD"} className="text-text" />
        </h1>
        <div className="text-[12.5px] text-muted mt-1">
          {[match.match_date, match.stadium, match.competition_stage].filter(Boolean).join(" · ")}
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3 mb-[18px]">
        <MetricTile label={`${match.home_team_name ?? "Home"} xG`} value={homeXg.toFixed(2)} />
        <MetricTile label={`${match.away_team_name ?? "Away"} xG`} value={awayXg.toFixed(2)} />
      </div>

      <div className="grid gap-4 items-start mb-4" style={{ gridTemplateColumns: "2fr 1fr" }}>
        <Card title="Shot map">
          <PitchMap
            shots={shots}
            homeTeamId={homeTeamId}
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
        </Card>

        <div className="grid grid-cols-1 gap-3.5">
          <Card title={`${match.home_team_name ?? "Home"} lineup`}>
            <LineupList players={homeLineup} />
          </Card>
          <Card title={`${match.away_team_name ?? "Away"} lineup`}>
            <LineupList players={awayLineup} />
          </Card>
        </div>
      </div>
    </section>
  );
}

function LineupList({ players }: { players: LineupPlayerResponse[] }) {
  if (players.length === 0) {
    return <p className="text-[12.5px] text-muted m-0">No lineup data.</p>;
  }
  return (
    <div>
      {players.map((p) => (
        <div
          key={p.player_id}
          className="flex items-center gap-2.5 py-[6px] border-b border-border last:border-b-0 text-[12.5px]"
        >
          <span className="w-6 font-data text-muted">{p.jersey_number ?? ""}</span>
          <PlayerLink
            playerId={p.player_id}
            name={p.player_name ?? "Unknown"}
            className="flex-1 text-text"
          />
          <span className="text-muted">{p.position_name ?? ""}</span>
        </div>
      ))}
    </div>
  );
}
