"use client";

import { useEffect, useState } from "react";
import { useParams, notFound } from "next/navigation";
import { Card } from "@/components/ui/Card";
import { MetricTile } from "@/components/ui/MetricTile";
import { PitchMap } from "@/components/ui/PitchMap";
import { Badge } from "@/components/ui/Badge";
import { Skeleton } from "@/components/ui/Skeleton";
import { TeamLink, PlayerLink } from "@/components/ui/EntityLink";
import { ShotDetailModal } from "@/components/shot/ShotDetailModal";
import { XgTimeline } from "@/components/shot/XgTimeline";
import { getMatch, getMatchShots, getCxgCoverage, getCxaCoverageByShot, ApiError } from "@/lib/api";
import { describeCxgCoverage } from "@/lib/analysis-helpers";
import { useMatchFilter } from "@/components/shell/MatchFilterProvider";
import type {
  MatchDetailResponse,
  ShotResponse,
  LineupPlayerResponse,
  CxaShotCoverageValues,
} from "@/lib/types";

export default function MatchDetailPage() {
  const params = useParams<{ matchId: string }>();
  const matchId = params.matchId;
  const { metricMode } = useMatchFilter();

  const [match, setMatch] = useState<MatchDetailResponse | null>(null);
  const [shots, setShots] = useState<ShotResponse[]>([]);
  const [cxgByEventId, setCxgByEventId] = useState<Record<string, number>>({});
  const [cxgPlusByEventId, setCxgPlusByEventId] = useState<Record<string, number>>({});
  const [cxaByEventId, setCxaByEventId] = useState<Record<string, CxaShotCoverageValues>>({});
  const [cxaPlusByEventId, setCxaPlusByEventId] = useState<Record<string, CxaShotCoverageValues>>({});
  const [loading, setLoading] = useState(true);
  const [missing, setMissing] = useState(false);
  const [error, setError] = useState(false);
  const [selectedShot, setSelectedShot] = useState<ShotResponse | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setMissing(false);
    setError(false);
    setCxgByEventId({});
    setCxgPlusByEventId({});
    setCxaByEventId({});
    setCxaPlusByEventId({});

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

  const outcomeBreakdown = deriveOutcomeBreakdown(shots, homeTeamId, awayTeamId);
  const biggestGap = deriveBiggestGap(shots, cxgByEventId);

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
            showLegend
            onShotClick={setSelectedShot}
          />
          {(() => {
            const captions = [
              describeCxgCoverage(shots.length, Object.keys(cxgByEventId).length, "CxG"),
              describeCxgCoverage(shots.length, Object.keys(cxgPlusByEventId).length, "CxG+"),
            ].filter(Boolean);
            return captions.length > 0 ? (
              <div className="flex items-center gap-2 mt-2">
                <Badge status="experimental" label="Experimental" />
                <p className="text-[11.5px] text-muted m-0">{captions.join(" · ")}</p>
              </div>
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

      <div className="grid gap-4 items-start mb-4" style={{ gridTemplateColumns: "2fr 1fr" }}>
        <Card title="Cumulative xG">
          <XgTimeline
            shots={shots}
            homeTeamId={homeTeamId}
            awayTeamId={awayTeamId}
            homeLabel={match.home_team_name ?? "Home"}
            awayLabel={match.away_team_name ?? "Away"}
          />
        </Card>

        <Card title="Shot outcomes">
          <OutcomeBreakdownList
            breakdown={outcomeBreakdown}
            homeLabel={match.home_team_name ?? "Home"}
            awayLabel={match.away_team_name ?? "Away"}
          />
        </Card>
      </div>

      {biggestGap && (
        <Card title="Biggest CxG-vs-xG gap" className="mb-4">
          <div className="flex items-center justify-between gap-3 flex-wrap">
            <p className="text-[12.5px] text-text2 m-0">
              <PlayerLink
                playerId={biggestGap.shot.player_id}
                name={biggestGap.shot.player_name ?? "Unknown player"}
                className="text-text"
              />{" "}
              — xG <span className="font-data text-text">{(biggestGap.shot.statsbomb_xg ?? 0).toFixed(2)}</span> vs
              CxG <span className="font-data" style={{ color: "var(--teal)" }}>{biggestGap.cxg.toFixed(2)}</span> (
              {biggestGap.diff >= 0 ? "+" : ""}
              {biggestGap.diff.toFixed(2)})
            </p>
            <button
              type="button"
              onClick={() => setSelectedShot(biggestGap.shot)}
              className="text-[11.5px] px-2.5 py-1 rounded border border-border bg-card-hi text-text cursor-pointer"
            >
              Jump to shot
            </button>
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

type OutcomeBreakdown = { outcome: string; home: number; away: number };

function deriveOutcomeBreakdown(
  shots: ShotResponse[],
  homeTeamId: number | null,
  awayTeamId: number | null
): OutcomeBreakdown[] {
  const byOutcome = new Map<string, { home: number; away: number }>();
  for (const shot of shots) {
    const outcome = shot.outcome_name ?? "Unknown";
    const entry = byOutcome.get(outcome) ?? { home: 0, away: 0 };
    if (shot.team_id === homeTeamId) entry.home += 1;
    else if (shot.team_id === awayTeamId) entry.away += 1;
    byOutcome.set(outcome, entry);
  }
  return Array.from(byOutcome.entries())
    .map(([outcome, counts]) => ({ outcome, ...counts }))
    .sort((a, b) => b.home + b.away - (a.home + a.away));
}

function OutcomeBreakdownList({
  breakdown,
  homeLabel,
  awayLabel,
}: {
  breakdown: OutcomeBreakdown[];
  homeLabel: string;
  awayLabel: string;
}) {
  if (breakdown.length === 0) {
    return <p className="text-[12.5px] text-muted m-0">No shots recorded.</p>;
  }
  return (
    <div>
      <div className="flex items-center justify-between text-[11px] text-muted mb-1.5">
        <span>Outcome</span>
        <span className="flex gap-3">
          <span>{homeLabel}</span>
          <span>{awayLabel}</span>
        </span>
      </div>
      {breakdown.map((row) => (
        <div
          key={row.outcome}
          className="flex items-center justify-between py-[6px] border-b border-border last:border-b-0 text-[12.5px]"
        >
          <span className="text-text">{row.outcome}</span>
          <span className="flex gap-3 font-data">
            <span style={{ color: "var(--home-team)" }}>{row.home}</span>
            <span style={{ color: "var(--away-team)" }}>{row.away}</span>
          </span>
        </div>
      ))}
    </div>
  );
}

function deriveBiggestGap(
  shots: ShotResponse[],
  cxgByEventId: Record<string, number>
): { shot: ShotResponse; cxg: number; diff: number } | null {
  let best: { shot: ShotResponse; cxg: number; diff: number } | null = null;
  for (const shot of shots) {
    const cxg = cxgByEventId[shot.event_id];
    if (cxg == null) continue;
    const diff = cxg - (shot.statsbomb_xg ?? 0);
    if (best == null || Math.abs(diff) > Math.abs(best.diff)) {
      best = { shot, cxg, diff };
    }
  }
  return best;
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
