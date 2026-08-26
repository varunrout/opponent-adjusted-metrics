"use client";

import { useEffect, useMemo, useState } from "react";
import { useParams } from "next/navigation";
import { Card } from "@/components/ui/Card";
import { MetricTile } from "@/components/ui/MetricTile";
import { PitchMap } from "@/components/ui/PitchMap";
import { Skeleton } from "@/components/ui/Skeleton";
import { ClickableRow } from "@/components/ui/ClickableRow";
import { TeamLink, PlayerLink } from "@/components/ui/EntityLink";
import { useMatchFilter } from "@/components/shell/MatchFilterProvider";
import { getTeamShots, getMatches, getCxgCoverage } from "@/lib/api";
import { summarizeShots } from "@/lib/shot-summary";
import { describeCxgCoverage } from "@/lib/analysis-helpers";
import type { MatchResponse, ShotResponse } from "@/lib/types";

export default function TeamDetailPage() {
  const params = useParams<{ teamId: string }>();
  const teamId = params.teamId;
  const { competitionId, seasonId } = useMatchFilter();

  const [shots, setShots] = useState<ShotResponse[]>([]);
  const [matches, setMatches] = useState<MatchResponse[]>([]);
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

    Promise.all([
      getTeamShots(teamId, { competition_id: competitionId, season_id: seasonId }),
      getMatches({ competition_id: competitionId, season_id: seasonId, team_id: Number(teamId) }),
    ])
      .then(([shotsData, matchesData]) => {
        if (cancelled) return;
        setShots(shotsData);
        setMatches(matchesData);

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
      .catch(() => {
        if (!cancelled) setError(true);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [teamId, competitionId, seasonId]);

  const numericTeamId = Number(teamId);

  const topScorers = useMemo(() => {
    const byPlayer = new Map<
      number,
      { player_id: number; player_name: string | null; goals: number; xg: number }
    >();
    for (const s of shots) {
      if (s.player_id == null) continue;
      const entry = byPlayer.get(s.player_id) ?? {
        player_id: s.player_id,
        player_name: s.player_name,
        goals: 0,
        xg: 0,
      };
      entry.goals += s.is_goal ? 1 : 0;
      entry.xg += s.statsbomb_xg ?? 0;
      byPlayer.set(s.player_id, entry);
    }
    return Array.from(byPlayer.values())
      .sort((a, b) => b.goals - a.goals || b.xg - a.xg)
      .slice(0, 8);
  }, [shots]);

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
          <p className="text-[12.5px] text-muted m-0">Couldn&apos;t load this team. Try again shortly.</p>
        </Card>
      </section>
    );
  }

  const summary = summarizeShots(shots);
  const xgPerShot = summary.shots > 0 ? summary.totalXg / summary.shots : 0;
  const goalsMinusXg = summary.goals - summary.totalXg;

  const sortedMatches = [...matches].sort((a, b) =>
    (b.match_date ?? "").localeCompare(a.match_date ?? "")
  );

  // ShotResponse carries no team_name field, so resolve it from the first
  // matching recent-matches row (mirrors the player page's use of the
  // shot's own player_name, the closest analogue available here).
  const matchWithTeam = sortedMatches.find(
    (m) => m.home_team_id === numericTeamId || m.away_team_id === numericTeamId
  );
  const displayName = matchWithTeam
    ? matchWithTeam.home_team_id === numericTeamId
      ? matchWithTeam.home_team_name ?? "Unknown team"
      : matchWithTeam.away_team_name ?? "Unknown team"
    : "Unknown team";

  const coverageCaptions = [
    describeCxgCoverage(shots.length, Object.keys(cxgByEventId).length, "CxG"),
    describeCxgCoverage(shots.length, Object.keys(cxgPlusByEventId).length, "CxG+"),
  ].filter(Boolean);

  return (
    <section>
      <div className="mb-[18px]">
        <h1 className="text-lg font-semibold m-0">{displayName}</h1>
        <div className="text-[12.5px] text-muted mt-1">Season shot record</div>
      </div>

      <div className="grid grid-cols-5 gap-3 mb-[18px]">
        <MetricTile label="Shots" value={String(summary.shots)} />
        <MetricTile label="Goals" value={String(summary.goals)} />
        <MetricTile label="Total xG" value={summary.totalXg.toFixed(2)} />
        <MetricTile label="xG/shot" value={xgPerShot.toFixed(2)} />
        <MetricTile
          label="G−xG"
          value={`${goalsMinusXg >= 0 ? "+" : ""}${goalsMinusXg.toFixed(2)}`}
          deltaTone={goalsMinusXg >= 0 ? "pos" : "neg"}
        />
      </div>

      <div className="grid gap-4 items-start mb-4" style={{ gridTemplateColumns: "2fr 1fr" }}>
        <Card title="Shot map">
          <PitchMap
            shots={shots}
            homeTeamId={numericTeamId}
            cxgByEventId={cxgByEventId}
            cxgPlusByEventId={cxgPlusByEventId}
          />
          {coverageCaptions.length > 0 && (
            <p className="text-[11.5px] text-muted mt-2 mb-0">{coverageCaptions.join(" · ")}</p>
          )}
        </Card>

        <Card title="Top scorers">
          {topScorers.length === 0 ? (
            <p className="text-[12.5px] text-muted m-0">No shots recorded for this team.</p>
          ) : (
            <div>
              {topScorers.map((p) => (
                <div
                  key={p.player_id}
                  className="flex items-center justify-between gap-2.5 py-[7px] border-b border-border last:border-b-0 text-[12.5px]"
                >
                  <PlayerLink playerId={p.player_id} name={p.player_name ?? "Unknown"} className="text-text" />
                  <span className="font-data text-text2">
                    {p.goals}g · {p.xg.toFixed(2)}xG
                  </span>
                </div>
              ))}
            </div>
          )}
        </Card>
      </div>

      <Card title="Recent matches">
        {sortedMatches.length === 0 ? (
          <p className="text-[12.5px] text-muted m-0">No matches found for this team in the current filters.</p>
        ) : (
          <div>
            {sortedMatches.map((match) => (
              <ClickableRow
                key={match.match_id}
                href={`/matches/${match.match_id}`}
                className="flex items-center justify-between gap-3 py-[10px] border-b border-border last:border-b-0 text-[12.5px] hover:bg-card-hi cursor-pointer"
              >
                <div className="flex-1 min-w-0">
                  <TeamLink teamId={match.home_team_id} name={match.home_team_name ?? "TBD"} className="text-text" />
                  <span className="text-muted mx-1.5">vs</span>
                  <TeamLink teamId={match.away_team_id} name={match.away_team_name ?? "TBD"} className="text-text" />
                </div>
                <div className="font-data text-text2 w-16 text-center">
                  {match.home_score ?? "-"} : {match.away_score ?? "-"}
                </div>
                <div className="text-muted w-24 text-right">{match.match_date ?? ""}</div>
              </ClickableRow>
            ))}
          </div>
        )}
      </Card>
    </section>
  );
}
