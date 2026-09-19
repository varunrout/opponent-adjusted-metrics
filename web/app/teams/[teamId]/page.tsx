"use client";

import { useEffect, useMemo, useState } from "react";
import { useParams } from "next/navigation";
import { Card } from "@/components/ui/Card";
import { MetricTile } from "@/components/ui/MetricTile";
import { PitchMap } from "@/components/ui/PitchMap";
import { Badge } from "@/components/ui/Badge";
import { Skeleton } from "@/components/ui/Skeleton";
import { ClickableRow } from "@/components/ui/ClickableRow";
import { TeamLink, PlayerLink } from "@/components/ui/EntityLink";
import { DivergingBar } from "@/components/ui/DivergingBar";
import { ShotDetailModal } from "@/components/shot/ShotDetailModal";
import { useMatchFilter } from "@/components/shell/MatchFilterProvider";
import { getTeamShots, getTeamShotsFaced, getMatches, getCxgCoverage } from "@/lib/api";
import { summarizeShots } from "@/lib/shot-summary";
import { describeCxgCoverage } from "@/lib/analysis-helpers";
import type { MatchResponse, ShotResponse } from "@/lib/types";

export default function TeamDetailPage() {
  const params = useParams<{ teamId: string }>();
  const teamId = params.teamId;
  const { competitionId, seasonId, metricMode, cxgScopeOnly } = useMatchFilter();

  const [shots, setShots] = useState<ShotResponse[]>([]);
  const [matches, setMatches] = useState<MatchResponse[]>([]);
  const [cxgByEventId, setCxgByEventId] = useState<Record<string, number>>({});
  const [cxgPlusByEventId, setCxgPlusByEventId] = useState<Record<string, number>>({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);
  const [selectedShot, setSelectedShot] = useState<ShotResponse | null>(null);
  const [shotsFaced, setShotsFaced] = useState<ShotResponse[]>([]);
  const [cxgByEventIdFaced, setCxgByEventIdFaced] = useState<Record<string, number>>({});

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

  useEffect(() => {
    let cancelled = false;
    getTeamShotsFaced(teamId, { competition_id: competitionId, season_id: seasonId })
      .then((data) => {
        if (cancelled) return;
        setShotsFaced(data);
        return getCxgCoverage(
          data.map((s) => s.event_id),
          "cxg_event"
        ).then((coverage) => {
          if (!cancelled) setCxgByEventIdFaced(coverage.values);
        });
      })
      .catch(() => {
        if (!cancelled) {
          setShotsFaced([]);
          setCxgByEventIdFaced({});
        }
      });
    return () => {
      cancelled = true;
    };
  }, [teamId, competitionId, seasonId]);

  // Same "CxG matches only" scope as the Matches page — match_status_360
  // === "available" is the real per-match signal for 360 coverage, already
  // present on the `matches` rows fetched above. No separate endpoint (there
  // isn't one), and no extra fetch needed — this team's matches are already
  // loaded.
  const coveredMatchIds = useMemo(
    () =>
      cxgScopeOnly
        ? new Set(matches.filter((m) => m.match_status_360 === "available").map((m) => m.match_id))
        : null,
    [cxgScopeOnly, matches]
  );

  const numericTeamId = Number(teamId);

  // "CxG matches only" scopes every shot/match-driven number on this page —
  // shot map, tiles, top scorers, attack/defence, and the recent-matches
  // list all derive from these, not the raw fetched arrays.
  const scopedShots = coveredMatchIds ? shots.filter((s) => coveredMatchIds.has(s.match_id)) : shots;
  const scopedShotsFaced = coveredMatchIds
    ? shotsFaced.filter((s) => coveredMatchIds.has(s.match_id))
    : shotsFaced;
  const scopedMatches = coveredMatchIds ? matches.filter((m) => coveredMatchIds.has(m.match_id)) : matches;

  const topScorers = useMemo(() => {
    const byPlayer = new Map<
      number,
      { player_id: number; player_name: string | null; goals: number; xg: number }
    >();
    for (const s of scopedShots) {
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
  }, [scopedShots]);

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

  const summary = summarizeShots(scopedShots);
  const xgPerShot = summary.shots > 0 ? summary.totalXg / summary.shots : 0;
  const goalsMinusXg = summary.goals - summary.totalXg;

  const sortedMatches = [...scopedMatches].sort((a, b) =>
    (b.match_date ?? "").localeCompare(a.match_date ?? "")
  );

  // Team name resolution uses the full unscoped match list — the team's
  // identity shouldn't depend on whether any of its matches happen to carry
  // 360 coverage. ShotResponse carries no team_name field, so resolve it
  // from the first matching match row (mirrors the player page's use of
  // the shot's own player_name, the closest analogue available here).
  const matchWithTeam = matches.find(
    (m) => m.home_team_id === numericTeamId || m.away_team_id === numericTeamId
  );
  const displayName = matchWithTeam
    ? matchWithTeam.home_team_id === numericTeamId
      ? matchWithTeam.home_team_name ?? "Unknown team"
      : matchWithTeam.away_team_name ?? "Unknown team"
    : "Unknown team";

  const coverageCaptions = [
    describeCxgCoverage(scopedShots.length, Object.keys(cxgByEventId).length, "CxG"),
    describeCxgCoverage(scopedShots.length, Object.keys(cxgPlusByEventId).length, "CxG+"),
  ].filter(Boolean);

  const goalsConceded = scopedShotsFaced.filter((s) => s.is_goal).length;
  const xgConceded = scopedShotsFaced.reduce((sum, s) => sum + (s.statsbomb_xg ?? 0), 0);
  const coveredShotsFaced = scopedShotsFaced.filter((s) => cxgByEventIdFaced[s.event_id] != null);
  const coveredGoalsConceded = coveredShotsFaced.filter((s) => s.is_goal).length;
  const cxgAllowed = coveredShotsFaced.reduce(
    (sum, s) => sum + (cxgByEventIdFaced[s.event_id] ?? 0),
    0
  );

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
            shots={scopedShots}
            homeTeamId={numericTeamId}
            cxgByEventId={cxgByEventId}
            cxgPlusByEventId={cxgPlusByEventId}
            sizeBy={metricMode}
            showLegend
            onShotClick={setSelectedShot}
          />
          {coverageCaptions.length > 0 && (
            <div className="flex items-center gap-2 mt-2">
              <Badge status="experimental" label="Experimental" />
              <p className="text-[11.5px] text-muted m-0">{coverageCaptions.join(" · ")}</p>
            </div>
          )}
        </Card>

        <Card title="Top scorers">
          {topScorers.length === 0 ? (
            <p className="text-[12.5px] text-muted m-0">No shots recorded for this team.</p>
          ) : (
            <div className="flex flex-col gap-2">
              {topScorers.map((p) => (
                <div key={p.player_id} className="flex flex-col gap-1">
                  <div className="flex items-center justify-between text-[12.5px]">
                    <PlayerLink playerId={p.player_id} name={p.player_name ?? "Unknown"} className="text-text" />
                    <span className="font-data text-text2">
                      {p.goals}g · {p.xg.toFixed(2)}xG
                    </span>
                  </div>
                  <div className="h-2 rounded-full overflow-hidden" style={{ background: "var(--card-hi)" }}>
                    <div
                      className="h-full rounded-full"
                      style={{
                        width: `${(p.goals / Math.max(...topScorers.map((row) => row.goals), 1)) * 100}%`,
                        background: "var(--teal)",
                      }}
                    />
                  </div>
                </div>
              ))}
            </div>
          )}
        </Card>
      </div>

      {scopedShotsFaced.length > 0 && (
        <Card title="Attack vs defence, both opponent-adjusted" className="mb-4">
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <DivergingBar
              left={{ label: "Goals", value: summary.goals, color: "var(--green)" }}
              right={{ label: "Total xG created", value: summary.totalXg, color: "var(--muted)" }}
            />
            <DivergingBar
              left={{ label: "Goals conceded", value: goalsConceded, color: "var(--red)" }}
              right={{ label: "xG conceded", value: xgConceded, color: "var(--muted)" }}
            />
            <DivergingBar
              left={{ label: "Goals conceded", value: coveredGoalsConceded, color: "var(--red)" }}
              right={{ label: "CxG allowed", value: cxgAllowed, color: "var(--teal)" }}
            />
          </div>
          {coveredShotsFaced.length > 0 && (
            <p className="text-[11px] text-muted mt-2 mb-0">
              CxG-allowed row scoped to {coveredShotsFaced.length} CxG-covered shot
              {coveredShotsFaced.length === 1 ? "" : "s"} faced.
            </p>
          )}
        </Card>
      )}

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

      <ShotDetailModal
        shot={selectedShot}
        open={selectedShot != null}
        onClose={() => setSelectedShot(null)}
        cxg={selectedShot ? cxgByEventId[selectedShot.event_id] : undefined}
        cxgPlus={selectedShot ? cxgPlusByEventId[selectedShot.event_id] : undefined}
      />
    </section>
  );
}
