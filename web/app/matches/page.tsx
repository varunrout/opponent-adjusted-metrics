"use client";

import { useEffect, useState } from "react";
import { PageHead } from "@/components/ui/PageHead";
import { TeamLink } from "@/components/ui/EntityLink";
import { DataTable, type DataTableColumn } from "@/components/ui/DataTable";
import { DivergingBar } from "@/components/ui/DivergingBar";
import { useMatchFilter } from "@/components/shell/MatchFilterProvider";
import { getMatches } from "@/lib/api";
import type { MatchResponse } from "@/lib/types";

function resultTone(match: MatchResponse, side: "home" | "away"): string {
  if (match.home_score == null || match.away_score == null || match.home_score === match.away_score) {
    return "text-text";
  }
  const homeWon = match.home_score > match.away_score;
  const sideWon = side === "home" ? homeWon : !homeWon;
  return sideWon ? "text-text" : "text-text2";
}

export default function MatchesPage() {
  const { competitionId, seasonId, teamId, cxgScopeOnly } = useMatchFilter();
  const [matches, setMatches] = useState<MatchResponse[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);
  const [attempt, setAttempt] = useState(0);
  const [coverage, setCoverage] = useState<{ covered: number; total: number } | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(false);

    getMatches({ competition_id: competitionId, season_id: seasonId, team_id: teamId })
      .then((data) => {
        if (cancelled) return;
        // "CxG matches only" scopes to matches with fully collected 360
        // data — match_status_360 === "available" is the real, already-
        // shipping per-match signal for that (no separate endpoint exists).
        const scoped = cxgScopeOnly ? data.filter((m) => m.match_status_360 === "available") : data;
        setMatches(scoped);
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
  }, [competitionId, seasonId, teamId, cxgScopeOnly, attempt]);

  // Dataset-wide coverage strip — how many of the 610 matches carry a CxG
  // prediction at all (event-wide track, which only needs collected event
  // data, not 360), independent of the current filters/scope toggle.
  // match_status === "available" is the real per-match signal for that.
  useEffect(() => {
    let cancelled = false;
    getMatches({})
      .then((all) => {
        if (cancelled) return;
        const covered = all.filter((m) => m.match_status === "available").length;
        setCoverage({ covered, total: all.length });
      })
      .catch(() => {
        if (!cancelled) setCoverage(null);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const columns: DataTableColumn<MatchResponse>[] = [
    {
      key: "match_date",
      label: "Date",
      width: "100px",
      sortable: true,
      render: (m) => m.match_date ?? "",
    },
    {
      key: "home_team_name",
      label: "Home",
      width: "1fr",
      render: (m) => (
        <TeamLink
          teamId={m.home_team_id}
          name={m.home_team_name ?? "TBD"}
          className={resultTone(m, "home")}
        />
      ),
    },
    {
      key: "score",
      label: "Score",
      width: "130px",
      align: "center",
      render: (m) =>
        m.home_xg != null && m.away_xg != null ? (
          <div className="flex flex-col items-center gap-0.5 w-24">
            <span className="font-data text-[11px] text-text">
              {m.home_score ?? "-"} : {m.away_score ?? "-"}
            </span>
            <div className="w-full">
              <DivergingBar
                left={{ label: "", value: m.home_xg, color: "var(--home-team)" }}
                right={{ label: "", value: m.away_xg, color: "var(--away-team)" }}
                formatValue={() => ""}
              />
            </div>
          </div>
        ) : (
          `${m.home_score ?? "-"} : ${m.away_score ?? "-"}`
        ),
    },
    {
      key: "away_team_name",
      label: "Away",
      width: "1fr",
      render: (m) => (
        <TeamLink
          teamId={m.away_team_id}
          name={m.away_team_name ?? "TBD"}
          className={resultTone(m, "away")}
        />
      ),
    },
    {
      key: "competition_stage",
      label: "Stage",
      width: "140px",
      render: (m) => m.competition_stage ?? "",
    },
    {
      key: "stadium",
      label: "Venue",
      width: "140px",
      render: (m) => m.stadium ?? "",
    },
  ];

  return (
    <section>
      <PageHead title="Matches" crumb={`${matches.length} match${matches.length === 1 ? "" : "es"}`} />

      {coverage && coverage.total > 0 && (
        <p className="text-[11.5px] text-muted mb-3" data-testid="cxg-coverage-strip">
          {coverage.covered} of {coverage.total} matches · {Math.round((coverage.covered / coverage.total) * 100)}%
          carry CxG predictions
        </p>
      )}

      <DataTable
        columns={columns}
        rows={matches}
        rowKey={(m) => m.match_id}
        rowHref={(m) => `/matches/${m.match_id}`}
        loading={loading}
        error={error}
        onRetry={() => setAttempt((n) => n + 1)}
        emptyMessage="No matches found for the current filters."
        pageSize={50}
      />
    </section>
  );
}
