"use client";

import { useEffect, useState } from "react";
import { PageHead } from "@/components/ui/PageHead";
import { TeamLink } from "@/components/ui/EntityLink";
import { DataTable, type DataTableColumn } from "@/components/ui/DataTable";
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
  const { competitionId, seasonId } = useMatchFilter();
  const [matches, setMatches] = useState<MatchResponse[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(false);
    getMatches({ competition_id: competitionId, season_id: seasonId })
      .then((data) => {
        if (!cancelled) setMatches(data);
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
  }, [competitionId, seasonId, attempt]);

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
      width: "70px",
      align: "center",
      render: (m) => `${m.home_score ?? "-"} : ${m.away_score ?? "-"}`,
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
