"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { PageHead } from "@/components/ui/PageHead";
import { Card } from "@/components/ui/Card";
import { Skeleton } from "@/components/ui/Skeleton";
import { useMatchFilter } from "@/components/shell/MatchFilterProvider";
import { getMatches } from "@/lib/api";
import type { MatchResponse } from "@/lib/types";

export default function MatchesPage() {
  const { competitionId, seasonId } = useMatchFilter();
  const [matches, setMatches] = useState<MatchResponse[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);

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
  }, [competitionId, seasonId]);

  return (
    <section>
      <PageHead title="Matches" crumb={`${matches.length} match${matches.length === 1 ? "" : "es"}`} />

      {loading && (
        <div className="bg-card border border-border rounded overflow-hidden">
          {Array.from({ length: 6 }).map((_, i) => (
            <div key={i} className="px-4 py-[10px] border-b border-border last:border-b-0">
              <Skeleton style={{ height: 14 }} />
            </div>
          ))}
        </div>
      )}

      {!loading && error && (
        <Card>
          <p className="text-[12.5px] text-muted m-0">Couldn&apos;t load matches. Try again shortly.</p>
        </Card>
      )}

      {!loading && !error && matches.length === 0 && (
        <Card>
          <p className="text-[12.5px] text-muted m-0">No matches found for the current filters.</p>
        </Card>
      )}

      {!loading && !error && matches.length > 0 && (
        <div className="bg-card border border-border rounded overflow-hidden">
          {matches.map((match) => (
            <Link
              key={match.match_id}
              href={`/matches/${match.match_id}`}
              className="flex items-center justify-between gap-3 px-4 py-[10px] border-b border-border last:border-b-0 text-[12.5px] hover:bg-card-hi"
            >
              <div className="flex-1 min-w-0">
                <span className="text-text">{match.home_team_name ?? "TBD"}</span>
                <span className="text-muted mx-1.5">vs</span>
                <span className="text-text">{match.away_team_name ?? "TBD"}</span>
              </div>
              <div className="font-data text-text2 w-16 text-center">
                {match.home_score ?? "-"} : {match.away_score ?? "-"}
              </div>
              <div className="text-muted w-24 text-right">{match.match_date ?? ""}</div>
              <div className="text-muted w-32 text-right truncate">{match.competition_stage ?? ""}</div>
            </Link>
          ))}
        </div>
      )}
    </section>
  );
}
