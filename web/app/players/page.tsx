"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { PageHead } from "@/components/ui/PageHead";
import { Card } from "@/components/ui/Card";
import { Skeleton } from "@/components/ui/Skeleton";
import { useMatchFilter } from "@/components/shell/MatchFilterProvider";
import { getPlayers } from "@/lib/api";
import type { PlayerSeasonResponse } from "@/lib/types";

export default function PlayersPage() {
  const { competitionId, seasonId } = useMatchFilter();
  const [players, setPlayers] = useState<PlayerSeasonResponse[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(false);
    getPlayers({ competition_id: competitionId, season_id: seasonId })
      .then((data) => {
        if (!cancelled) setPlayers(data);
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
      <PageHead title="Players" crumb={`${players.length} player${players.length === 1 ? "" : "s"}`} />

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
          <p className="text-[12.5px] text-muted m-0">Couldn&apos;t load players. Try again shortly.</p>
        </Card>
      )}

      {!loading && !error && players.length === 0 && (
        <Card>
          <p className="text-[12.5px] text-muted m-0">No players found for the current filters.</p>
        </Card>
      )}

      {!loading && !error && players.length > 0 && (
        <div className="bg-card border border-border rounded overflow-hidden">
          {players.map((player, i) => (
            <Link
              key={player.player_id}
              href={`/players/${player.player_id}`}
              className="flex items-center justify-between gap-3 px-4 py-[10px] border-b border-border last:border-b-0 text-[12.5px] hover:bg-card-hi"
            >
              <span className="w-6 text-muted font-data">{i + 1}</span>
              <div className="flex-1 min-w-0">
                <span className="text-text">{player.player_name ?? "Unknown"}</span>
                <span className="text-muted mx-1.5">·</span>
                <span className="text-muted">{player.team_name ?? "Unknown team"}</span>
              </div>
              <div className="font-data text-text2 w-16 text-right">{player.shots}</div>
              <div className="font-data text-text2 w-16 text-right">{player.goals}</div>
              <div className="font-data text-text2 w-20 text-right">{player.total_xg.toFixed(2)}</div>
            </Link>
          ))}
        </div>
      )}
    </section>
  );
}
