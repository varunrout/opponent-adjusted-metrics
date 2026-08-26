"use client";

import { useEffect, useMemo, useState } from "react";
import { PageHead } from "@/components/ui/PageHead";
import { TeamLink } from "@/components/ui/EntityLink";
import { DataTable, type DataTableColumn } from "@/components/ui/DataTable";
import { useMatchFilter } from "@/components/shell/MatchFilterProvider";
import { getPlayers } from "@/lib/api";
import type { PlayerSeasonResponse } from "@/lib/types";

const DEFAULT_MIN_SHOTS = 10;

type PlayerRow = PlayerSeasonResponse & { xgPerShot: number; goalsMinusXg: number };

export default function PlayersPage() {
  const { competitionId, seasonId } = useMatchFilter();
  const [players, setPlayers] = useState<PlayerSeasonResponse[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);
  const [minShots, setMinShots] = useState(DEFAULT_MIN_SHOTS);
  const [attempt, setAttempt] = useState(0);

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
  }, [competitionId, seasonId, attempt]);

  const rows: PlayerRow[] = useMemo(
    () =>
      players
        .filter((p) => p.shots >= minShots)
        .map((p) => ({
          ...p,
          xgPerShot: p.shots > 0 ? p.total_xg / p.shots : 0,
          goalsMinusXg: p.goals - p.total_xg,
        })),
    [players, minShots]
  );

  const columns: DataTableColumn<PlayerRow>[] = [
    { key: "rank", label: "#", width: "32px", render: (_row, index) => index + 1 },
    {
      key: "player_name",
      label: "Player",
      width: "1fr",
      sortable: true,
      render: (row) => row.player_name ?? "Unknown",
    },
    {
      key: "team_name",
      label: "Team",
      width: "1fr",
      render: (row) => (
        <TeamLink teamId={row.team_id} name={row.team_name ?? "Unknown team"} className="text-muted" />
      ),
    },
    { key: "shots", label: "Shots", width: "70px", align: "right", sortable: true },
    { key: "goals", label: "Goals", width: "70px", align: "right", sortable: true },
    {
      key: "total_xg",
      label: "xG",
      width: "80px",
      align: "right",
      sortable: true,
      render: (row) => row.total_xg.toFixed(2),
    },
    {
      key: "xgPerShot",
      label: "xG/shot",
      width: "80px",
      align: "right",
      sortable: true,
      render: (row) => row.xgPerShot.toFixed(2),
    },
    {
      key: "goalsMinusXg",
      label: "G−xG",
      width: "80px",
      align: "right",
      sortable: true,
      render: (row) => (
        <span style={{ color: row.goalsMinusXg >= 0 ? "var(--green)" : "var(--red)" }}>
          {row.goalsMinusXg >= 0 ? "+" : ""}
          {row.goalsMinusXg.toFixed(2)}
        </span>
      ),
    },
  ];

  return (
    <section>
      <PageHead
        title="Players"
        crumb={`${rows.length} player${rows.length === 1 ? "" : "s"} · min. ${minShots} shots`}
      />

      <div className="flex items-center gap-2 mb-3">
        <label htmlFor="min-shots" className="text-[12px] text-muted">
          Minimum shots
        </label>
        <input
          id="min-shots"
          type="number"
          min={0}
          value={minShots}
          onChange={(e) => setMinShots(Math.max(0, Number(e.target.value) || 0))}
          className="w-16 bg-card border border-border text-text rounded px-2 py-1 text-[12px] font-data"
        />
      </div>

      <DataTable
        columns={columns}
        rows={rows}
        rowKey={(row) => row.player_id}
        rowHref={(row) => `/players/${row.player_id}`}
        loading={loading}
        error={error}
        onRetry={() => setAttempt((n) => n + 1)}
        emptyMessage="No players found for the current filters."
        pageSize={50}
      />
    </section>
  );
}
