"use client";

import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import { getCompetitions } from "@/lib/api";
import type { CompetitionResponse } from "@/lib/types";

// Per docs/dashboard_content_spec_v3.md §2.2: the metric toggle is a
// display mode for PitchMap (sizeBy), not a data refetch — coverage is
// already loaded wherever PitchMap is used. CxA/CxT stay disabled ("soon")
// in the Sidebar and have no state here yet.
export type MetricMode = "xg" | "cxg";

type MatchFilterContextValue = {
  competitions: CompetitionResponse[];
  competitionsLoading: boolean;
  competitionId: number | null;
  seasonId: number | null;
  teamId: number | null;
  metricMode: MetricMode;
  setCompetitionId: (id: number | null) => void;
  setSeasonId: (id: number | null) => void;
  setTeamId: (id: number | null) => void;
  setMetricMode: (mode: MetricMode) => void;
};

const MatchFilterContext = createContext<MatchFilterContextValue | undefined>(undefined);

export function MatchFilterProvider({ children }: { children: ReactNode }) {
  const [competitions, setCompetitions] = useState<CompetitionResponse[]>([]);
  const [competitionsLoading, setCompetitionsLoading] = useState(true);
  const [competitionId, setCompetitionIdState] = useState<number | null>(null);
  const [seasonId, setSeasonIdState] = useState<number | null>(null);
  const [teamId, setTeamId] = useState<number | null>(null);
  const [metricMode, setMetricMode] = useState<MetricMode>("xg");

  useEffect(() => {
    let cancelled = false;
    getCompetitions()
      .then((data) => {
        if (!cancelled) setCompetitions(data);
      })
      .catch(() => {
        if (!cancelled) setCompetitions([]);
      })
      .finally(() => {
        if (!cancelled) setCompetitionsLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  function setCompetitionId(id: number | null) {
    setCompetitionIdState(id);
    setSeasonIdState(null);
    setTeamId(null);
  }

  function setSeasonId(id: number | null) {
    setSeasonIdState(id);
    setTeamId(null);
  }

  const value = useMemo(
    () => ({
      competitions,
      competitionsLoading,
      competitionId,
      seasonId,
      teamId,
      metricMode,
      setCompetitionId,
      setSeasonId,
      setTeamId,
      setMetricMode,
    }),
    [competitions, competitionsLoading, competitionId, seasonId, teamId, metricMode]
  );

  return <MatchFilterContext.Provider value={value}>{children}</MatchFilterContext.Provider>;
}

export function useMatchFilter(): MatchFilterContextValue {
  const ctx = useContext(MatchFilterContext);
  if (!ctx) {
    throw new Error("useMatchFilter must be used within a MatchFilterProvider");
  }
  return ctx;
}
