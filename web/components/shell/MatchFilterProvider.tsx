"use client";

import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import { getCompetitions } from "@/lib/api";
import type { CompetitionResponse } from "@/lib/types";

type MatchFilterContextValue = {
  competitions: CompetitionResponse[];
  competitionsLoading: boolean;
  competitionId: number | null;
  seasonId: number | null;
  setCompetitionId: (id: number | null) => void;
  setSeasonId: (id: number | null) => void;
};

const MatchFilterContext = createContext<MatchFilterContextValue | undefined>(undefined);

export function MatchFilterProvider({ children }: { children: ReactNode }) {
  const [competitions, setCompetitions] = useState<CompetitionResponse[]>([]);
  const [competitionsLoading, setCompetitionsLoading] = useState(true);
  const [competitionId, setCompetitionIdState] = useState<number | null>(null);
  const [seasonId, setSeasonId] = useState<number | null>(null);

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
    setSeasonId(null);
  }

  const value = useMemo(
    () => ({
      competitions,
      competitionsLoading,
      competitionId,
      seasonId,
      setCompetitionId,
      setSeasonId,
    }),
    [competitions, competitionsLoading, competitionId, seasonId]
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
