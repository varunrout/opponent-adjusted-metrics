"use client";

import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { getCompetitions } from "@/lib/api";
import type { CompetitionResponse } from "@/lib/types";

// Per docs/dashboard_content_spec_v3.md §2.2: the metric toggle is a
// display mode for PitchMap (sizeBy), not a data refetch — coverage is
// already loaded wherever PitchMap is used. CxA/CxT stay disabled ("soon")
// in the Sidebar and have no state here yet.
export type MetricMode = "xg" | "cxg";

// Per content_spec_v3.md §2.2a: default ON ("CxG matches only"). Reflected
// in the URL as ?scope=all when OFF (absent means ON) so a scoped or
// unscoped view is shareable via a plain link.
const SCOPE_PARAM = "scope";
const SCOPE_ALL_VALUE = "all";

type MatchFilterContextValue = {
  competitions: CompetitionResponse[];
  competitionsLoading: boolean;
  competitionId: number | null;
  seasonId: number | null;
  teamId: number | null;
  metricMode: MetricMode;
  cxgScopeOnly: boolean;
  setCompetitionId: (id: number | null) => void;
  setSeasonId: (id: number | null) => void;
  setTeamId: (id: number | null) => void;
  setMetricMode: (mode: MetricMode) => void;
  setCxgScopeOnly: (value: boolean) => void;
};

const MatchFilterContext = createContext<MatchFilterContextValue | undefined>(undefined);

export function MatchFilterProvider({ children }: { children: ReactNode }) {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();

  const [competitions, setCompetitions] = useState<CompetitionResponse[]>([]);
  const [competitionsLoading, setCompetitionsLoading] = useState(true);
  const [competitionId, setCompetitionIdState] = useState<number | null>(null);
  const [seasonId, setSeasonIdState] = useState<number | null>(null);
  const [teamId, setTeamId] = useState<number | null>(null);
  const [metricMode, setMetricMode] = useState<MetricMode>("xg");
  const [cxgScopeOnly, setCxgScopeOnly] = useState<boolean>(
    () => searchParams?.get(SCOPE_PARAM) !== SCOPE_ALL_VALUE
  );

  // Keeps the URL's ?scope= param in sync with state, both when the user
  // toggles it and when Explore-zone navigation (a plain <Link>, which
  // doesn't carry query params along) would otherwise drop it.
  useEffect(() => {
    const params = new URLSearchParams(searchParams?.toString());
    const urlWantsAll = params.get(SCOPE_PARAM) === SCOPE_ALL_VALUE;
    const stateWantsAll = !cxgScopeOnly;
    if (urlWantsAll === stateWantsAll) return;
    if (stateWantsAll) {
      params.set(SCOPE_PARAM, SCOPE_ALL_VALUE);
    } else {
      params.delete(SCOPE_PARAM);
    }
    const qs = params.toString();
    router.replace(qs ? `${pathname}?${qs}` : pathname, { scroll: false });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pathname, cxgScopeOnly]);

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
      cxgScopeOnly,
      setCompetitionId,
      setSeasonId,
      setTeamId,
      setMetricMode,
      setCxgScopeOnly,
    }),
    [competitions, competitionsLoading, competitionId, seasonId, teamId, metricMode, cxgScopeOnly]
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
