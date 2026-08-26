"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { PageHead } from "@/components/ui/PageHead";
import { Card } from "@/components/ui/Card";
import { Skeleton } from "@/components/ui/Skeleton";
import { ClickableRow } from "@/components/ui/ClickableRow";
import { TeamLink } from "@/components/ui/EntityLink";
import { StoryCard } from "@/components/ui/StoryCard";
import { STORIES } from "@/lib/stories-data";
import { getMatches } from "@/lib/api";
import type { MatchResponse } from "@/lib/types";

const GITHUB_URL = "https://github.com/varunrout/opponent-adjusted-metrics";

// Six-row CxG vs StatsBomb comparison, per docs/dashboard_design_spec_v2.md §11.
// Numbers copied verbatim — do not recompute or round differently here.
const CXG_COMPARISON = [
  { track: "Event-wide (cxg_event)", metric: "log_loss", cxg: "0.3003", xg: "0.2597" },
  { track: "Event-wide (cxg_event)", metric: "Brier", cxg: "0.0852", xg: "0.0718" },
  { track: "Event-wide (cxg_event)", metric: "AUC", cxg: "0.7148", xg: "0.7972" },
  { track: "CxG+", metric: "log_loss", cxg: "0.2555", xg: "0.2430" },
  { track: "CxG+", metric: "Brier", cxg: "0.0713", xg: "0.0665" },
  { track: "CxG+", metric: "AUC", cxg: "0.8313", xg: "0.8476" },
];

const featuredStory = STORIES.find((s) => s.slug === "cxg-v3-honest-comparison") ?? STORIES[0];

function RecentMatchesStrip() {
  const [matches, setMatches] = useState<MatchResponse[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(false);
    getMatches({})
      .then((data) => {
        if (cancelled) return;
        const sorted = [...data].sort((a, b) => {
          const da = a.match_date ? Date.parse(a.match_date) : 0;
          const db = b.match_date ? Date.parse(b.match_date) : 0;
          return db - da;
        });
        setMatches(sorted.slice(0, 6));
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
  }, [attempt]);

  if (loading) {
    return (
      <div className="bg-card border border-border rounded overflow-hidden">
        {Array.from({ length: 6 }).map((_, i) => (
          <div key={i} className="px-4 py-[10px] border-b border-border last:border-b-0">
            <Skeleton style={{ height: 14 }} />
          </div>
        ))}
      </div>
    );
  }

  if (error) {
    return (
      <Card className="border-l-[3px]" style={{ borderLeftColor: "var(--amber)" }}>
        <p className="text-[12.5px] text-muted m-0 mb-2">Couldn&apos;t load recent matches.</p>
        <button
          type="button"
          onClick={() => setAttempt((n) => n + 1)}
          className="text-[12px] text-teal hover:underline"
        >
          Retry
        </button>
      </Card>
    );
  }

  if (matches.length === 0) {
    return (
      <Card>
        <p className="text-[12.5px] text-muted m-0">No matches available.</p>
      </Card>
    );
  }

  return (
    <div className="bg-card border border-border rounded overflow-hidden">
      {matches.map((match) => (
        <ClickableRow
          key={match.match_id}
          href={`/matches/${match.match_id}`}
          className="flex items-center justify-between gap-3 px-4 py-[10px] border-b border-border last:border-b-0 text-[12.5px] hover:bg-card-hi cursor-pointer"
        >
          <div className="text-muted w-24">{match.match_date ?? ""}</div>
          <div className="flex-1 min-w-0 text-right">
            <TeamLink teamId={match.home_team_id} name={match.home_team_name ?? "TBD"} className="text-text" />
          </div>
          <div className="font-data text-text2 w-16 text-center">
            {match.home_score ?? "-"} : {match.away_score ?? "-"}
          </div>
          <div className="flex-1 min-w-0">
            <TeamLink teamId={match.away_team_id} name={match.away_team_name ?? "TBD"} className="text-text" />
          </div>
        </ClickableRow>
      ))}
    </div>
  );
}

export default function OverviewPage() {
  return (
    <section>
      <PageHead title="Overview" crumb="Opponent-adjusted expected goals, built and evaluated in the open" />

      {/* Row 0 — statement block */}
      <div className="mb-[18px]">
        <h1 className="text-2xl font-semibold m-0 mb-2">Opponent-Adjusted Metrics</h1>
        <p className="text-[13.5px] text-text2 m-0 mb-2 max-w-[720px]">
          A from-scratch expected-goals pipeline that adjusts a shot&apos;s value for the defensive context
          around it — nearest defender, backline shape, goalkeeper positioning — built and evaluated end to
          end by Varun Rout.
        </p>
        <p className="text-[12px] text-muted m-0">
          Data: StatsBomb Open Data · 3 competitions · 5 seasons ·{" "}
          <a href={GITHUB_URL} target="_blank" rel="noreferrer" className="text-teal hover:underline">
            source on GitHub
          </a>
        </p>
      </div>

      {/* Row 2 — recent matches strip */}
      <div className="mb-[18px]">
        <h3 className="text-[12.5px] font-medium text-text2 m-0 mb-2.5">Recent matches</h3>
        <RecentMatchesStrip />
      </div>

      {/* Row 3 — two-column split */}
      <div className="grid gap-4 items-start mb-[18px]" style={{ gridTemplateColumns: "3fr 2fr" }}>
        <Card title="The honest result">
          <div className="overflow-x-auto">
            <table className="w-full text-[12px] border-collapse">
              <thead>
                <tr className="text-muted text-left">
                  <th className="font-normal pb-1.5 pr-3">Track</th>
                  <th className="font-normal pb-1.5 pr-3">Metric</th>
                  <th className="font-normal pb-1.5 pr-3 text-right">CxG v3</th>
                  <th className="font-normal pb-1.5 text-right">StatsBomb xG</th>
                </tr>
              </thead>
              <tbody>
                {CXG_COMPARISON.map((row, i) => (
                  <tr key={i} className="border-t border-border">
                    <td className="py-1.5 pr-3 text-text2">{row.track}</td>
                    <td className="py-1.5 pr-3 text-text2">{row.metric}</td>
                    <td className="py-1.5 pr-3 text-right font-data" style={{ color: "var(--red)" }}>
                      {row.cxg}
                    </td>
                    <td className="py-1.5 text-right font-data text-text2">{row.xg}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="text-[12.5px] text-text2 mt-3 mb-0">
            CxG v3 trails the StatsBomb baseline on every metric, on both tracks. That result is published
            here rather than hidden, because a candidate willing to show a negative result with its numbers
            intact is a stronger credibility signal than a cherry-picked win.
          </p>
          <Link href="/stories/cxg-v3-honest-comparison" className="text-teal hover:underline text-[12.5px]">
            Read the full comparison →
          </Link>
        </Card>

        <div className="flex flex-col gap-3.5">
          <Card title="Built with">
            <ul className="text-[12.5px] text-text2 m-0 pl-4">
              <li>FastAPI</li>
              <li>BigQuery</li>
              <li>Next.js</li>
              <li>Firebase Auth</li>
              <li>Cloud Run</li>
            </ul>
          </Card>

          <StoryCard story={featuredStory} />
        </div>
      </div>

      {/* Row 4 — entry points */}
      <div className="grid grid-cols-3 gap-3.5">
        <Link href="/matches" className="block">
          <Card className="h-full hover:bg-card-hi transition-colors">
            <p className="text-[13px] m-0 mb-1">Browse matches</p>
            <p className="text-[11.5px] text-muted m-0">610 matches across 3 competitions and 5 seasons.</p>
          </Card>
        </Link>
        <Link href="/players" className="block">
          <Card className="h-full hover:bg-card-hi transition-colors">
            <p className="text-[13px] m-0 mb-1">Explore players</p>
            <p className="text-[11.5px] text-muted m-0">
              1,457 players, ranked by shots, xG, and who&apos;s beating their expected goals.
            </p>
          </Card>
        </Link>
        <Link href="/stories" className="block">
          <Card className="h-full hover:bg-card-hi transition-colors">
            <p className="text-[13px] m-0 mb-1">Read the methodology</p>
            <p className="text-[11.5px] text-muted m-0">
              How CxG was built, what worked, what didn&apos;t, and why it&apos;s published either way.
            </p>
          </Card>
        </Link>
      </div>
    </section>
  );
}
