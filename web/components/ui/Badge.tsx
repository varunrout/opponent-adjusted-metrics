import type { ModelStatus } from "@/lib/models-data";

export type BadgeStatus = ModelStatus | "experimental";

const STATUS_STYLES: Record<BadgeStatus, { background: string; color: string }> = {
  promoted: { background: "rgba(20,184,166,.15)", color: "var(--teal)" },
  training: { background: "rgba(227,160,8,.15)", color: "var(--amber)" },
  // Deliberately neutral, not teal/green — "evaluated" means real results
  // exist and are honestly compared to baseline, not "shipped to
  // production" (that's what "promoted" implies). Per the design system's
  // colour discipline, teal/violet/amber are reserved as metric-family
  // identities and green/red as outcome semantics, so this status gets
  // a plain text2-toned badge rather than borrowing either.
  evaluated: { background: "rgba(151,161,173,.14)", color: "var(--text2)" },
  planned: { background: "var(--card-hi)", color: "var(--muted)" },
  // Per docs/dashboard_content_spec_v3.md §8.1: every CxG value/coverage
  // caption needs a *visible* disclosure badge, not a hover-only <title>
  // (which fails entirely on touch devices). --amber is the design
  // system's "warning / experimental" token (globals design tokens, v3
  // §1.4) — same styling as "training" is deliberate, both are pre-
  // production states, just distinguished by label text.
  experimental: { background: "rgba(227,160,8,.15)", color: "var(--amber)" },
};

export function Badge({ status, label }: { status: BadgeStatus; label: string }) {
  const style = STATUS_STYLES[status];
  return (
    <span className="text-[10px] px-2 py-0.5 rounded font-medium" style={style}>
      {label}
    </span>
  );
}
