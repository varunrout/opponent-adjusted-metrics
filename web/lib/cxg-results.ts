// Six-row CxG v3 vs StatsBomb xG comparison, per docs/dashboard_design_spec_v2.md
// §11. Numbers copied verbatim from that table — do not recompute or round
// differently here. Shared by the Overview page and the
// cxg-v3-honest-comparison story article so the two never drift apart.
export type CxgComparisonRow = {
  track: string;
  metric: string;
  cxg: string;
  xg: string;
};

export const CXG_COMPARISON: CxgComparisonRow[] = [
  { track: "Event-wide (cxg_event)", metric: "log_loss", cxg: "0.3003", xg: "0.2597" },
  { track: "Event-wide (cxg_event)", metric: "Brier", cxg: "0.0852", xg: "0.0718" },
  { track: "Event-wide (cxg_event)", metric: "AUC", cxg: "0.7148", xg: "0.7972" },
  { track: "CxG+", metric: "log_loss", cxg: "0.2555", xg: "0.2430" },
  { track: "CxG+", metric: "Brier", cxg: "0.0713", xg: "0.0665" },
  { track: "CxG+", metric: "AUC", cxg: "0.8313", xg: "0.8476" },
];
