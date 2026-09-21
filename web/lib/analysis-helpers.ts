import type {
  CxgModelResultResponse,
  FeatureInventoryResponse,
  PlayerSeasonQuadrantRow,
} from "@/lib/types";

/**
 * Builds the honest "N of M shots have CxG coverage" caption used on the
 * Matches/Players shot-map cards. Returns null when there's nothing worth
 * saying (no shots, or zero of them covered) rather than a confusing
 * "0 of 0"/"0 of 12" line.
 */
export function describeCxgCoverage(
  totalShots: number,
  coveredCount: number,
  label: string = "CxG"
): string | null {
  if (totalShots === 0 || coveredCount === 0) return null;
  return `${coveredCount} of ${totalShots} shot${totalShots === 1 ? "" : "s"} have ${label} coverage (experimental)`;
}

/**
 * There's no dedicated "list feature families" endpoint — the family
 * selector on the Analysis page derives its options from an unfiltered
 * /v1/analysis/features response by taking the distinct feature_family
 * values, in first-seen order.
 */
export function deriveFamilies(features: FeatureInventoryResponse[]): string[] {
  const seen = new Set<string>();
  const families: string[] = [];
  for (const f of features) {
    if (!seen.has(f.feature_family)) {
      seen.add(f.feature_family);
      families.push(f.feature_family);
    }
  }
  return families;
}

/**
 * Groups CxG model-result rows by track, preserving first-seen track order.
 * The /v1/analysis/cxg-models endpoint returns rows across all model_keys
 * and both tracks in one flat array — the "results table per track" panel
 * groups client-side rather than adding query params to the endpoint.
 */
export function groupCxgResultsByTrack(
  results: CxgModelResultResponse[]
): { track: string; rows: CxgModelResultResponse[] }[] {
  const order: string[] = [];
  const byTrack = new Map<string, CxgModelResultResponse[]>();
  for (const row of results) {
    if (!byTrack.has(row.track)) {
      byTrack.set(row.track, []);
      order.push(row.track);
    }
    byTrack.get(row.track)!.push(row);
  }
  return order.map((track) => ({ track, rows: byTrack.get(track)! }));
}

export type CxgModelVersion = {
  model_key: string;
  track: string;
  is_frozen: boolean;
  is_current: boolean;
};

/**
 * Derives the distinct model_key/track/is_frozen/is_current combinations
 * present in a set of CxG model-result rows, in first-seen order. Used to
 * drive the "version history" list without hardcoding the known model_keys.
 */
export function deriveCxgModelVersions(results: CxgModelResultResponse[]): CxgModelVersion[] {
  const seen = new Set<string>();
  const versions: CxgModelVersion[] = [];
  for (const row of results) {
    if (seen.has(row.model_key)) continue;
    seen.add(row.model_key);
    versions.push({
      model_key: row.model_key,
      track: row.track,
      is_frozen: row.is_frozen,
      is_current: row.is_current,
    });
  }
  return versions;
}

// --- Track B: quadrant scatter (Hard gate 2) ------------------------------
// "Build-your-own" per the design spec's component inventory: a fixed list of
// metrics a caller picks two of for X/Y, rather than one hardcoded pairing.
// Reuses the exact rollup fields /v1/analysis/quadrant-scatter returns —
// mean CxG (per shot) and mean CxA (per chance-creating pass), event and plus
// tracks each — the same metric definitions already computed server-side, no
// new metric invented client-side.

export type QuadrantMetricKey =
  | "cxg_event_mean"
  | "cxg_plus_mean"
  | "cxa_event_mean"
  | "cxa_plus_mean";

export const QUADRANT_METRICS: {
  key: QuadrantMetricKey;
  label: string;
  nField: "cxg_event_n_shots" | "cxg_plus_n_shots" | "cxa_event_n_passes_created" | "cxa_plus_n_passes_created";
  nLabel: string;
}[] = [
  { key: "cxg_event_mean", label: "CxG (mean, per shot)", nField: "cxg_event_n_shots", nLabel: "shots" },
  { key: "cxg_plus_mean", label: "CxG+ (mean, per shot)", nField: "cxg_plus_n_shots", nLabel: "shots" },
  { key: "cxa_event_mean", label: "CxA (mean, per chance-creating pass)", nField: "cxa_event_n_passes_created", nLabel: "passes" },
  { key: "cxa_plus_mean", label: "CxA+ (mean, per chance-creating pass)", nField: "cxa_plus_n_passes_created", nLabel: "passes" },
];

export function quadrantMetricLabel(key: QuadrantMetricKey): string {
  return QUADRANT_METRICS.find((m) => m.key === key)?.label ?? key;
}

/** The `_n` count backing a given metric for one row — used for low-n dimming/
 * tooltip text, never to gate whether the metric itself is null vs 0. */
export function quadrantMetricN(row: PlayerSeasonQuadrantRow, key: QuadrantMetricKey): number {
  const nField = QUADRANT_METRICS.find((m) => m.key === key)!.nField;
  return row[nField];
}

/** Plain median — no interpolation beyond the standard even-count average,
 * matching this project's existing `percentileRank` helper's simplicity
 * (web/app/players/[playerId]/page.tsx) rather than a fancier estimator. */
export function median(values: number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
}
