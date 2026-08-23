import type { CxgModelResultResponse, FeatureInventoryResponse } from "@/lib/types";

/**
 * Builds the honest "N of M shots have CxG coverage" caption used on the
 * Matches/Players shot-map cards. Returns null when there's nothing worth
 * saying (no shots, or zero of them covered) rather than a confusing
 * "0 of 0"/"0 of 12" line.
 */
export function describeCxgCoverage(totalShots: number, coveredCount: number): string | null {
  if (totalShots === 0 || coveredCount === 0) return null;
  return `${coveredCount} of ${totalShots} shot${totalShots === 1 ? "" : "s"} have CxG coverage (experimental)`;
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
