import type { FeatureInventoryResponse } from "@/lib/types";

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
