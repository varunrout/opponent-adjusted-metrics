import { describe, expect, it } from "vitest";

import { deriveFamilies } from "@/lib/analysis-helpers";
import type { FeatureInventoryResponse } from "@/lib/types";

function makeFeature(overrides: Partial<FeatureInventoryResponse>): FeatureInventoryResponse {
  return {
    feature_family: "shot_geometry",
    source_table: "some_table",
    column_name: "some_column",
    data_type: "FLOAT64",
    column_role: "feature",
    is_numeric: true,
    is_categorical: false,
    ...overrides,
  };
}

describe("deriveFamilies", () => {
  it("returns distinct families in first-seen order", () => {
    const features = [
      makeFeature({ feature_family: "shot_geometry", column_name: "a" }),
      makeFeature({ feature_family: "buildup", column_name: "b" }),
      makeFeature({ feature_family: "shot_geometry", column_name: "c" }),
      makeFeature({ feature_family: "opponent_adjusted", column_name: "d" }),
    ];

    expect(deriveFamilies(features)).toEqual(["shot_geometry", "buildup", "opponent_adjusted"]);
  });

  it("returns an empty array for an empty list", () => {
    expect(deriveFamilies([])).toEqual([]);
  });
});
