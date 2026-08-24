import { describe, expect, it } from "vitest";

import { deriveCxgModelVersions, deriveFamilies, groupCxgResultsByTrack } from "@/lib/analysis-helpers";
import type { CxgModelResultResponse, FeatureInventoryResponse } from "@/lib/types";

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

function makeCxgResult(overrides: Partial<CxgModelResultResponse>): CxgModelResultResponse {
  return {
    model_key: "event_v3",
    track: "cxg_event",
    split: "test",
    model: "v3",
    n: 1000,
    log_loss: 0.2,
    brier_score: 0.05,
    roc_auc: 0.8,
    is_frozen: true,
    is_current: true,
    ...overrides,
  };
}

describe("groupCxgResultsByTrack", () => {
  it("groups rows by track in first-seen order", () => {
    const rows = [
      makeCxgResult({ track: "cxg_event", model_key: "event_v3" }),
      makeCxgResult({ track: "cxg_plus", model_key: "plus_v3" }),
      makeCxgResult({ track: "cxg_event", model_key: "baseline_v1", model: "statsbomb_xg" }),
    ];

    const grouped = groupCxgResultsByTrack(rows);
    expect(grouped.map((g) => g.track)).toEqual(["cxg_event", "cxg_plus"]);
    expect(grouped[0].rows).toHaveLength(2);
    expect(grouped[1].rows).toHaveLength(1);
  });

  it("returns an empty array for an empty list", () => {
    expect(groupCxgResultsByTrack([])).toEqual([]);
  });
});

describe("deriveCxgModelVersions", () => {
  it("derives one distinct entry per model_key, in first-seen order", () => {
    const rows = [
      makeCxgResult({ model_key: "baseline_v1", track: "cxg_event", is_frozen: true, is_current: false }),
      makeCxgResult({ model_key: "baseline_v1", track: "cxg_event", model: "dumb_baseline" }),
      makeCxgResult({ model_key: "event_v3", track: "cxg_event", is_frozen: true, is_current: true }),
      makeCxgResult({ model_key: "plus_v2", track: "cxg_plus", is_frozen: true, is_current: false }),
      makeCxgResult({ model_key: "plus_v3", track: "cxg_plus", is_frozen: true, is_current: true }),
    ];

    expect(deriveCxgModelVersions(rows)).toEqual([
      { model_key: "baseline_v1", track: "cxg_event", is_frozen: true, is_current: false },
      { model_key: "event_v3", track: "cxg_event", is_frozen: true, is_current: true },
      { model_key: "plus_v2", track: "cxg_plus", is_frozen: true, is_current: false },
      { model_key: "plus_v3", track: "cxg_plus", is_frozen: true, is_current: true },
    ]);
  });

  it("returns an empty array for an empty list", () => {
    expect(deriveCxgModelVersions([])).toEqual([]);
  });
});
