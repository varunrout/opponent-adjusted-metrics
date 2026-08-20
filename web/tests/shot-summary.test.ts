import { describe, expect, it } from "vitest";

import { summarizeShots } from "@/lib/shot-summary";
import type { ShotResponse } from "@/lib/types";

function makeShot(overrides: Partial<ShotResponse>): ShotResponse {
  return {
    event_id: "evt-1",
    match_id: 1,
    team_id: 10,
    player_id: 1,
    player_name: "Test Player",
    minute: 10,
    period: 1,
    location_x: 100,
    location_y: 40,
    end_x: 120,
    end_y: 40,
    statsbomb_xg: 0.2,
    outcome_name: "Off T",
    body_part_name: "Right Foot",
    is_goal: false,
    ...overrides,
  };
}

describe("summarizeShots", () => {
  it("computes shots, goals, and total xG from a known shot list", () => {
    const shots: ShotResponse[] = [
      makeShot({ event_id: "evt-1", is_goal: true, statsbomb_xg: 0.5 }),
      makeShot({ event_id: "evt-2", is_goal: false, statsbomb_xg: 0.1 }),
      makeShot({ event_id: "evt-3", is_goal: false, statsbomb_xg: 0.05 }),
    ];

    expect(summarizeShots(shots)).toEqual({
      shots: 3,
      goals: 1,
      totalXg: 0.65,
    });
  });

  it("treats a null statsbomb_xg as zero and handles an empty list", () => {
    expect(summarizeShots([])).toEqual({ shots: 0, goals: 0, totalXg: 0 });

    const shots: ShotResponse[] = [makeShot({ event_id: "evt-4", statsbomb_xg: null })];
    expect(summarizeShots(shots)).toEqual({ shots: 1, goals: 0, totalXg: 0 });
  });
});
