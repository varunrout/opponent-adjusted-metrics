import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";

import { PitchMap } from "@/components/ui/PitchMap";
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

const shots: ShotResponse[] = [
  makeShot({ event_id: "evt-1", team_id: 10, is_goal: true, statsbomb_xg: 0.5 }),
  makeShot({ event_id: "evt-2", team_id: 10, is_goal: false, statsbomb_xg: 0.1 }),
  makeShot({ event_id: "evt-3", team_id: 20, is_goal: false, statsbomb_xg: 0.05, location_x: 8, location_y: 40 }),
  makeShot({ event_id: "evt-4", team_id: 20, is_goal: true, statsbomb_xg: 0.8, location_x: 6, location_y: 42 }),
];

describe("PitchMap", () => {
  it("renders exactly one marker per shot", () => {
    render(<PitchMap shots={shots} homeTeamId={10} />);

    expect(screen.getAllByTestId("shot-marker")).toHaveLength(4);
  });
});
