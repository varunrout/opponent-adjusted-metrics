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

  it("discloses CxG via a title only on covered shots", () => {
    const { container } = render(
      <PitchMap shots={shots} homeTeamId={10} cxgByEventId={{ "evt-1": 0.42 }} />
    );

    const markers = container.querySelectorAll('[data-testid="shot-marker"]');
    expect(markers).toHaveLength(4);

    const titles = Array.from(markers).map((marker) => marker.querySelector("title")?.textContent ?? null);

    // evt-1 is the first shot in `shots` and is the only one present in
    // cxgByEventId — it should carry the disclosure title.
    expect(titles[0]).toContain("Experimental");
    expect(titles[0]).toContain("CxG");

    // The remaining shots (evt-2, evt-3, evt-4) have no coverage entry and
    // must render with no title at all.
    expect(titles.slice(1)).toEqual([null, null, null]);
  });

  it("discloses CxG+ independently of CxG, and both together when a shot has both", () => {
    const { container } = render(
      <PitchMap
        shots={shots}
        homeTeamId={10}
        cxgByEventId={{ "evt-1": 0.42 }}
        cxgPlusByEventId={{ "evt-1": 0.55, "evt-2": 0.3 }}
      />
    );

    const markers = container.querySelectorAll('[data-testid="shot-marker"]');
    const titles = Array.from(markers).map((marker) => marker.querySelector("title")?.textContent ?? null);

    // evt-1 has both tracks covered — both values must appear in one title.
    expect(titles[0]).toContain("CxG 0.42");
    expect(titles[0]).toContain("CxG+ 0.55");
    expect(titles[0]).toContain("Experimental");

    // evt-2 has only CxG+ coverage — its title must disclose CxG+ but not a
    // CxG value (no placeholder for the uncovered track).
    expect(titles[1]).toContain("CxG+ 0.30");
    expect(titles[1]).not.toContain("CxG 0.");

    // evt-3/evt-4 have no coverage on either track.
    expect(titles.slice(2)).toEqual([null, null]);
  });
});

describe("PitchMap legend", () => {
  it("is hidden by default", () => {
    render(<PitchMap shots={shots} homeTeamId={10} />);
    expect(screen.queryByTestId("pitch-map-legend")).not.toBeInTheDocument();
  });

  it("shows home/away colour, goal, and dot-size explanation when showLegend is set", () => {
    render(<PitchMap shots={shots} homeTeamId={10} showLegend />);
    const legend = screen.getByTestId("pitch-map-legend");
    expect(legend).toHaveTextContent("Home");
    expect(legend).toHaveTextContent("Away");
    expect(legend).toHaveTextContent("Filled = goal");
    expect(legend).toHaveTextContent("Dot size = xG");
  });

  it("labels dot size as CxG when sizeBy is cxg", () => {
    render(<PitchMap shots={shots} homeTeamId={10} showLegend sizeBy="cxg" />);
    expect(screen.getByTestId("pitch-map-legend")).toHaveTextContent("Dot size = CxG");
  });
});

describe("PitchMap sizeBy (display-mode toggle, per content_spec_v3.md §2.2)", () => {
  it("defaults to sizing by xG when sizeBy is omitted", () => {
    const { container } = render(<PitchMap shots={shots} homeTeamId={10} cxgByEventId={{ "evt-1": 0.9 }} />);
    const marker = container.querySelector('[data-testid="shot-marker"]');
    // evt-1 has statsbomb_xg 0.5 — radius should reflect xG (0.5), not the
    // much larger covered CxG value (0.9), when sizeBy isn't "cxg".
    expect(marker?.getAttribute("data-cxg-uncovered")).toBeNull();
  });

  it('sizes covered shots by CxG and marks uncovered shots as data-cxg-uncovered when sizeBy="cxg"', () => {
    const { container } = render(
      <PitchMap shots={shots} homeTeamId={10} cxgByEventId={{ "evt-1": 0.9 }} sizeBy="cxg" />
    );
    const markers = Array.from(container.querySelectorAll('[data-testid="shot-marker"]'));

    // evt-1 is covered — not flagged uncovered.
    expect(markers[0].getAttribute("data-cxg-uncovered")).toBeNull();
    // evt-2/evt-3/evt-4 have no cxg_event coverage — flagged uncovered,
    // rendered at reduced opacity with a dashed stroke, never substituting xG.
    for (const marker of markers.slice(1)) {
      expect(marker.getAttribute("data-cxg-uncovered")).toBe("true");
      expect(marker.getAttribute("fill-opacity")).toBe("0.12");
      expect(marker.getAttribute("stroke-dasharray")).toBeTruthy();
    }
  });

  it('never flags shots as uncovered when sizeBy="xg", regardless of coverage', () => {
    const { container } = render(<PitchMap shots={shots} homeTeamId={10} sizeBy="xg" />);
    const markers = container.querySelectorAll('[data-testid="shot-marker"]');
    for (const marker of Array.from(markers)) {
      expect(marker.getAttribute("data-cxg-uncovered")).toBeNull();
    }
  });
});
