import { beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { OpponentContextResponse, ShotFreezeFrameResponse, ShotResponse } from "@/lib/types";

vi.mock("@/lib/api", () => ({
  getShotOpponentContext: vi.fn(),
  getShotFreezeFrame: vi.fn(),
}));

import { getShotOpponentContext, getShotFreezeFrame } from "@/lib/api";
import { ShotDetailModal } from "@/components/shot/ShotDetailModal";

function makeShot(overrides: Partial<ShotResponse>): ShotResponse {
  return {
    event_id: "shot-1",
    match_id: 1,
    team_id: 10,
    player_id: 100,
    player_name: "Test Player",
    minute: 34,
    period: 1,
    location_x: 105,
    location_y: 38,
    end_x: 120,
    end_y: 40,
    statsbomb_xg: 0.24,
    outcome_name: "Goal",
    body_part_name: "Right Foot",
    is_goal: true,
    ...overrides,
  };
}

function makeContext(overrides: Partial<OpponentContextResponse> = {}): OpponentContextResponse {
  return {
    event_id: "shot-1",
    match_id: 1,
    player_id: 100,
    team_id: 10,
    nearest_defender_odi: 1.1,
    mean_backline_odi: 2.2,
    gk_odi: 3.3,
    defensive_profile_cluster: 2,
    nearest_defender_role: "Center Back",
    nearest_defender_zone_displacement: 0.42,
    nearest_defender_gap: 1.8,
    nearest_defender_style_archetype: "Interceptor",
    has_360_frame: true,
    ...overrides,
  };
}

function makeFreezeFrame(overrides: Partial<ShotFreezeFrameResponse> = {}): ShotFreezeFrameResponse {
  return {
    event_id: "shot-1",
    match_id: 1,
    visible_area: [0, 0, 120, 0, 120, 80, 0, 80],
    players: [
      { ordinal: 0, teammate: true, actor: true, keeper: false, x: 105, y: 38 },
      { ordinal: 1, teammate: true, actor: false, keeper: false, x: 90, y: 40 },
      { ordinal: 2, teammate: false, actor: false, keeper: false, x: 108, y: 39 },
      { ordinal: 3, teammate: false, actor: false, keeper: true, x: 118, y: 40 },
    ],
    ...overrides,
  };
}

describe("ShotDetailModal", () => {
  beforeEach(() => {
    vi.mocked(getShotOpponentContext).mockReset();
    vi.mocked(getShotFreezeFrame).mockReset().mockResolvedValue(null);
  });

  it("renders nothing when closed", () => {
    render(<ShotDetailModal shot={makeShot({})} open={false} onClose={vi.fn()} />);
    expect(screen.queryByTestId("shot-detail-modal")).not.toBeInTheDocument();
  });

  it("shows the covered CxG+ feature list when opponent context exists", async () => {
    vi.mocked(getShotOpponentContext).mockResolvedValue([makeContext({})]);

    render(<ShotDetailModal shot={makeShot({})} open={true} onClose={vi.fn()} cxg={0.3} cxgPlus={0.35} />);

    await waitFor(() => expect(screen.getByTestId("cxg-plus-features")).toBeInTheDocument());
    expect(screen.getByText("Center Back")).toBeInTheDocument();
    expect(screen.getByText("Interceptor")).toBeInTheDocument();
    expect(screen.queryByTestId("event-wide-fallback")).not.toBeInTheDocument();
    // Zone displacement is flagged unexplained, per the design spec.
    expect(screen.getByText(/unexplained/)).toBeInTheDocument();
  });

  it("falls back to the event-wide message when the shot has no opponent context", async () => {
    vi.mocked(getShotOpponentContext).mockResolvedValue([]);

    render(<ShotDetailModal shot={makeShot({})} open={true} onClose={vi.fn()} />);

    await waitFor(() => expect(screen.getByTestId("event-wide-fallback")).toBeInTheDocument());
    expect(screen.queryByTestId("cxg-plus-features")).not.toBeInTheDocument();
  });

  it("calls onClose when the backdrop is clicked", async () => {
    vi.mocked(getShotOpponentContext).mockResolvedValue([]);
    const onClose = vi.fn();
    render(<ShotDetailModal shot={makeShot({})} open={true} onClose={onClose} />);

    await waitFor(() => expect(screen.getByTestId("shot-detail-modal")).toBeInTheDocument());
    fireEvent.click(screen.getByTestId("shot-detail-modal-backdrop"));
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("does not call onClose when the modal content itself is clicked", async () => {
    vi.mocked(getShotOpponentContext).mockResolvedValue([]);
    const onClose = vi.fn();
    render(<ShotDetailModal shot={makeShot({})} open={true} onClose={onClose} />);

    await waitFor(() => expect(screen.getByTestId("shot-detail-modal")).toBeInTheDocument());
    fireEvent.click(screen.getByTestId("shot-detail-modal"));
    expect(onClose).not.toHaveBeenCalled();
  });

  it("calls onClose on Escape", async () => {
    vi.mocked(getShotOpponentContext).mockResolvedValue([]);
    const onClose = vi.fn();
    render(<ShotDetailModal shot={makeShot({})} open={true} onClose={onClose} />);

    await waitFor(() => expect(screen.getByTestId("shot-detail-modal")).toBeInTheDocument());
    fireEvent.keyDown(window, { key: "Escape" });
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  describe("360 freeze frame", () => {
    it("renders teammate, opponent, and GK dots plus the visible-area polygon when a frame exists", async () => {
      vi.mocked(getShotOpponentContext).mockResolvedValue([]);
      vi.mocked(getShotFreezeFrame).mockResolvedValue(makeFreezeFrame());

      render(<ShotDetailModal shot={makeShot({})} open={true} onClose={vi.fn()} />);

      await waitFor(() => expect(screen.getAllByTestId("freeze-frame-teammate").length).toBe(1));
      expect(screen.getAllByTestId("freeze-frame-opponent").length).toBe(1);
      expect(screen.getAllByTestId("freeze-frame-gk").length).toBe(1);
      expect(screen.getByTestId("gk-line")).toBeInTheDocument();
      expect(screen.getByTestId("visible-area")).toBeInTheDocument();
      // The event-wide fallback and freeze-frame dots are independent —
      // this shot has no opponent-context row but still has real 360 dots.
      expect(screen.getByTestId("event-wide-fallback")).toBeInTheDocument();
    });

    it("does not double-draw the actor as a teammate dot", async () => {
      vi.mocked(getShotOpponentContext).mockResolvedValue([]);
      vi.mocked(getShotFreezeFrame).mockResolvedValue(
        makeFreezeFrame({
          players: [
            { ordinal: 0, teammate: true, actor: true, keeper: false, x: 105, y: 38 },
            { ordinal: 1, teammate: true, actor: false, keeper: false, x: 90, y: 40 },
          ],
        })
      );

      render(<ShotDetailModal shot={makeShot({})} open={true} onClose={vi.fn()} />);

      await waitFor(() => expect(screen.getAllByTestId("freeze-frame-teammate").length).toBe(1));
    });

    it("renders no dots, and no error, when the shot has no 360 frame (404 -> null)", async () => {
      vi.mocked(getShotOpponentContext).mockResolvedValue([]);
      vi.mocked(getShotFreezeFrame).mockResolvedValue(null);

      render(<ShotDetailModal shot={makeShot({})} open={true} onClose={vi.fn()} />);

      await waitFor(() => expect(screen.getByTestId("event-wide-fallback")).toBeInTheDocument());
      expect(screen.queryByTestId("freeze-frame-teammate")).not.toBeInTheDocument();
      expect(screen.queryByTestId("freeze-frame-opponent")).not.toBeInTheDocument();
      expect(screen.queryByTestId("freeze-frame-gk")).not.toBeInTheDocument();
      expect(screen.queryByTestId("visible-area")).not.toBeInTheDocument();
    });

    it("swallows a rejected freeze-frame fetch without surfacing an error", async () => {
      vi.mocked(getShotOpponentContext).mockResolvedValue([]);
      vi.mocked(getShotFreezeFrame).mockRejectedValue(new Error("boom"));

      render(<ShotDetailModal shot={makeShot({})} open={true} onClose={vi.fn()} />);

      await waitFor(() => expect(screen.getByTestId("shot-detail-modal")).toBeInTheDocument());
      expect(screen.queryByTestId("freeze-frame-teammate")).not.toBeInTheDocument();
    });
  });
});
