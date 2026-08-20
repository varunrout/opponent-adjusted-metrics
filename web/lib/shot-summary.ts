import type { ShotResponse } from "@/lib/types";

export function summarizeShots(shots: ShotResponse[]): { shots: number; goals: number; totalXg: number } {
  return {
    shots: shots.length,
    goals: shots.filter((s) => s.is_goal).length,
    totalXg: shots.reduce((sum, s) => sum + (s.statsbomb_xg ?? 0), 0),
  };
}
