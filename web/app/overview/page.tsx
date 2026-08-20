import { PageHead } from "@/components/ui/PageHead";
import { MetricTile } from "@/components/ui/MetricTile";
import { Card } from "@/components/ui/Card";
import { PitchMap } from "@/components/ui/PitchMap";
import { Leaderboard } from "@/components/ui/Leaderboard";
import { ProfileRadar } from "@/components/ui/ProfileRadar";
import type { ShotResponse } from "@/lib/types";

// Static demo shots reproducing the layout skeleton's illustrative pitch dots
// (this page isn't wired to a real match yet — see Matches for live shot data).
const DEMO_HOME_TEAM_ID = 1;
const DEMO_SHOTS: ShotResponse[] = [
  { event_id: "demo-1", match_id: 0, team_id: 1, player_id: null, player_name: null, minute: null, period: null, location_x: 112, location_y: 39, end_x: null, end_y: null, statsbomb_xg: 0.4, outcome_name: "Goal", body_part_name: null, is_goal: true },
  { event_id: "demo-2", match_id: 0, team_id: 1, player_id: null, player_name: null, minute: null, period: null, location_x: 99, location_y: 47, end_x: null, end_y: null, statsbomb_xg: 0.03, outcome_name: null, body_part_name: null, is_goal: false },
  { event_id: "demo-3", match_id: 0, team_id: 1, player_id: null, player_name: null, minute: null, period: null, location_x: 104, location_y: 30, end_x: null, end_y: null, statsbomb_xg: 0, outcome_name: null, body_part_name: null, is_goal: false },
  { event_id: "demo-4", match_id: 0, team_id: 2, player_id: null, player_name: null, minute: null, period: null, location_x: 9, location_y: 43, end_x: null, end_y: null, statsbomb_xg: 0.1, outcome_name: null, body_part_name: null, is_goal: false },
  { event_id: "demo-5", match_id: 0, team_id: 2, player_id: null, player_name: null, minute: null, period: null, location_x: 7, location_y: 40, end_x: null, end_y: null, statsbomb_xg: 0.48, outcome_name: "Goal", body_part_name: null, is_goal: true },
];

export default function OverviewPage() {
  return (
    <section>
      <PageHead title="Overview" crumb="Premier League · 2015/2016" />

      <div className="grid grid-cols-4 gap-3 mb-[18px]">
        <MetricTile label="Team CxG" value="1.82" delta="+0.24 vs xG" deltaTone="pos" />
        <MetricTile label="Team CxA" loading />
        <MetricTile label="Shots" value="14" delta="6 on target" deltaTone="muted" />
        <MetricTile label="Big chances" value="5" delta="+2 vs season avg" deltaTone="pos" />
      </div>

      <div className="grid gap-4 items-start" style={{ gridTemplateColumns: "2fr 1fr" }}>
        <Card title="Shot map">
          <PitchMap shots={DEMO_SHOTS} homeTeamId={DEMO_HOME_TEAM_ID} />
        </Card>

        <div className="flex flex-col gap-3.5">
          <Card title="Leaderboard — CxG per 90">
            <Leaderboard />
          </Card>

          <Card title="Profile radar">
            <ProfileRadar />
          </Card>

          <Card>
            <span className="text-[10px]" style={{ color: "var(--violet)" }}>
              Methodology
            </span>
            <p className="text-[13px] mt-1.5 mb-0">Why CxG moves the number on late-game shots</p>
          </Card>
        </div>
      </div>
    </section>
  );
}
