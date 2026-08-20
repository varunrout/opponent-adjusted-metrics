import { PageHead } from "@/components/ui/PageHead";
import { MetricTile } from "@/components/ui/MetricTile";
import { Card } from "@/components/ui/Card";
import { PitchMap } from "@/components/ui/PitchMap";
import { Leaderboard } from "@/components/ui/Leaderboard";
import { ProfileRadar } from "@/components/ui/ProfileRadar";

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
          <PitchMap />
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
