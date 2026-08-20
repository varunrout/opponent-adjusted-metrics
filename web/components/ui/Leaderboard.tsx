import { Skeleton } from "@/components/ui/Skeleton";

type LeaderboardRow = {
  rank: number;
  pct: number;
  value: string;
};

const rows: LeaderboardRow[] = [
  { rank: 1, pct: 88, value: "0.71" },
  { rank: 2, pct: 74, value: "0.58" },
  { rank: 3, pct: 61, value: "0.49" },
];

const loadingRows = [4, 5];

export function Leaderboard() {
  return (
    <div>
      {rows.map((row) => (
        <div key={row.rank} className="flex items-center gap-2.5 py-[7px] border-b border-border text-[12.5px] font-data">
          <span className="w-5 text-muted">{row.rank}</span>
          <div className="flex-1 h-1.5 bg-card-hi rounded-full overflow-hidden">
            <div className="h-full bg-teal" style={{ width: `${row.pct}%` }} />
          </div>
          <span className="w-11 text-right text-text2">{row.value}</span>
        </div>
      ))}
      {loadingRows.map((rank) => (
        <div key={rank} className="flex items-center gap-2.5 py-[7px] border-b border-border last:border-b-0 text-[12.5px] font-data">
          <span className="w-5 text-muted">{rank}</span>
          <Skeleton className="flex-1" style={{ height: 6 }} />
          <span className="w-11 text-right text-text2">&nbsp;</span>
        </div>
      ))}
    </div>
  );
}
