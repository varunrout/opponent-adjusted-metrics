import { PrimaryNav } from "@/components/shell/PrimaryNav";
import { RoleSwitch } from "@/components/shell/RoleSwitch";

export function TopBar() {
  return (
    <div className="h-14 flex items-center justify-between px-5 bg-surface border-b border-border sticky top-0 z-10">
      <div className="flex items-center gap-2.5">
        <div className="w-[26px] h-[26px] rounded-md bg-teal flex items-center justify-center font-bold text-[13px]" style={{ color: "#04231f" }}>
          OA
        </div>
        <div className="text-[13px] text-text2">Opponent-Adjusted Metrics</div>
      </div>

      <PrimaryNav />

      <div className="flex items-center gap-3">
        <RoleSwitch />
        <div
          className="w-7 h-7 rounded-full bg-violet flex items-center justify-center text-[11px] font-semibold"
          style={{ color: "#1c1030" }}
        >
          VR
        </div>
      </div>
    </div>
  );
}
