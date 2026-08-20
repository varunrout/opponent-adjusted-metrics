import type { ModelTier } from "@/lib/models-data";

export function TierChip({ tier }: { tier: ModelTier }) {
  return (
    <span className="text-[10px] text-muted border border-border px-[7px] py-px rounded-full">
      {tier}
    </span>
  );
}
