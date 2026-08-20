import type { ModelInfo } from "@/lib/models-data";
import { Badge } from "@/components/ui/Badge";
import { TierChip } from "@/components/ui/TierChip";
import { Skeleton } from "@/components/ui/Skeleton";

export function ModelCard({ model }: { model: ModelInfo }) {
  return (
    <div className="bg-card border border-border rounded p-3.5">
      <div className="flex justify-between items-center">
        <span className="text-[13px]">{model.name}</span>
        <Badge status={model.status} label={model.statusLabel} />
      </div>
      <div className="mt-2">
        <TierChip tier={model.tier} />
      </div>
      {model.caption ? (
        <p className="text-[11.5px] text-muted mt-2.5 mb-0">{model.caption}</p>
      ) : (
        <Skeleton style={{ height: 12, width: "65%", marginTop: 12 }} />
      )}
    </div>
  );
}
