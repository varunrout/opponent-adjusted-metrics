import type { ModelInfo } from "@/lib/models-data";
import { Badge } from "@/components/ui/Badge";
import { TierChip } from "@/components/ui/TierChip";
import { Skeleton } from "@/components/ui/Skeleton";

export function ModelCard({ model }: { model: ModelInfo }) {
  const hasContent = model.validationMetrics.length > 0 || Boolean(model.featureFamilyCount);

  return (
    <div className="bg-card border border-border rounded p-3.5">
      <div className="flex justify-between items-center">
        <span className="text-[13px]">{model.name}</span>
        <Badge status={model.status} label={model.statusLabel} />
      </div>

      <div className="mt-2">
        <TierChip tier={model.tier} />
      </div>

      {model.validationMetrics.length > 0 && (
        <dl className="mt-2.5 flex flex-col gap-1">
          {model.validationMetrics.map((metric) => (
            <div key={metric.label} className="flex justify-between gap-2">
              <dt className="text-[11.5px] text-muted m-0">{metric.label}</dt>
              <dd className="text-[11.5px] text-text2 font-data m-0">{metric.value}</dd>
            </div>
          ))}
        </dl>
      )}

      {model.featureFamilyCount ? (
        <p className="text-[11.5px] text-muted mt-2 mb-0">{model.featureFamilyCount}</p>
      ) : (
        !hasContent && <Skeleton style={{ height: 12, width: "65%", marginTop: 12 }} />
      )}
    </div>
  );
}
