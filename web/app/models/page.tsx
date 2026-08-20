import { PageHead } from "@/components/ui/PageHead";
import { ModelCard } from "@/components/ui/ModelCard";
import { MODELS } from "@/lib/models-data";

export default function ModelsPage() {
  return (
    <section>
      <PageHead title="Models" crumb="Registry" />
      <div className="grid gap-3.5" style={{ gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))" }}>
        {MODELS.map((model) => (
          <ModelCard key={model.name} model={model} />
        ))}
      </div>
    </section>
  );
}
