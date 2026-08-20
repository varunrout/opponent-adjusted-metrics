import { PageHead } from "@/components/ui/PageHead";
import { Skeleton } from "@/components/ui/Skeleton";

export type PlaceholderVariant = "table" | "cards" | "block" | "text";

type PlaceholderPanelProps = {
  title: string;
  variant: PlaceholderVariant;
  /** Row count for "table" or card count for "cards". Ignored for block/text. */
  count?: number;
};

// Ports the skeleton's genericTemplates (skelTableRows / skelCardGrid / skelBlock / skelText)
// from docs/dashboard_layout_skeleton.html into one shared, data-driven placeholder.
export function PlaceholderPanel({ title, variant, count = 6 }: PlaceholderPanelProps) {
  return (
    <section>
      <PageHead title={title} crumb="Not yet built" />
      {variant === "table" && <SkelTableRows n={count} />}
      {variant === "cards" && <SkelCardGrid n={count} />}
      {variant === "block" && <SkelBlock />}
      {variant === "text" && <SkelText />}
    </section>
  );
}

function SkelTableRows({ n }: { n: number }) {
  return (
    <div className="bg-card border border-border rounded p-3.5 px-4">
      {Array.from({ length: n }).map((_, i) => (
        <div
          key={i}
          className="grid gap-3 items-center py-[9px] border-b border-border last:border-b-0"
          style={{ gridTemplateColumns: "70px 1fr 60px 60px" }}
        >
          <Skeleton style={{ height: 12 }} />
          <Skeleton style={{ height: 12 }} />
          <Skeleton style={{ height: 12 }} />
          <Skeleton style={{ height: 12 }} />
        </div>
      ))}
    </div>
  );
}

function SkelCardGrid({ n }: { n: number }) {
  return (
    <div className="grid gap-3.5" style={{ gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))" }}>
      {Array.from({ length: n }).map((_, i) => (
        <div key={i} className="bg-card border border-border rounded p-3.5">
          <Skeleton className="rounded-full" style={{ width: 34, height: 34, marginBottom: 10 }} />
          <Skeleton style={{ height: 11, width: "70%", marginBottom: 6 }} />
          <Skeleton style={{ height: 11, width: "45%" }} />
        </div>
      ))}
    </div>
  );
}

function SkelBlock() {
  return (
    <div className="bg-card border border-border rounded p-3.5" style={{ height: 260 }}>
      <Skeleton style={{ height: "100%" }} />
    </div>
  );
}

function SkelText() {
  return (
    <div className="bg-card border border-border rounded p-3.5 px-4">
      <Skeleton style={{ height: 11, width: "90%", marginBottom: 10 }} />
      <Skeleton style={{ height: 11, width: "80%", marginBottom: 10 }} />
      <Skeleton style={{ height: 11, width: "60%" }} />
    </div>
  );
}
