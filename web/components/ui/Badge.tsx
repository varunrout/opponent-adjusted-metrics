import type { ModelStatus } from "@/lib/models-data";

const STATUS_STYLES: Record<ModelStatus, { background: string; color: string }> = {
  promoted: { background: "rgba(20,184,166,.15)", color: "var(--teal)" },
  training: { background: "rgba(227,160,8,.15)", color: "var(--amber)" },
  planned: { background: "var(--card-hi)", color: "var(--muted)" },
};

export function Badge({ status, label }: { status: ModelStatus; label: string }) {
  const style = STATUS_STYLES[status];
  return (
    <span className="text-[10px] px-2 py-0.5 rounded font-medium" style={style}>
      {label}
    </span>
  );
}
