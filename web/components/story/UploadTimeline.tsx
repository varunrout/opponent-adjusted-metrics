// Two-column "claimed vs actual" upload sequence — what an incident note
// described side by side with what the code trace + real timestamps showed.
export function UploadTimeline({
  claimed,
  actual,
}: {
  claimed: { label: string }[];
  actual: { label: string; detail?: string }[];
}) {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
      <div>
        <div className="text-[10.5px] text-muted mb-2 uppercase tracking-wide">Incident note claimed</div>
        <ol className="flex flex-col gap-2 m-0 pl-0 list-none" data-testid="upload-timeline-claimed">
          {claimed.map((step, i) => (
            <li key={i} className="flex items-start gap-2 text-[11.5px] text-text2">
              <span
                className="shrink-0 w-4 h-4 rounded-full flex items-center justify-center text-[9px] font-data"
                style={{ background: "rgba(239,68,68,.15)", color: "var(--red)" }}
              >
                {i + 1}
              </span>
              {step.label}
            </li>
          ))}
        </ol>
      </div>

      <div>
        <div className="text-[10.5px] text-muted mb-2 uppercase tracking-wide">Code trace showed</div>
        <ol className="flex flex-col gap-2 m-0 pl-0 list-none" data-testid="upload-timeline-actual">
          {actual.map((step, i) => (
            <li key={i} className="flex items-start gap-2 text-[11.5px] text-text2">
              <span
                className="shrink-0 w-4 h-4 rounded-full flex items-center justify-center text-[9px] font-data"
                style={{ background: "rgba(20,184,166,.15)", color: "var(--teal)" }}
              >
                {i + 1}
              </span>
              <span>
                {step.label}
                {step.detail && <span className="font-data text-muted"> — {step.detail}</span>}
              </span>
            </li>
          ))}
        </ol>
      </div>
    </div>
  );
}
