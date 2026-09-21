import { Card } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import type { CxaRollup } from "@/lib/types";

/**
 * Shared CxA summary card for the Players/Teams pages — the event track shown
 * plain, CxA+ shown with the same Experimental badge treatment used
 * everywhere else CxA+ appears (Models page, CxA detail page, the per-shot
 * display). Absent entirely (renders nothing) when both tracks have zero
 * chance-creating-pass coverage for this player/team-season — never a 0.00
 * placeholder tile, matching this project's established null-vs-zero
 * discipline.
 */
export function CxaSummaryCard({
  title,
  event,
  plus,
}: {
  title: string;
  event: CxaRollup;
  plus: CxaRollup;
}) {
  if (event.n === 0 && plus.n === 0) return null;

  return (
    <Card title={title} className="mt-4">
      <div className="flex flex-col gap-3">
        {event.n > 0 && event.mean != null && (
          <div className="flex items-center justify-between gap-3 flex-wrap">
            <span className="text-[11px] text-muted">CxA — mean per chance-creating pass</span>
            <span className="font-data text-[15px] text-text">
              {event.mean.toFixed(3)}{" "}
              <span className="text-[11px] text-muted">
                (n={event.n} test-split pass{event.n === 1 ? "" : "es"})
              </span>
            </span>
          </div>
        )}
        {plus.n > 0 && plus.mean != null && (
          <div className="flex items-center justify-between gap-3 flex-wrap">
            <span className="text-[11px] text-muted flex items-center gap-1.5">
              CxA+ — mean per chance-creating pass
              <Badge status="experimental" label="Experimental" />
            </span>
            <span className="font-data text-[15px] text-text">
              {plus.mean.toFixed(3)}{" "}
              <span className="text-[11px] text-muted">
                (n={plus.n} test-split pass{plus.n === 1 ? "" : "es"})
              </span>
            </span>
          </div>
        )}
      </div>
    </Card>
  );
}
