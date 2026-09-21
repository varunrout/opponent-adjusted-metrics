"use client";

import { useEffect, useState } from "react";
import { Badge } from "@/components/ui/Badge";
import { DivergingBar } from "@/components/ui/DivergingBar";
import { Skeleton } from "@/components/ui/Skeleton";
import { ShotFreezeFrame } from "@/components/shot/ShotFreezeFrame";
import { getShotOpponentContext } from "@/lib/api";
import type { CxaShotCoverageValues, OpponentContextResponse, ShotResponse } from "@/lib/types";

const FEATURE_LABELS: Record<string, string> = {
  nearest_defender_role: "Nearest defender role",
  nearest_defender_style_archetype: "Defender-style archetype",
  gk_odi: "GK distance index",
  nearest_defender_zone_displacement: "Zone displacement",
};

/**
 * Shared shot-detail modal — opens on a PitchMap shot-dot click, reused
 * unchanged across Matches/Players/Teams detail. On open, lazily fetches
 * this one shot's opponent-adjusted context (defender role/distance,
 * archetype, zone displacement) rather than requiring callers to have
 * bulk-fetched it already.
 */
export function ShotDetailModal({
  shot,
  open,
  onClose,
  cxg,
  cxgPlus,
  cxaEvent,
  cxaPlus,
}: {
  shot: ShotResponse | null;
  open: boolean;
  onClose: () => void;
  cxg?: number;
  cxgPlus?: number;
  // The CxA scores of the pass that created this shot, when a test-split
  // chance-creating pass exists for it (~2% of all passes overall -- most
  // shots will have neither of these set). Absent, not a zero/placeholder,
  // when this shot has no such pass on record.
  cxaEvent?: CxaShotCoverageValues;
  cxaPlus?: CxaShotCoverageValues;
}) {
  const [context, setContext] = useState<OpponentContextResponse | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!open || !shot) {
      setContext(null);
      return;
    }
    let cancelled = false;
    setLoading(true);
    setContext(null);
    getShotOpponentContext([shot.event_id])
      .then((rows) => {
        if (!cancelled) setContext(rows[0] ?? null);
      })
      .catch(() => {
        if (!cancelled) setContext(null);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [open, shot]);

  useEffect(() => {
    if (!open) return;
    function handleKey(e: KeyboardEvent) {
      if (e.key === "Escape") onClose();
    }
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [open, onClose]);

  if (!open || !shot) return null;

  const covered = context != null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4"
      style={{ background: "rgba(0,0,0,0.6)" }}
      onClick={onClose}
      data-testid="shot-detail-modal-backdrop"
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-label="Shot detail"
        className="bg-surface border border-border rounded-lg max-w-2xl w-full max-h-[85vh] overflow-y-auto p-4"
        onClick={(e) => e.stopPropagation()}
        data-testid="shot-detail-modal"
      >
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-[13px] font-medium text-text m-0">
            {shot.player_name ?? "Unknown player"} · {shot.minute != null ? `${shot.minute}'` : ""}
          </h2>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close shot detail"
            className="text-text2 hover:text-text text-[18px] leading-none bg-transparent border-none cursor-pointer"
          >
            ×
          </button>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <ShotFreezeFrame shot={shot} context={context} />

          <div className="flex flex-col gap-3">
            <div className="flex items-center gap-2 flex-wrap">
              <Badge
                status={shot.is_goal ? "promoted" : "evaluated"}
                label={shot.outcome_name ?? "Unknown outcome"}
              />
              {shot.body_part_name && (
                <span className="text-[11px] text-text2">{shot.body_part_name}</span>
              )}
            </div>

            <div className="flex items-center gap-4 font-data text-[13px]">
              <span>
                xG <span className="text-text">{(shot.statsbomb_xg ?? 0).toFixed(2)}</span>
              </span>
              {cxg != null && (
                <span>
                  CxG <span style={{ color: "var(--teal)" }}>{cxg.toFixed(2)}</span>
                </span>
              )}
              {cxgPlus != null && (
                <span>
                  CxG+ <span style={{ color: "var(--violet)" }}>{cxgPlus.toFixed(2)}</span>
                </span>
              )}
            </div>

            {cxg != null && (
              <DivergingBar
                left={{ label: "xG", value: shot.statsbomb_xg ?? 0, color: "var(--muted)" }}
                right={{ label: "CxG", value: cxg, color: "var(--teal)" }}
              />
            )}

            {(cxaEvent != null || cxaPlus != null) && (
              <div className="flex flex-col gap-1.5" data-testid="cxa-pass-context">
                <div className="flex items-center gap-2 text-[11px] text-text2 mb-1">
                  <span>Chance-creating pass</span>
                  <Badge status="experimental" label="Experimental" />
                </div>
                {cxaEvent != null && <CxaTrackRow label="CxA" values={cxaEvent} color="var(--teal)" />}
                {cxaPlus != null && <CxaTrackRow label="CxA+" values={cxaPlus} color="var(--violet)" />}
              </div>
            )}

            {loading ? (
              <Skeleton className="h-16" />
            ) : covered && context ? (
              <div className="flex flex-col gap-1.5" data-testid="cxg-plus-features">
                <div className="text-[11px] text-text2 mb-1">CxG+ features</div>
                {context.nearest_defender_role && (
                  <FeatureRow label={FEATURE_LABELS.nearest_defender_role} value={context.nearest_defender_role} />
                )}
                {context.nearest_defender_gap != null && (
                  <FeatureRow
                    label="Nearest defender distance"
                    value={`${context.nearest_defender_gap.toFixed(1)}m`}
                  />
                )}
                {context.nearest_defender_style_archetype && (
                  <FeatureRow
                    label={FEATURE_LABELS.nearest_defender_style_archetype}
                    value={context.nearest_defender_style_archetype}
                  />
                )}
                {context.gk_odi != null && (
                  <FeatureRow label={FEATURE_LABELS.gk_odi} value={context.gk_odi.toFixed(2)} />
                )}
                {context.nearest_defender_zone_displacement != null && (
                  <FeatureRow
                    label={FEATURE_LABELS.nearest_defender_zone_displacement}
                    value={context.nearest_defender_zone_displacement.toFixed(2)}
                    flagged
                  />
                )}
              </div>
            ) : (
              <div className="text-[12px] text-muted" data-testid="event-wide-fallback">
                Event-wide features used instead — this shot falls outside the CxG+ opponent-adjusted
                coverage.
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function CxaTrackRow({
  label,
  values,
  color,
}: {
  label: string;
  values: CxaShotCoverageValues;
  color: string;
}) {
  // p_convert/cxa_combined_score are contractually non-null on this type (this
  // row only renders once the API has confirmed a chance-creating pass
  // produced this shot), but still guarded defensively rather than asserted,
  // matching this file's existing `!= null` style throughout.
  return (
    <div className="flex items-center justify-between text-[11.5px]" data-testid={`cxa-track-row-${label}`}>
      <span className="text-text2">{label}</span>
      <span className="font-data flex items-center gap-2.5" style={{ color }}>
        <span>P(create) {values.p_create_predicted_prob != null ? values.p_create_predicted_prob.toFixed(2) : "—"}</span>
        <span>P(convert) {values.p_convert_predicted_prob != null ? values.p_convert_predicted_prob.toFixed(2) : "—"}</span>
        <span>combined {values.cxa_combined_score != null ? values.cxa_combined_score.toFixed(3) : "—"}</span>
      </span>
    </div>
  );
}

function FeatureRow({ label, value, flagged }: { label: string; value: string; flagged?: boolean }) {
  return (
    <div className="flex items-center justify-between text-[11.5px]">
      <span className="text-text2">{label}</span>
      <span className="font-data" style={{ color: flagged ? "var(--amber)" : "var(--text)" }}>
        {flagged ? "⚠ " : ""}
        {value}
        {flagged ? " unexplained" : ""}
      </span>
    </div>
  );
}
