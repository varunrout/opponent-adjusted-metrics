"use client";

import { useEffect, useMemo, useState } from "react";
import { useParams } from "next/navigation";
import { PageHead } from "@/components/ui/PageHead";
import { Card } from "@/components/ui/Card";
import { Skeleton } from "@/components/ui/Skeleton";
import { SimpleTable } from "@/components/ui/SimpleTable";
import { Badge } from "@/components/ui/Badge";
import { CoefficientForestPlot } from "@/components/models/CoefficientForestPlot";
import { getPublicCxaModelSummaries, getPublicCxaExplainability } from "@/lib/api";
import type { CxaExplainability, CxaModelSummary, CxgCoefficientResponse } from "@/lib/types";

const fmtNum = (n: number | null | undefined, digits = 4) => (n == null ? "—" : n.toFixed(digits));

const TRACK_TITLES: Record<string, string> = {
  event: "CxA (event-only)",
  plus: "CxA+",
};

/**
 * Public CxA detail page -- P_create + P_convert breakdown for one track.
 *
 * Deliberately a SEPARATE route/component from /models/[modelKey] (CxG's own
 * detail page), not a generalization of it. CxG has one stage per model_key and
 * always has coefficients; CxA has TWO stages per track (P_create, P_convert)
 * that need to be shown separately, and three of its four sub-models are
 * tree-family (feature importances, no coefficients at all) while only one
 * (CxA+'s P_convert) is logistic. Retrofitting CxG's page to branch on all of
 * that would have turned one component into two, in effect — see
 * docs/analysis/cxa_detail_page_v1.md section 1 for the full reasoning.
 *
 * One parameterized component for both tracks (event, plus) via the [track]
 * route param — not two near-duplicate files, since both tracks' data comes
 * back from the same /v1/models/cxa-models{,/[track]/explainability} shape.
 */
export default function CxaDetailPage() {
  const params = useParams<{ track: string }>();
  const track = params.track;

  const [summaries, setSummaries] = useState<CxaModelSummary[]>([]);
  const [summariesLoading, setSummariesLoading] = useState(true);
  const [summariesError, setSummariesError] = useState(false);

  const [explainability, setExplainability] = useState<CxaExplainability | null>(null);
  const [explainabilityLoading, setExplainabilityLoading] = useState(true);
  const [explainabilityError, setExplainabilityError] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setSummariesLoading(true);
    setSummariesError(false);
    getPublicCxaModelSummaries()
      .then((data) => {
        if (!cancelled) setSummaries(data);
      })
      .catch(() => {
        if (!cancelled) setSummariesError(true);
      })
      .finally(() => {
        if (!cancelled) setSummariesLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    setExplainabilityLoading(true);
    setExplainabilityError(false);
    setExplainability(null);
    getPublicCxaExplainability(track)
      .then((data) => {
        if (!cancelled) setExplainability(data);
      })
      .catch(() => {
        if (!cancelled) setExplainabilityError(true);
      })
      .finally(() => {
        if (!cancelled) setExplainabilityLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [track]);

  const summary = useMemo(() => summaries.find((s) => s.track === track) ?? null, [summaries, track]);

  const pCreateMetrics = useMemo(
    () => summary?.stage_metrics.filter((m) => m.stage === "p_create") ?? [],
    [summary]
  );
  const pConvertMetrics = useMemo(
    () => summary?.stage_metrics.filter((m) => m.stage === "p_convert") ?? [],
    [summary]
  );

  const pCreateImportances = useMemo(
    () => explainability?.feature_importances.filter((f) => f.stage === "p_create") ?? [],
    [explainability]
  );
  const pConvertImportances = useMemo(
    () => explainability?.feature_importances.filter((f) => f.stage === "p_convert") ?? [],
    [explainability]
  );
  const pConvertCoefficients = useMemo(
    () => explainability?.coefficients.filter((c) => c.stage === "p_convert") ?? [],
    [explainability]
  );
  // Adapter, not a shared-component change: CoefficientForestPlot expects
  // CxgCoefficientResponse's shape (model_key/track fields it never reads,
  // per its own source — only .feature/.coefficient are used). Reused as-is
  // rather than forking the component for one extra field name difference.
  const pConvertCoefficientsForPlot: CxgCoefficientResponse[] = useMemo(
    () =>
      pConvertCoefficients.map((c) => ({
        model_key: `cxconvert_${track}`,
        track,
        feature: c.feature,
        coefficient: c.coefficient,
        std_error: c.std_error,
        p_value: c.p_value,
      })),
    [pConvertCoefficients, track]
  );

  const title = TRACK_TITLES[track] ?? `CxA: ${track}`;

  return (
    <section>
      <PageHead title={title} crumb="CxA model registry" />

      {track === "plus" && (
        <div className="flex items-center gap-2 mb-4">
          <Badge status="experimental" label="Experimental" />
          <p className="text-[11.5px] text-muted m-0">
            2,830 total chance-creating passes, 419 in test, zero Premier League rows.
          </p>
        </div>
      )}

      {summariesLoading && <Skeleton style={{ height: 80, marginBottom: 16 }} />}
      {!summariesLoading && summariesError && (
        <p className="text-[12.5px] text-muted mb-4">Couldn&apos;t load model summary. Try again shortly.</p>
      )}
      {!summariesLoading && !summariesError && summary && (
        <Card className="mb-4">
          <p className="text-[12.5px] text-text2 m-0">{summary.combined_score_caveat}</p>
          <p className="text-[11px] text-muted mt-2 mb-0">
            Coverage (test split): {summary.coverage.chance_creating_n} of {summary.coverage.population_n} passes
            (
            {summary.coverage.coverage_pct}
            %) created a chance and were scored by P_convert.
          </p>
          <p className="text-[11px] text-muted mt-2 mb-0">
            <span className="text-text2">Combined-score quality vs. y_goal</span> (worse than P_convert alone — see{" "}
            <code className="font-data">docs/analysis/cxa_combined_scorer_v1.md</code>) is a separate, already-
            documented finding, not shown as a headline metric on this page.
          </p>
        </Card>
      )}

      {/* --- P_create section --- */}
      <Card title={`P_create — ${summary?.p_create_model_family ?? "…"}`} className="mb-4">
        {summariesLoading && <Skeleton style={{ height: 140 }} />}
        {!summariesLoading && !summariesError && summary && (
          <div className="flex flex-col gap-3">
            <SimpleTable
              columns={["Model", "Split", "n", "log_loss", "Brier", "ROC AUC"]}
              rows={pCreateMetrics.map((row) => [
                `${row.model}${row.is_frozen ? " (frozen)" : ""}`,
                row.split,
                String(row.n),
                fmtNum(row.log_loss),
                fmtNum(row.brier_score),
                fmtNum(row.roc_auc),
              ])}
            />
            <p className="text-[11px] text-muted m-0">
              {summary.p_create_feature_list.length} locked feature{summary.p_create_feature_list.length === 1 ? "" : "s"}
              : {summary.p_create_feature_list.join(", ")}
            </p>
            <div>
              <div className="text-[11px] text-muted mb-1.5">
                Feature importance (split / gain — tree model, no coefficients)
              </div>
              {explainabilityLoading && <Skeleton style={{ height: 100 }} />}
              {!explainabilityLoading && explainabilityError && (
                <p className="text-[12px] text-muted m-0">Couldn&apos;t load feature importances.</p>
              )}
              {!explainabilityLoading && !explainabilityError && pCreateImportances.length > 0 && (
                <SimpleTable
                  columns={["Feature", "Split count", "Gain"]}
                  rows={pCreateImportances.map((f) => [f.feature, String(f.importance_split), fmtNum(f.importance_gain, 1)])}
                />
              )}
              {!explainabilityLoading && !explainabilityError && pCreateImportances.length === 0 && (
                <p className="text-[12px] text-muted m-0">No feature-importance data found.</p>
              )}
            </div>
          </div>
        )}
      </Card>

      {/* --- P_convert section --- */}
      <Card title={`P_convert — ${summary?.p_convert_model_family ?? "…"}`} className="mb-4">
        {summariesLoading && <Skeleton style={{ height: 140 }} />}
        {!summariesLoading && !summariesError && summary && (
          <div className="flex flex-col gap-3">
            <SimpleTable
              columns={["Model", "Split", "n", "log_loss", "Brier", "ROC AUC"]}
              rows={pConvertMetrics.map((row) => [
                `${row.model}${row.is_frozen ? " (frozen)" : ""}`,
                row.split,
                String(row.n),
                fmtNum(row.log_loss),
                fmtNum(row.brier_score),
                fmtNum(row.roc_auc),
              ])}
            />
            <p className="text-[11px] text-muted m-0">
              {summary.p_convert_feature_list.length} locked feature{summary.p_convert_feature_list.length === 1 ? "" : "s"}
              : {summary.p_convert_feature_list.join(", ")}
            </p>

            {summary.p_convert_model_family === "logistic_mle" ? (
              <div>
                <div className="text-[11px] text-muted mb-1.5">Coefficients (forest plot, excludes intercept)</div>
                {explainabilityLoading && <Skeleton style={{ height: 100 }} />}
                {!explainabilityLoading && explainabilityError && (
                  <p className="text-[12px] text-muted m-0">Couldn&apos;t load coefficients.</p>
                )}
                {!explainabilityLoading && !explainabilityError && pConvertCoefficients.length > 0 && (
                  <div className="flex flex-col gap-3">
                    <CoefficientForestPlot coefficients={pConvertCoefficientsForPlot} />
                    <SimpleTable
                      columns={["Feature", "Coefficient", "Std. error", "p-value"]}
                      rows={pConvertCoefficients.map((c) => [
                        c.feature,
                        fmtNum(c.coefficient),
                        fmtNum(c.std_error),
                        fmtNum(c.p_value),
                      ])}
                    />
                  </div>
                )}
              </div>
            ) : (
              <div>
                <div className="text-[11px] text-muted mb-1.5">
                  Feature importance (split / gain — tree model, no coefficients)
                </div>
                {explainabilityLoading && <Skeleton style={{ height: 100 }} />}
                {!explainabilityLoading && !explainabilityError && pConvertImportances.length > 0 && (
                  <SimpleTable
                    columns={["Feature", "Split count", "Gain"]}
                    rows={pConvertImportances.map((f) => [f.feature, String(f.importance_split), fmtNum(f.importance_gain, 1)])}
                  />
                )}
              </div>
            )}
          </div>
        )}
      </Card>
    </section>
  );
}
