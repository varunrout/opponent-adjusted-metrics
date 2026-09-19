"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { useParams } from "next/navigation";
import { PageHead } from "@/components/ui/PageHead";
import { Card } from "@/components/ui/Card";
import { Skeleton } from "@/components/ui/Skeleton";
import { SimpleTable } from "@/components/ui/SimpleTable";
import { CoefficientForestPlot } from "@/components/models/CoefficientForestPlot";
import { getPublicCxgModelResults, getPublicCxgModelCoefficients } from "@/lib/api";
import { groupCxgResultsByTrack, deriveCxgModelVersions } from "@/lib/analysis-helpers";
import type { CxgCoefficientResponse, CxgModelResultResponse } from "@/lib/types";

const fmtNum = (n: number | null | undefined, digits = 4) => (n == null ? "—" : n.toFixed(digits));

const REAL_MODEL_KEYS = ["baseline_v1", "event_v3", "plus_v2", "plus_v3"];

/**
 * Public "model detail" page — results + version history + coefficients
 * for one of the 4 real, frozen CxG model versions. Deliberately public
 * (no RoleProvider/admin gate): unlike the identical-shaped data behind the
 * admin-only /analysis tab, this is the portfolio-facing surface showing
 * real modelling rigor to any visitor. Data comes from the public
 * /v1/models/cxg-models* endpoints (routers/models.py) — never
 * /v1/analysis/..., which stays admin-gated and untouched by this page.
 */
export default function ModelDetailPage() {
  const params = useParams<{ modelKey: string }>();
  const modelKey = params.modelKey;

  const [results, setResults] = useState<CxgModelResultResponse[]>([]);
  const [resultsLoading, setResultsLoading] = useState(true);
  const [resultsError, setResultsError] = useState(false);

  const [coefficients, setCoefficients] = useState<CxgCoefficientResponse[]>([]);
  const [coefficientsLoading, setCoefficientsLoading] = useState(true);
  const [coefficientsError, setCoefficientsError] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setResultsLoading(true);
    setResultsError(false);
    getPublicCxgModelResults()
      .then((data) => {
        if (!cancelled) setResults(data);
      })
      .catch(() => {
        if (!cancelled) setResultsError(true);
      })
      .finally(() => {
        if (!cancelled) setResultsLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    setCoefficientsLoading(true);
    setCoefficientsError(false);
    setCoefficients([]);
    getPublicCxgModelCoefficients(modelKey)
      .then((data) => {
        if (!cancelled) setCoefficients(data);
      })
      .catch(() => {
        if (!cancelled) setCoefficientsError(true);
      })
      .finally(() => {
        if (!cancelled) setCoefficientsLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [modelKey]);

  const trackGroups = useMemo(() => groupCxgResultsByTrack(results), [results]);
  const versions = useMemo(() => deriveCxgModelVersions(results), [results]);

  return (
    <section>
      <PageHead title={`Model: ${modelKey}`} crumb="CxG model registry" />

      <div className="flex flex-wrap gap-2 mb-4">
        {REAL_MODEL_KEYS.map((key) => {
          const version = versions.find((v) => v.model_key === key);
          return (
            <Link
              key={key}
              href={`/models/${key}`}
              className={[
                "px-2.5 py-[6px] rounded-lg text-[12px] border no-underline",
                key === modelKey
                  ? "border-teal text-text bg-teal/[0.08]"
                  : "border-border text-text2 hover:bg-card-hi",
              ].join(" ")}
            >
              {key}
              {version ? ` · ${version.track}${version.is_current ? " · current" : ""}` : ""}
            </Link>
          );
        })}
      </div>

      <Card title="Results vs StatsBomb baseline, per track" className="mb-4">
        {resultsLoading && <Skeleton style={{ height: 160 }} />}
        {!resultsLoading && resultsError && (
          <p className="text-[12.5px] text-muted m-0">Couldn&apos;t load model results. Try again shortly.</p>
        )}
        {!resultsLoading && !resultsError && trackGroups.length === 0 && (
          <p className="text-[12.5px] text-muted m-0">No model results found.</p>
        )}
        {!resultsLoading && !resultsError && trackGroups.length > 0 && (
          <div className="flex flex-col gap-4">
            {trackGroups.map((group) => (
              <div key={group.track}>
                <div className="text-[11px] text-muted mb-1.5">{group.track}</div>
                <SimpleTable
                  columns={["Model", "Split", "n", "log_loss", "Brier", "ROC AUC"]}
                  rows={group.rows.map((row) => [
                    `${row.model}${row.is_current && row.model !== "statsbomb_xg" ? " (current)" : ""}`,
                    row.split,
                    String(row.n),
                    fmtNum(row.log_loss),
                    fmtNum(row.brier_score),
                    fmtNum(row.roc_auc),
                  ])}
                />
              </div>
            ))}
          </div>
        )}
      </Card>

      <Card
        title={`Coefficients — ${modelKey}`}
        className="mb-4"
      >
        {coefficientsLoading && <Skeleton style={{ height: 160 }} />}
        {!coefficientsLoading && coefficientsError && (
          <p className="text-[12.5px] text-muted m-0">Couldn&apos;t load coefficients. Try again shortly.</p>
        )}
        {!coefficientsLoading && !coefficientsError && coefficients.length === 0 && (
          <p className="text-[12.5px] text-muted m-0">No coefficients found for this model.</p>
        )}
        {!coefficientsLoading && !coefficientsError && coefficients.length > 0 && (
          <div className="flex flex-col gap-4">
            <p className="text-[11px] text-muted m-0">
              {coefficients.length} coefficient row{coefficients.length === 1 ? "" : "s"} — includes the intercept,
              one-hot dummy columns, and interaction terms, so this is higher than a plain &quot;feature count&quot;
              (see caveats below).
              {coefficients.every((c) => c.std_error == null) && (
                <> Standard errors and p-values are unavailable for this version — shown as &quot;—&quot;, not zero.</>
              )}
            </p>

            <div>
              <div className="text-[11px] text-muted mb-1.5">Forest plot (excludes the intercept)</div>
              <CoefficientForestPlot coefficients={coefficients} />
            </div>

            <SimpleTable
              columns={["Feature", "Coefficient", "Std. error", "p-value"]}
              rows={coefficients.map((c) => [c.feature, fmtNum(c.coefficient), fmtNum(c.std_error), fmtNum(c.p_value)])}
            />
          </div>
        )}
      </Card>

      <Card title="Known caveats">
        <div className="flex flex-col gap-3 text-[12.5px] text-text2">
          <p className="m-0">
            <span className="text-text font-semibold">Feature-pool asymmetry.</span> Event-wide uses 8 base
            features; CxG+ uses 24. The two tracks aren&apos;t a like-for-like comparison of &quot;does adding 360
            data help&quot; — some of the metric gap between tracks reflects feature count, not just data richness.
          </p>
          <p className="m-0">
            <span className="text-text font-semibold">
              <code className="font-data">zone_displacement</code>&apos;s unexplained bimodality.
            </span>{" "}
            This CxG+ feature shows a bimodal distribution with no documented cause yet — an open question, not
            resolved, and not something to present as a clean, well-understood feature.
          </p>
          <p className="m-0 text-muted">
            Raw coefficient-row counts (11 / 28 / 54 / 58 across the four model versions) are higher than
            &quot;feature count&quot; because they include one-hot-encoded categorical dummy columns, a
            missingness-indicator column, an intercept (<code className="font-data">const</code>), and interaction
            terms (feature names containing <code className="font-data">:</code>).
          </p>
        </div>
      </Card>
    </section>
  );
}
