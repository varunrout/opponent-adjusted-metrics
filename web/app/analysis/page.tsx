"use client";

import { useEffect, useMemo, useState } from "react";
import { PageHead } from "@/components/ui/PageHead";
import { Card } from "@/components/ui/Card";
import { Skeleton } from "@/components/ui/Skeleton";
import { useRole } from "@/components/shell/RoleProvider";
import {
  ApiError,
  getAnalysisBivariate,
  getAnalysisCharts,
  getAnalysisCorrelation,
  getAnalysisFeatures,
  getAnalysisPca,
  getAnalysisUnivariate,
} from "@/lib/api";
import { deriveFamilies } from "@/lib/analysis-helpers";
import type {
  BivariateResponse,
  ChartsResponse,
  FeatureCorrelationResponse,
  FeatureInventoryResponse,
  PcaResponse,
  UnivariateTargetResponse,
} from "@/lib/types";

type Tab = "features" | "charts";

const fmtNum = (n: number | null | undefined, digits = 3) => (n == null ? "—" : n.toFixed(digits));
const fmtPct = (n: number | null | undefined) => (n == null ? "—" : `${(n * 100).toFixed(1)}%`);

export default function AnalysisPage() {
  const { user, roleResolved } = useRole();
  const [tab, setTab] = useState<Tab>("features");

  if (!roleResolved || !user) {
    // Either auth hasn't resolved yet, or (transiently, before RoleGate
    // redirects a non-admin away) there's no authenticated user at all.
    return (
      <section>
        <PageHead title="Analysis" crumb="Loading" />
        <Skeleton style={{ height: 240 }} />
      </section>
    );
  }

  return (
    <section>
      <PageHead title="Analysis" crumb="Admin-only diagnostics" />

      <div className="flex gap-2 mb-4">
        <TabButton active={tab === "features"} onClick={() => setTab("features")}>
          Feature browser
        </TabButton>
        <TabButton active={tab === "charts"} onClick={() => setTab("charts")}>
          Rendered charts
        </TabButton>
      </div>

      {tab === "features" ? <FeatureBrowserPanel /> : <ChartGalleryPanel />}
    </section>
  );
}

function TabButton({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={[
        "px-3 py-[7px] rounded-lg text-[12.5px] border",
        active ? "border-teal text-text bg-teal/[0.08]" : "border-border text-text2 hover:bg-card-hi",
      ].join(" ")}
    >
      {children}
    </button>
  );
}

function ErrorState({ error }: { error: unknown }) {
  const isForbidden = error instanceof ApiError && error.status === 403;
  return (
    <Card>
      <p className="text-[12.5px] text-muted m-0">
        {isForbidden
          ? "You don't have access to this view. Sign in with an admin account and try again."
          : "Couldn't load this data. Try again shortly."}
      </p>
    </Card>
  );
}

// --- Panel A: feature-family browser -------------------------------------

function FeatureBrowserPanel() {
  const { user } = useRole();

  const [allFeatures, setAllFeatures] = useState<FeatureInventoryResponse[]>([]);
  const [familiesLoading, setFamiliesLoading] = useState(true);
  const [familiesError, setFamiliesError] = useState<unknown>(null);

  const [family, setFamily] = useState<string>("");

  const [features, setFeatures] = useState<FeatureInventoryResponse[]>([]);
  const [correlation, setCorrelation] = useState<FeatureCorrelationResponse[]>([]);
  const [univariate, setUnivariate] = useState<UnivariateTargetResponse[]>([]);
  const [familyDataLoading, setFamilyDataLoading] = useState(false);
  const [familyDataError, setFamilyDataError] = useState<unknown>(null);

  const [bivariate, setBivariate] = useState<BivariateResponse | null>(null);
  const [bivariateLoading, setBivariateLoading] = useState(true);
  const [bivariateError, setBivariateError] = useState<unknown>(null);

  const [pca, setPca] = useState<PcaResponse | null>(null);
  const [pcaLoading, setPcaLoading] = useState(true);
  const [pcaError, setPcaError] = useState<unknown>(null);

  // Load the unfiltered feature inventory once, to derive the family list.
  useEffect(() => {
    if (!user) return;
    let cancelled = false;
    setFamiliesLoading(true);
    setFamiliesError(null);
    user
      .getIdToken()
      .then((idToken) => getAnalysisFeatures(idToken))
      .then((data) => {
        if (cancelled) return;
        setAllFeatures(data);
        const families = deriveFamilies(data);
        if (families.length > 0) setFamily((prev) => prev || families[0]);
      })
      .catch((err) => {
        if (!cancelled) setFamiliesError(err);
      })
      .finally(() => {
        if (!cancelled) setFamiliesLoading(false);
      });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [user]);

  const families = useMemo(() => deriveFamilies(allFeatures), [allFeatures]);

  // Load per-family data whenever the selected family changes.
  useEffect(() => {
    if (!user || !family) return;
    let cancelled = false;
    setFamilyDataLoading(true);
    setFamilyDataError(null);
    user
      .getIdToken()
      .then((idToken) =>
        Promise.all([
          getAnalysisFeatures(idToken, family),
          getAnalysisCorrelation(idToken, family),
          getAnalysisUnivariate(idToken, family),
        ])
      )
      .then(([featuresRes, correlationRes, univariateRes]) => {
        if (cancelled) return;
        setFeatures(featuresRes);
        setCorrelation(correlationRes);
        setUnivariate(univariateRes);
      })
      .catch((err) => {
        if (!cancelled) setFamilyDataError(err);
      })
      .finally(() => {
        if (!cancelled) setFamilyDataLoading(false);
      });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [user, family]);

  // Load the (non-family-scoped) bivariate + PCA summaries once.
  useEffect(() => {
    if (!user) return;
    let cancelled = false;
    setBivariateLoading(true);
    setBivariateError(null);
    user
      .getIdToken()
      .then((idToken) => getAnalysisBivariate(idToken))
      .then((data) => {
        if (!cancelled) setBivariate(data);
      })
      .catch((err) => {
        if (!cancelled) setBivariateError(err);
      })
      .finally(() => {
        if (!cancelled) setBivariateLoading(false);
      });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [user]);

  useEffect(() => {
    if (!user) return;
    let cancelled = false;
    setPcaLoading(true);
    setPcaError(null);
    user
      .getIdToken()
      .then((idToken) => getAnalysisPca(idToken))
      .then((data) => {
        if (!cancelled) setPca(data);
      })
      .catch((err) => {
        if (!cancelled) setPcaError(err);
      })
      .finally(() => {
        if (!cancelled) setPcaLoading(false);
      });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [user]);

  return (
    <div className="flex flex-col gap-4">
      <Card title="Feature family">
        {familiesLoading && <Skeleton style={{ height: 32 }} />}
        {!familiesLoading && !!familiesError && <ErrorState error={familiesError} />}
        {!familiesLoading && !familiesError && families.length === 0 && (
          <p className="text-[12.5px] text-muted m-0">No feature families found.</p>
        )}
        {!familiesLoading && !familiesError && families.length > 0 && (
          <select
            className="w-full max-w-xs bg-card border border-border text-text rounded-lg px-2.5 py-[7px] text-[12.5px]"
            value={family}
            onChange={(e) => setFamily(e.target.value)}
          >
            {families.map((f) => (
              <option key={f} value={f}>
                {f}
              </option>
            ))}
          </select>
        )}
      </Card>

      {familyDataLoading && <Skeleton style={{ height: 160 }} />}
      {!familyDataLoading && !!familyDataError && <ErrorState error={familyDataError} />}

      {!familyDataLoading && !familyDataError && family && (
        <>
          <Card title={`Feature inventory — ${family}`}>
            {features.length === 0 ? (
              <p className="text-[12.5px] text-muted m-0">No features found for this family.</p>
            ) : (
              <TableWrap>
                <thead>
                  <tr>
                    <Th>Column</Th>
                    <Th>Data type</Th>
                    <Th>Role</Th>
                    <Th>Numeric</Th>
                    <Th>Categorical</Th>
                  </tr>
                </thead>
                <tbody>
                  {features.map((f) => (
                    <tr key={f.column_name} className="border-b border-border last:border-b-0">
                      <Td className="font-data">{f.column_name}</Td>
                      <Td className="font-data">{f.data_type ?? "—"}</Td>
                      <Td>{f.column_role ?? "—"}</Td>
                      <Td>{f.is_numeric == null ? "—" : f.is_numeric ? "yes" : "no"}</Td>
                      <Td>{f.is_categorical == null ? "—" : f.is_categorical ? "yes" : "no"}</Td>
                    </tr>
                  ))}
                </tbody>
              </TableWrap>
            )}
          </Card>

          <Card title={`Inter-feature correlation (redundancy) — ${family}`}>
            <p className="text-[11px] text-muted mt-0 mb-2.5">
              Pairwise correlation between features, used to flag redundant pairs. This is not
              correlation to the target — see the univariate table below for that.
            </p>
            {correlation.length === 0 ? (
              <p className="text-[12.5px] text-muted m-0">No correlation pairs found for this family.</p>
            ) : (
              <TableWrap>
                <thead>
                  <tr>
                    <Th>Track</Th>
                    <Th>Feature A</Th>
                    <Th>Feature B</Th>
                    <Th align="right">r (train)</Th>
                    <Th>Redundant</Th>
                    <Th>Resolution</Th>
                  </tr>
                </thead>
                <tbody>
                  {correlation.map((c, i) => (
                    <tr key={`${c.feature_a}-${c.feature_b}-${i}`} className="border-b border-border last:border-b-0">
                      <Td className="font-data">{c.track}</Td>
                      <Td className="font-data">{c.feature_a}</Td>
                      <Td className="font-data">{c.feature_b}</Td>
                      <Td align="right" className="font-data">
                        {fmtNum(c.r_train)}
                      </Td>
                      <Td>{c.is_redundant ? "yes" : "no"}</Td>
                      <Td title={c.resolution_reason ?? undefined}>{c.resolution}</Td>
                    </tr>
                  ))}
                </tbody>
              </TableWrap>
            )}
          </Card>

          <Card title={`Univariate target lift — ${family}`}>
            <p className="text-[11px] text-muted mt-0 mb-2.5">
              Each feature&apos;s relationship to the goal outcome (goal rate, mean when a goal was
              scored vs not, and point-biserial correlation to the target).
            </p>
            {univariate.length === 0 ? (
              <p className="text-[12.5px] text-muted m-0">No univariate stats found for this family.</p>
            ) : (
              <TableWrap>
                <thead>
                  <tr>
                    <Th>Column</Th>
                    <Th align="right">Goal rate</Th>
                    <Th align="right">Mean (goals)</Th>
                    <Th align="right">Mean (non-goals)</Th>
                    <Th align="right">Point-biserial r</Th>
                  </tr>
                </thead>
                <tbody>
                  {univariate.map((u) => (
                    <tr key={u.column_name} className="border-b border-border last:border-b-0">
                      <Td className="font-data">{u.column_name}</Td>
                      <Td align="right" className="font-data">
                        {fmtPct(u.goal_rate)}
                      </Td>
                      <Td align="right" className="font-data">
                        {fmtNum(u.mean_for_goals)}
                      </Td>
                      <Td align="right" className="font-data">
                        {fmtNum(u.mean_for_non_goals)}
                      </Td>
                      <Td align="right" className="font-data">
                        {fmtNum(u.point_biserial_corr)}
                      </Td>
                    </tr>
                  ))}
                </tbody>
              </TableWrap>
            )}
          </Card>
        </>
      )}

      <Card title="Bivariate interactions (all tracks)">
        {bivariateLoading && <Skeleton style={{ height: 100 }} />}
        {!bivariateLoading && !!bivariateError && <ErrorState error={bivariateError} />}
        {!bivariateLoading && !bivariateError && bivariate && (
          <>
            {bivariate.interactions.length === 0 ? (
              <p className="text-[12.5px] text-muted m-0">No bivariate interactions found.</p>
            ) : (
              <TableWrap>
                <thead>
                  <tr>
                    <Th>Track</Th>
                    <Th align="right">Tier</Th>
                    <Th>Feature A</Th>
                    <Th>Feature B</Th>
                    <Th align="right">Interaction coef</Th>
                    <Th align="right">p (FDR)</Th>
                    <Th>Status</Th>
                  </tr>
                </thead>
                <tbody>
                  {bivariate.interactions.map((it, i) => (
                    <tr key={`${it.feature_a}-${it.feature_b}-${i}`} className="border-b border-border last:border-b-0">
                      <Td className="font-data">{it.track}</Td>
                      <Td align="right" className="font-data">
                        {it.tier}
                      </Td>
                      <Td className="font-data">{it.feature_a}</Td>
                      <Td className="font-data">{it.feature_b}</Td>
                      <Td align="right" className="font-data">
                        {fmtNum(it.interaction_coef)}
                      </Td>
                      <Td align="right" className="font-data">
                        {fmtNum(it.interaction_p_fdr, 4)}
                      </Td>
                      <Td>{it.fit_status}</Td>
                    </tr>
                  ))}
                </tbody>
              </TableWrap>
            )}
          </>
        )}
      </Card>

      <Card title="PCA — explained variance (all tracks)">
        {pcaLoading && <Skeleton style={{ height: 100 }} />}
        {!pcaLoading && !!pcaError && <ErrorState error={pcaError} />}
        {!pcaLoading && !pcaError && pca && (
          <>
            {pca.components.length === 0 ? (
              <p className="text-[12.5px] text-muted m-0">No PCA components found.</p>
            ) : (
              <TableWrap>
                <thead>
                  <tr>
                    <Th>Track</Th>
                    <Th align="right">Component</Th>
                    <Th align="right">Explained variance</Th>
                    <Th align="right">Cumulative variance</Th>
                  </tr>
                </thead>
                <tbody>
                  {pca.components.map((c) => (
                    <tr key={`${c.track}-${c.component_number}`} className="border-b border-border last:border-b-0">
                      <Td className="font-data">{c.track}</Td>
                      <Td align="right" className="font-data">
                        {c.component_number}
                      </Td>
                      <Td align="right" className="font-data">
                        {fmtPct(c.explained_variance_ratio)}
                      </Td>
                      <Td align="right" className="font-data">
                        {fmtPct(c.cumulative_variance_ratio)}
                      </Td>
                    </tr>
                  ))}
                </tbody>
              </TableWrap>
            )}
          </>
        )}
      </Card>
    </div>
  );
}

// --- Panel B: rendered-chart gallery --------------------------------------

function ChartGalleryPanel() {
  const { user } = useRole();

  const [runIdInput, setRunIdInput] = useState("");
  const [requestedRunId, setRequestedRunId] = useState<string | undefined>(undefined);

  const [charts, setCharts] = useState<ChartsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<unknown>(null);

  useEffect(() => {
    if (!user) return;
    let cancelled = false;
    setLoading(true);
    setError(null);
    user
      .getIdToken()
      .then((idToken) => getAnalysisCharts(idToken, requestedRunId))
      .then((data) => {
        if (!cancelled) setCharts(data);
      })
      .catch((err) => {
        if (!cancelled) setError(err);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [user, requestedRunId]);

  return (
    <div className="flex flex-col gap-4">
      <Card title="Run selection">
        <p className="text-[11px] text-muted mt-0 mb-2.5">
          Defaults to the latest run. There&apos;s no endpoint to list all historical run ids — type
          one below to load a specific run instead.
        </p>
        <form
          className="flex gap-2"
          onSubmit={(e) => {
            e.preventDefault();
            setRequestedRunId(runIdInput.trim() || undefined);
          }}
        >
          <input
            type="text"
            placeholder="e.g. cxg-analysis-20260822T014705Z"
            value={runIdInput}
            onChange={(e) => setRunIdInput(e.target.value)}
            className="flex-1 bg-card border border-border text-text rounded-lg px-2.5 py-[7px] text-[12.5px] font-data"
          />
          <button
            type="submit"
            className="px-3 py-[7px] rounded-lg text-[12.5px] border border-teal text-text bg-teal/[0.08]"
          >
            Load run
          </button>
          {requestedRunId && (
            <button
              type="button"
              onClick={() => {
                setRunIdInput("");
                setRequestedRunId(undefined);
              }}
              className="px-3 py-[7px] rounded-lg text-[12.5px] border border-border text-text2 hover:bg-card-hi"
            >
              Reset to latest
            </button>
          )}
        </form>
      </Card>

      {loading && <Skeleton style={{ height: 160 }} />}
      {!loading && !!error && <ErrorState error={error} />}

      {!loading && !error && charts && (
        <Card>
          <div className="text-[12.5px] text-text2 mb-3">
            Showing run: <span className="font-data text-text">{charts.run_id}</span>
          </div>
          {charts.charts.length === 0 ? (
            <p className="text-[12.5px] text-muted m-0">No rendered charts found for this run.</p>
          ) : (
            <div className="grid grid-cols-1 gap-3">
              {charts.charts.map((chart) => (
                <div key={chart.chart_name} className="border border-border rounded p-3 bg-card-hi">
                  <div className="text-[12.5px] text-text mb-2">{chart.chart_name}</div>

                  {chart.signed_html_url ? (
                    <iframe
                      src={chart.signed_html_url}
                      title={chart.chart_name}
                      style={{ width: "100%", height: 500, border: "none", background: "#fff" }}
                      className="rounded mb-2"
                    />
                  ) : (
                    <div className="text-[11px] text-muted mb-1">
                      HTML:{" "}
                      <span className="font-data break-all text-text2">{chart.html_uri ?? "—"}</span>
                    </div>
                  )}

                  {chart.signed_png_url ? (
                    <a
                      href={chart.signed_png_url}
                      target="_blank"
                      rel="noreferrer"
                      className="text-[11px] text-teal hover:underline"
                    >
                      Open PNG
                    </a>
                  ) : (
                    <div className="text-[11px] text-muted mb-1">
                      PNG:{" "}
                      <span className="font-data break-all text-text2">{chart.png_uri ?? "—"}</span>
                    </div>
                  )}

                  {chart.rendered_at && (
                    <div className="text-[10.5px] text-muted mt-2">Rendered at {chart.rendered_at}</div>
                  )}
                  {/* Signed URLs (15 min expiry) come from the API when GCS URL
                      signing is available in the current environment — see
                      src/opponent_adjusted/api/gcs_signing.py for the exact IAM
                      constraint. When signing isn't available, signed_html_url/
                      signed_png_url are null and this falls back to the raw
                      gs:// text, same as before. */}
                </div>
              ))}
            </div>
          )}
        </Card>
      )}
    </div>
  );
}

// --- shared table primitives -----------------------------------------------

function TableWrap({ children }: { children: React.ReactNode }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full border-collapse text-[12.5px]">{children}</table>
    </div>
  );
}

function Th({ children, align = "left" }: { children: React.ReactNode; align?: "left" | "right" }) {
  return (
    <th
      className={`${align === "right" ? "text-right" : "text-left"} text-[11px] text-muted font-medium px-2 py-1.5 border-b border-border whitespace-nowrap`}
    >
      {children}
    </th>
  );
}

function Td({
  children,
  align = "left",
  className = "",
  title,
}: {
  children: React.ReactNode;
  align?: "left" | "right";
  className?: string;
  title?: string;
}) {
  return (
    <td
      className={`${align === "right" ? "text-right" : "text-left"} px-2 py-1.5 text-text2 whitespace-nowrap ${className}`}
      title={title}
    >
      {children}
    </td>
  );
}
