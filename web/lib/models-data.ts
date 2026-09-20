export type ModelStatus = "promoted" | "training" | "evaluated" | "planned";
export type ModelTier = "Core" | "Spatial" | "Advanced";

export type ValidationMetric = { label: string; value: string };

export type ModelInfo = {
  name: string;
  status: ModelStatus;
  statusLabel: string;
  tier: ModelTier;
  // Public validation metrics — this is the public registry layer per
  // docs/dashboard_content_ideation.md ("Models" section): status badge,
  // tier chip, validation metrics, feature-family counts. Raw validation
  // logs, arbitrary-version pinning, and promote/retire controls are
  // deliberately NOT part of this — those are admin internals, not built
  // yet, and would need their own role check when they are.
  validationMetrics: ValidationMetric[];
  featureFamilyCount: string | null;
  // One-line honest framing sentence, e.g. how a model compares against
  // a baseline. Optional — most entries don't need one. Per
  // docs/dashboard_design_spec_v2.md Hard gate 1: status language must
  // not overclaim, so this is where an honest "trails the baseline" note
  // lives rather than being hidden behind a bare metric number.
  comparisonNote?: string | null;
  // Where "See Stories for the full comparison →" (rendered alongside
  // comparisonNote) should link. Was hardcoded to the CxG story before this
  // field existed — extracted here because a bare hardcoded link would
  // silently point CxA's comparisonNote at CxG's own story too, which says
  // nothing about CxA. Only rendered when both this and comparisonNote are
  // set; CxA/CxA+ below intentionally leave it unset (no such story exists
  // yet) so their comparisonNote renders as plain text, no dangling link.
  comparisonStoryHref?: string | null;
  // Path segment appended after "/models/" for the "View full results..."
  // link. CxG/CxG+ point at /models/[modelKey] (event_v3 etc., backed by
  // /v1/models/cxg-models*). CxA/CxA+ point at /models/cxa/[track]
  // (event/plus, backed by /v1/models/cxa-models* — a separate route and
  // page component per docs/analysis/cxa_detail_page_v1.md's own
  // architecture decision, NOT a generalization of /models/[modelKey]).
  // Null for families with no real model yet (CxT).
  detailModelKey?: string | null;
  // Per docs/dashboard_design_spec_v2.md's existing "Experimental" disclosure
  // pattern (Badge status="experimental", already used on Match/Player/Team
  // pages for CxG+ 360-coverage captions) — when set, ModelCard renders the
  // same badge + caption treatment for this card specifically. Optional;
  // most entries don't need it.
  experimentalNote?: string | null;
};

// Data-driven per docs/dashboard_design_spec.md section 4: the Models tab
// should render generically off this list, not one hardcoded JSX block per family.
//
// CxG/CxG+ below reflect real v3 test-set results, per
// docs/dashboard_design_spec_v2.md §11 (verified against live oam_ml
// BigQuery data while building the Analysis tab's Model Results panel,
// 77adbe5/35fc92b). Status is "evaluated," not "promoted" — no serving
// layer exists yet (Hard gate 2 is still blocked) so "promoted" would
// overclaim production-readiness; not "training" either, since v3
// training is done. CxG trails the StatsBomb xG baseline on every
// captured metric on both tracks — that's disclosed here, not hidden,
// per Hard gate 1's reframing and the data-scientist persona's own
// credibility framing in §1.
export const MODELS: ModelInfo[] = [
  {
    name: "CxG",
    status: "evaluated",
    statusLabel: "Evaluated",
    tier: "Core",
    validationMetrics: [
      { label: "Test log_loss", value: "0.3003" },
      { label: "Test Brier", value: "0.0852" },
      { label: "Test AUC", value: "0.7148" },
    ],
    featureFamilyCount: "8 features",
    comparisonNote: "Trails the StatsBomb xG baseline (log_loss 0.2597).",
    comparisonStoryHref: "/stories/cxg-v3-honest-comparison",
    detailModelKey: "event_v3",
  },
  {
    name: "CxG+",
    status: "evaluated",
    statusLabel: "Evaluated",
    tier: "Spatial",
    validationMetrics: [
      { label: "Test log_loss", value: "0.2555" },
      { label: "Test Brier", value: "0.0713" },
      { label: "Test AUC", value: "0.8313" },
    ],
    featureFamilyCount: "24 features",
    comparisonNote: "Trails the StatsBomb xG baseline (log_loss 0.2430).",
    comparisonStoryHref: "/stories/cxg-v3-honest-comparison",
    detailModelKey: "plus_v3",
  },
  // CxA = P_create x P_convert (docs/analysis/cxa_combined_scorer_design_v1.md
  // section 1c). Two tracks, two cards (event-only, CxA+), mirroring the
  // CxG/CxG+ split above exactly — not one merged card. Real, frozen,
  // test-evaluated numbers per docs/analysis/cxa_p_create_test_eval_v1.md /
  // cxa_p_convert_test_eval_v1.md (verified against live oam_ml BigQuery data
  // while building /v1/models/cxa-models,
  // docs/analysis/cxa_dashboard_models_page_v1.md). Status "evaluated," not
  // "promoted" — same reasoning as CxG/CxG+ above: oam_serving now holds the
  // combined per-pass table, but that alone doesn't mean "production," and no
  // per-pass display exists yet either (deferred, see that doc's "what's
  // next"). Unlike CxG, there is no single "the CxA model" log_loss/AUC to
  // show — P_create and P_convert are separately frozen, separately
  // evaluated models, so validationMetrics shows both stages' own test
  // numbers, clearly labelled, never a fake unioned "combined" metric
  // (cxa_combined_score's own log_loss/AUC is a worse y_goal predictor than
  // P_convert alone — a real, checked finding, not something to hide by
  // presenting a flattering combined number instead).
  {
    name: "CxA (event-only)",
    status: "evaluated",
    statusLabel: "Evaluated",
    tier: "Core",
    validationMetrics: [
      { label: "P_create test log_loss", value: "0.0654" },
      { label: "P_create test AUC", value: "0.9153" },
      { label: "P_convert test log_loss", value: "0.2465" },
      { label: "P_convert test AUC", value: "0.7646" },
    ],
    featureFamilyCount: "10 P_create + 15 P_convert features",
    comparisonNote:
      "Combined CxA (create x convert) -- chance-creating passes only, ~2% of all passes; undefined, not zero, elsewhere.",
    detailModelKey: "cxa/event",
  },
  {
    name: "CxA+",
    status: "evaluated",
    statusLabel: "Evaluated",
    tier: "Spatial",
    validationMetrics: [
      { label: "P_create test log_loss", value: "0.0567" },
      { label: "P_create test AUC", value: "0.9565" },
      { label: "P_convert test log_loss", value: "0.2496" },
      { label: "P_convert test AUC", value: "0.7939" },
    ],
    featureFamilyCount: "11 P_create + 12 P_convert features",
    comparisonNote:
      "Combined CxA (create x convert) -- chance-creating passes only, ~2% of all passes; undefined, not zero, elsewhere.",
    detailModelKey: "cxa/plus",
    experimentalNote:
      "2,830 total chance-creating passes, 419 in test, zero Premier League rows.",
  },
  {
    name: "CxT",
    status: "planned",
    statusLabel: "Planned",
    tier: "Core",
    validationMetrics: [],
    featureFamilyCount: null,
  },
];
