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
    comparisonNote: "Trails the StatsBomb xG baseline (log_loss 0.2597) — see Stories for the full, honest comparison.",
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
    comparisonNote: "Trails the StatsBomb xG baseline (log_loss 0.2430) — see Stories for the full, honest comparison.",
  },
  {
    name: "CxA",
    status: "planned",
    statusLabel: "Planned",
    tier: "Core",
    validationMetrics: [],
    featureFamilyCount: null,
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
