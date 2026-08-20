export type ModelStatus = "promoted" | "training" | "planned";
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
};

// Data-driven per docs/dashboard_design_spec.md section 4: the Models tab
// should render generically off this list, not one hardcoded JSX block per family.
export const MODELS: ModelInfo[] = [
  {
    name: "CxG",
    status: "promoted",
    statusLabel: "Promoted",
    tier: "Core",
    validationMetrics: [{ label: "Brier score", value: "0.071" }],
    featureFamilyCount: "13 event-context feature families (E1–E13)",
  },
  {
    name: "CxG+",
    status: "training",
    statusLabel: "In training",
    tier: "Spatial",
    validationMetrics: [],
    featureFamilyCount: "28 candidates in development (13 event-context + 15 physical, E1–E13 + F1–F15)",
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
