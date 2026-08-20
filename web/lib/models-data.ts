export type ModelStatus = "promoted" | "training" | "planned";
export type ModelTier = "Core" | "Spatial" | "Advanced";

export type ModelInfo = {
  name: string;
  status: ModelStatus;
  statusLabel: string;
  tier: ModelTier;
  caption: string | null;
};

// Data-driven per docs/dashboard_design_spec.md section 4: the Models tab
// should render generically off this list, not one hardcoded JSX block per family.
export const MODELS: ModelInfo[] = [
  {
    name: "CxG",
    status: "promoted",
    statusLabel: "Promoted",
    tier: "Core",
    caption: "Brier 0.071 · 13 feature families",
  },
  {
    name: "CxG+",
    status: "training",
    statusLabel: "In training",
    tier: "Spatial",
    caption: null,
  },
  {
    name: "CxA",
    status: "planned",
    statusLabel: "Planned",
    tier: "Core",
    caption: null,
  },
  {
    name: "CxT",
    status: "planned",
    statusLabel: "Planned",
    tier: "Core",
    caption: null,
  },
];
