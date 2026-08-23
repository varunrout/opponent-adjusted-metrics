// Response types for the v1 API. Mirrors the fixed backend contract exactly —
// do not add/rename fields without checking with the backend agent first.

export type CompetitionResponse = {
  competition_id: number;
  season_id: number;
  competition_name: string | null;
  competition_gender: string | null;
  country_name: string | null;
  season_name: string | null;
  match_updated: string | null;
  match_available: string | null;
  match_updated_360: string | null;
  match_available_360: string | null;
};

export type MatchResponse = {
  match_id: number;
  competition_id: number;
  season_id: number;
  match_date: string | null;
  kick_off: string | null;
  home_team_id: number | null;
  home_team_name: string | null;
  away_team_id: number | null;
  away_team_name: string | null;
  home_score: number | null;
  away_score: number | null;
  competition_stage: string | null;
  stadium: string | null;
  referee: string | null;
  match_status: string | null;
  match_status_360: string | null;
  last_updated: string | null;
  last_updated_360: string | null;
};

export type LineupPlayerResponse = {
  team_id: number | null;
  team_name: string | null;
  formation: number | null;
  player_id: number;
  player_name: string | null;
  position_name: string | null;
  jersey_number: number | null;
};

export type MatchDetailResponse = MatchResponse & {
  lineups: LineupPlayerResponse[];
};

export type PlayerSeasonResponse = {
  player_id: number;
  player_name: string | null;
  team_id: number | null;
  team_name: string | null;
  shots: number;
  goals: number;
  total_xg: number;
};

export type TeamSeasonResponse = {
  team_id: number;
  team_name: string | null;
  shots: number;
  goals: number;
  total_xg: number;
};

export type MeResponse = {
  role: "guest" | "viewer" | "admin";
  uid: string | null;
  email: string | null;
};

export type ShotResponse = {
  event_id: string;
  match_id: number;
  team_id: number | null;
  player_id: number | null;
  player_name: string | null;
  minute: number | null;
  period: number | null;
  location_x: number | null;
  location_y: number | null;
  end_x: number | null;
  end_y: number | null;
  statsbomb_xg: number | null;
  outcome_name: string | null;
  body_part_name: string | null;
  is_goal: boolean;
};

// --- /v1/analysis/* (admin-gated) ---------------------------------------

export type FeatureInventoryResponse = {
  feature_family: string;
  source_table: string | null;
  column_name: string;
  data_type: string | null;
  column_role: string | null;
  is_numeric: boolean | null;
  is_categorical: boolean | null;
};

// Inter-feature (redundancy) correlation between pairs of features — NOT
// correlation to the target. See UnivariateTargetResponse for target lift.
export type FeatureCorrelationResponse = {
  track: string;
  feature_a: string;
  feature_b: string;
  r_train: number | null;
  n_train: number;
  is_redundant: boolean;
  resolution: string;
  resolution_reason: string | null;
};

// The actual target-lift table (mean_for_goals vs mean_for_non_goals).
export type UnivariateTargetResponse = {
  feature_family: string;
  column_name: string;
  data_type: string | null;
  row_count: number | null;
  non_null_count: number | null;
  goal_rate: number | null;
  mean_when_available: number | null;
  mean_for_goals: number | null;
  mean_for_non_goals: number | null;
  point_biserial_corr: number | null;
};

export type BivariateCandidateResponse = {
  track: string;
  feature_family: string;
  column_name: string;
  data_type: string;
  qualification_reason: string;
};

export type BivariateInteractionResponse = {
  track: string;
  tier: number;
  feature_a: string;
  feature_b: string;
  n_train: number;
  interaction_coef: number | null;
  interaction_se: number | null;
  interaction_p_raw: number | null;
  interaction_p_fdr: number | null;
  lr_stat: number | null;
  main_effect_a_coef: number | null;
  main_effect_b_coef: number | null;
  validated_on_val_split: boolean | null;
  fit_status: string;
};

export type BivariateStratifiedResponse = {
  track: string;
  tier: number;
  feature_a: string;
  feature_b: string;
  stratum_a: string;
  stratum_b: string;
  n: number;
  goal_count: number;
  goal_rate: number | null;
};

export type BivariateResponse = {
  candidates: BivariateCandidateResponse[];
  interactions: BivariateInteractionResponse[];
  stratified: BivariateStratifiedResponse[];
};

export type PcaComponentResponse = {
  track: string;
  component_number: number;
  explained_variance_ratio: number;
  cumulative_variance_ratio: number;
};

export type PcaLoadingResponse = {
  track: string;
  component_number: number;
  feature_name: string;
  loading: number;
};

export type PcaResponse = {
  components: PcaComponentResponse[];
  loadings: PcaLoadingResponse[];
};

export type RenderedChartResponse = {
  run_id: string;
  chart_name: string;
  html_uri: string | null;
  png_uri: string | null;
  rendered_at: string | null;
  // Short-lived (15 min) signed HTTPS URLs, populated by the API only when
  // signing succeeds. Null when signing is unavailable/fails for this
  // chart — fall back to showing the raw html_uri/png_uri text in that case.
  signed_html_url: string | null;
  signed_png_url: string | null;
};

export type ChartsResponse = {
  run_id: string;
  charts: RenderedChartResponse[];
};

// --- /v1/analysis/cxg-models* (admin-gated) -------------------------------

export type CxgModelResultResponse = {
  model_key: string; // "baseline_v1" | "event_v3" | "plus_v2" | "plus_v3"
  track: string; // "cxg_event" | "cxg_plus"
  split: string; // "test" | "validation"
  model: string; // "v1" | "v2" | "v3" | "statsbomb_xg" | "dumb_baseline" (varies by model_key)
  n: number;
  log_loss: number | null;
  brier_score: number | null;
  roc_auc: number | null;
  is_frozen: boolean; // always true for all four model_keys
  is_current: boolean; // true only for event_v3/plus_v3, false for baseline_v1/plus_v2
};

export type CxgCoefficientResponse = {
  model_key: string;
  track: string;
  feature: string;
  coefficient: number | null;
  std_error: number | null;
  p_value: number | null;
};
