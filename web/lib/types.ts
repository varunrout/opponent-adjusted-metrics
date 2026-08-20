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
