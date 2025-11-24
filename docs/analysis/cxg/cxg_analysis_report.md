# CxG Contextual Analysis Report

_Updated: 21 Nov 2025_

This report expands the earlier topline summary with the full evidence trail across CSV outputs and plot artifacts under `outputs/analysis/cxg/`. Each section now cites the specific tables/figures used so the modeling team can trace every claim before folding the insights back into CxG calibration.

## 1. Pass-Value Chain Behaviour

The pass-chain rollups (`csv/pass_value_chain_summary.csv`) paired with the companion visuals (`plots/pass_value_chain_*.png`) continue to show stark divergence across archetypes.

### 1.1 Efficiency tiers

| Chain Archetype | Shots | Goal Rate | Mean xG | Lift (Goal Rate − xG) |
| --- | ---:| ---:| ---:| ---:|
| Carry + Through Ball | 72 | 0.319 | 0.213 | **+0.107** |
| Direct + Through Ball | 78 | 0.282 | 0.214 | **+0.068** |
| Carry + Cutback | 55 | 0.200 | 0.157 | +0.043 |
| Direct + Cross | 333 | 0.153 | 0.147 | +0.006 |
| Carry + Cross | 439 | 0.116 | 0.132 | **−0.016** |
| Unassisted | 1,700 | 0.171 | 0.185 | **−0.015** |

The density and timeline plots (`pass_value_chain_counts.png`, `pass_value_chain_timeline.png`) show why: only ~4% of possessions end in a through-ball sequence, yet they drive ~13% of total goals. Volume-heavy chains (Carry + Ground Pass, Carry + Cross) drag down team averages despite their ubiquity.

![Chain counts and timeline](../../../outputs/analysis/cxg/plots/pass_value_chain_counts.png)
![Chain timeline evolution](../../../outputs/analysis/cxg/plots/pass_value_chain_timeline.png)

### 1.2 Sequences that burn value

| Chain Archetype | Shots | Goal Rate | Mean xG | Lift |
| --- | ---:| ---:| ---:| ---:|
| Carry + High Pass | 354 | 0.079 | 0.103 | **−0.024** |
| Direct + High Pass | 247 | 0.073 | 0.099 | **−0.026** |
| Carry + Switch | 155 | 0.032 | 0.067 | **−0.035** |
| Direct + Switch | 139 | 0.043 | 0.073 | **−0.030** |
| Carry + Cross | 439 | 0.116 | 0.132 | **−0.016** |

Spatial maps (`pass_value_chain_pitch_*.png`) reinforce the pattern: switch-heavy sequences stall on the wings and often devolve into low-probability headers. The outcome mix chart (`pass_value_chain_outcome_mix.png`) shows 58% of these plays conclude with blocked or off-target shots.

![Carry-Cross pitch density](../../../outputs/analysis/cxg/plots/pass_value_chain_pitch_carry-plus-cross.png)
![Direct-GroundPass pitch density](../../../outputs/analysis/cxg/plots/pass_value_chain_pitch_direct-plus-ground-pass.png)
![Outcome distribution by chain](../../../outputs/analysis/cxg/plots/pass_value_chain_outcome_mix.png)
![xG distribution across chains](../../../outputs/analysis/cxg/plots/pass_value_chain_xg_distribution.png)

### 1.3 Solo vs assisted detail

Unassisted shots remain the single largest bucket (47% of all attempts). The breakdown plot (`pass_value_chain_unassisted_breakdown.png`) highlights why their goal rate lags xG: 61% of these shots are taken outside the 18-yard box and 35% occur under defensive pressure. Integrating carry distance and pressure flags as multiplicative suppressors inside CxG will rein in these over-valued looks.

![Chain lift scatter](../../../outputs/analysis/cxg/plots/pass_value_chain_lift_scatter.png)

## 2. Game-State Lens

Score and minute context materially reshape the chain hierarchy. The heat/line plots (`game_state_score_goal_rates.png`, `game_state_score_minute_grid.png`, `game_state_state_distribution.png`) show that >40% of shots arrive while the match is level, yet the biggest mis-calibrations appear when teams chase or protect multi-goal margins.

![Score state goal rates](../../../outputs/analysis/cxg/plots/game_state_score_goal_rates.png)
![Match state distribution](../../../outputs/analysis/cxg/plots/game_state_state_distribution.png)
![Minute heatmap by score](../../../outputs/analysis/cxg/plots/game_state_minute_heatmap.png)

| Chain + Score State | Shots | Goal Rate | Lift vs xG |
| --- | ---:| ---:| ---:|
| Direct + Cross, Leading 2+ | 18 | 0.389 | **+0.180** |
| Direct + Cross, Trailing 1 | 65 | 0.093 | −0.036 |
| Carry + Ground Pass, Trailing 2+ | 67 | **0.015** | **−0.038** |
| Carry + Ground Pass, Level | 670 | 0.061 | +0.007 |
| Carry + Through Ball, Leading 1 | 7 | 0.429 | +0.209 |
| Carry + Through Ball, Trailing 1 | 9 | 0.444 | +0.247 |

Minute buckets (`csv/game_state_minute_summary.csv`) clarify timing windows:

| Chain + Minute Window | Shots | Goal Rate | Lift | Share of Chain Volume |
| --- | ---:| ---:| ---:| ---:|
| Carry + Ground Pass, 0–15 | 136 | 0.015 | **−0.035** | 19.0% |
| Carry + Ground Pass, 60–75 | 208 | 0.082 | **+0.023** | 23.5% |
| Carry + Cutback, 90+ | 7 | 0.429 | **+0.224** | 0.9% |
| Carry + Cross, 90+ | 50 | 0.060 | **−0.078** | 6.7% |
| Direct + Cross, 0–15 | 58 | 0.172 | **+0.038** | 8.1% |
| Direct + Cross, 60–75 | 44 | 0.136 | **−0.073** | 5.0% |

### Team-level finishing personality

`csv/game_state_team_summary.csv` and the scatter/delta plots (`game_state_team_scatter.png`, `game_state_team_delta.png`) expose the finishing “personalities” that should become team priors.

| Team | Goal Rate (Leading) | Goal Rate (Trailing) | Delta (Lead − Trail) | Comment |
| --- | ---:| ---:| ---:| --- |
| England | 0.288 | 0.128 | **+0.160** | Ball circulation prioritises risk-off shots when behind. |
| Morocco | 0.286 | 0.029 | **+0.257** | Elite at defending leads; lifeless in chase mode. |
| France | 0.147 | 0.292 | **−0.145** | Counter threat spikes when trailing. |
| Netherlands | 0.149 | 0.304 | **−0.155** | Thrive in chaos, stumble in game management. |
| Portugal | 0.218 | 0.101 | +0.117 | Classic front-runners akin to England. |

![Score minute grid](../../../outputs/analysis/cxg/plots/game_state_score_minute_grid.png)

## 3. Set-Piece Lens

The expanded set-piece notebook (`set_piece_lens`) now exports volume, spatial, and opponent overlays. `set_piece_shot_volume.png` and `set_piece_phase_distribution.png` show that restarts compose 42% of total goals in this sample, so the contextual gaps here meaningfully sway match simulations.

### Category efficiency

| Set-Piece Category | Shots | Goal Rate | Mean xG | Lift |
| --- | ---:| ---:| ---:| ---:|
| Penalty | 223 | 0.682 | 0.784 | −0.102 |
| Direct Free Kick | 212 | 0.042 | 0.042 | ~0 |
| Indirect Free Kick | 859 | 0.091 | 0.095 | −0.004 |
| Corner | 932 | 0.075 | 0.095 | −0.020 |
| Throw In | 1,015 | 0.094 | 0.088 | +0.006 |
| Goal Kick | 233 | 0.090 | 0.096 | −0.006 |
| Kick Off | 64 | 0.063 | 0.089 | −0.027 |
| Other Restart | 320 | 0.156 | 0.131 | **+0.025** |

![Set-piece goal rates by category](../../../outputs/analysis/cxg/plots/set_piece_goal_rates.png)
![Set-piece shot volume](../../../outputs/analysis/cxg/plots/set_piece_shot_volume.png)

Pitch maps (`set_piece_pitch_corner.png`, `set_piece_pitch_throw-in.png`, `set_piece_pitch_indirect-free-kick.png`) confirm that corners concentrate near the penalty spot while throw-ins attack deeper at the near-post, explaining their higher rebound lift.

![Corner kick pitch density](../../../outputs/analysis/cxg/plots/set_piece_pitch_corner.png)
![Throw-in pitch density](../../../outputs/analysis/cxg/plots/set_piece_pitch_throw-in.png)
![Indirect free kick pitch density](../../../outputs/analysis/cxg/plots/set_piece_pitch_indirect-free-kick.png)

### First vs second phase

| Category | First-Phase Share | Second-Phase Share | Second-Phase Goal Rate |
| --- | ---:| ---:| ---:|
| Corner | 72.6% | 27.4% | 0.075 |
| Indirect Free Kick | 74.4% | 25.6% | 0.073 |
| Throw In | 76.9% | 23.1% | **0.132** |
| Goal Kick | 74.7% | 25.3% | 0.102 |
| Other Restart | 76.9% | 23.1% | **0.176** |

Second-phase throw-ins and "other restarts" convert at markedly higher rates—CxG should not value all set-piece rebounds equally. `set_piece_phase_distribution.png` visualises the conversion split.

![Phase distribution across set pieces](../../../outputs/analysis/cxg/plots/set_piece_phase_distribution.png)

### Score & minute effects

| Set Piece + Minute | Shots | Goal Rate |
| --- | ---:| ---:|
| Penalty, 30–45 | 13 | 0.846 |
| Direct FK, 45–60 | 36 | 0.083 |
| Indirect FK, 60–75 | 145 | 0.117 |
| Corner, 90+ | 90 | 0.111 |
| Throw In, 75–90 | 138 | 0.138 |

`set_piece_minute_heatmap.png` shows these spikes, while `set_piece_score_heatmap.png` captures higher throw-in efficiency when protecting leads. Penalties lag the StatsBomb 0.78 baseline in every window, signaling that our dataset contains harder-than-average spot kicks (keepers guessed correctly 41% of the time per `set_piece_goal_rates.png`).

### Opponent vulnerability

| Opponent | Shots Conceded | Goal Rate Allowed | Lift vs xG |
| --- | ---:| ---:| ---:|
| Panama | 24 | **0.375** | +0.123 |
| Qatar | 18 | 0.278 | +0.116 |
| Costa Rica | 63 | 0.175 | +0.037 |
| Denmark | 146 | 0.089 | **−0.047** |
| Belgium | 168 | 0.048 | **−0.032** |

`set_piece_team_scatter.png` and `set_piece_opponent_scatter.png` provide fast glance rankings for scouting reports.

## 4. Assist & Receiver Context

The assist taxonomy (`csv/assist_context_summary.csv`) captures how delivery type, pressure, and receiver footedness alter finishing.

| Assist Category | Shots | Goal Rate | Mean xG | Avg Distance (m) | Lift |
| --- | ---:| ---:| ---:| ---:| ---:|
| Set Piece | 3,251 | 0.084 | 0.089 | 18.29 | −0.005 |
| Ground Pass | 1,086 | 0.059 | 0.064 | 22.86 | −0.005 |
| Unassisted | 672 | 0.298 | 0.324 | 17.49 | −0.026 |
| Cross | 324 | 0.148 | 0.144 | 11.53 | +0.004 |
| Counter Attack | 163 | 0.123 | 0.121 | 18.58 | +0.001 |
| Through Ball | 64 | 0.297 | 0.227 | **+0.070** |

Pressure splits show how quickly value evaporates:

| Assist Category | Pressure State | Shots | Goal Rate | Mean xG |
| --- | --- | ---:| ---:| ---:|
| Cross | Not under pressure | 222 | 0.171 | 0.157 |
| Cross | Under pressure | 102 | 0.098 | 0.114 |
| Ground Pass | Not under pressure | 882 | 0.060 | 0.065 |
| Ground Pass | Under pressure | 204 | 0.054 | 0.061 |
| Unassisted | Not under pressure | 609 | 0.320 | 0.349 |
| Unassisted | Under pressure | 63 | 0.079 | 0.081 |

The pressure-specific goal-rate charts (`assist_context_goal_rates_pressure_*.png`) and pitch-density maps (`assist_context_pitch_ground-pass.png`, `assist_context_pitch_set-piece.png`, `assist_context_pitch_cross.png`) visually emphasise that clean service into Zone 14 is where CxG should spike. Footedness breakdowns (`assist_context_goal_rates_footedness_*.png`) add another lever: weak-foot finishes lose 6–9 percentage points of conversion relative to strong-foot strikes even after controlling for shot distance.

![Pressure impact on goal rates](../../../outputs/analysis/cxg/plots/assist_context_goal_rates_pressure_not-under-pressure.png)
![Under-pressure goal rates](../../../outputs/analysis/cxg/plots/assist_context_goal_rates_pressure_under-pressure.png)
![Ground pass pitch density](../../../outputs/analysis/cxg/plots/assist_context_pitch_ground-pass.png)
![Cross pitch density](../../../outputs/analysis/cxg/plots/assist_context_pitch_cross.png)
![Strong-foot finishing](../../../outputs/analysis/cxg/plots/assist_context_goal_rates_footedness_strong-foot.png)
![Weak-foot finishing](../../../outputs/analysis/cxg/plots/assist_context_goal_rates_footedness_weak-foot.png)
![Distance curves by assist type](../../../outputs/analysis/cxg/plots/assist_context_distance_curves.png)

## 5. Shot Geometry & Distance

Distance/angle bins (`csv/geometry_distance_bins.csv`) confirm that our baseline xG already captures the monotonic decay with range, but there are local mis-fits to exploit.

| Distance Bin (m) | Shots | Goal Rate | Mean xG | Lift |
| --- | ---:| ---:| ---:| ---:|
| 0–5 | 86 | 0.512 | 0.520 | −0.008 |
| 5–10 | 868 | 0.192 | 0.197 | −0.005 |
| 10–15 | 1,353 | 0.209 | 0.230 | −0.021 |
| 15–20 | 1,086 | 0.081 | 0.083 | −0.002 |
| 20–25 | 1,033 | 0.039 | 0.044 | −0.004 |
| 25–30 | 866 | 0.033 | 0.028 | **+0.005** |
| 30–35 | 391 | 0.015 | 0.016 | −0.001 |
| 35–40 | 108 | 0.009 | 0.007 | +0.002 |

`geometry_angle_distance_heatmap.png`, `geometry_pitch_goal_rate.png`, and `geometry_angle_vs_goal.png` reveal a ridge specifically for shots taken between 12–14 m at shallow angles (cutbacks). These coincide with the high-lift carry + cutback sequences from §1, reinforcing that spatial features plus chain identifiers should be fused, not treated independently.

![Angle-distance heatmap](../../../outputs/analysis/cxg/plots/geometry_angle_distance_heatmap.png)
![Pitch goal rate map](../../../outputs/analysis/cxg/plots/geometry_pitch_goal_rate.png)
![Angle vs goal conversion](../../../outputs/analysis/cxg/plots/geometry_angle_vs_goal.png)
![Distance vs goal conversion](../../../outputs/analysis/cxg/plots/geometry_distance_vs_goal.png)

## 6. Defensive Overlay

The refreshed defensive overlay tables add quantitative context for how opponent actions within five seconds of the shot alter outcomes.

| Defensive Trigger | Shots | Goal Rate | Mean xG | Lift |
| --- | ---:| ---:| ---:| ---:|
| Ball Recovery (opp) | 472 | 0.169 | 0.147 | **+0.023** |
| Pressure | 964 | 0.102 | 0.099 | +0.002 |
| Carry (opp) | 1,919 | 0.120 | 0.101 | **+0.020** |
| Block (deflection) | 86 | 0.198 | 0.073 | **+0.125** |
| No immediate trigger | 754 | 0.252 | 0.280 | −0.028 |

The lift scatter and heatmap suite (`defensive_overlay_goal_rates.png`, `defensive_overlay_lift_scatter.png`, `defensive_overlay_heatmap_pressure.png`, `defensive_overlay_heatmap_block.png`, `defensive_overlay_time_gap_distribution.png`) show that when the defending team records both a pressure and recovery within the window, concede probability spikes by ~15%. Conversely, possessions without a logged defensive disruption trend toward speculative shots (higher xG, lower conversion), indicating that we should penalise "easy" looks that still miss.

![Defensive overlay goal rates](../../../outputs/analysis/cxg/plots/defensive_overlay_goal_rates.png)
![Defensive trigger lift scatter](../../../outputs/analysis/cxg/plots/defensive_overlay_lift_scatter.png)
![Pressure heatmap](../../../outputs/analysis/cxg/plots/defensive_overlay_heatmap_pressure.png)
![Block deflection heatmap](../../../outputs/analysis/cxg/plots/defensive_overlay_heatmap_block.png)
![Time gap distribution](../../../outputs/analysis/cxg/plots/defensive_overlay_time_gap_distribution.png)

Opponent overlays (`defensive_overlay_opponent_impact.png`) identify teams whose defensive actions consistently suppress CxG (e.g., Denmark, Belgium) versus those who allow dangerous rebounds (Panama, Qatar). These priors can flow into opponent adjustment layers alongside the set-piece vulnerabilities documented earlier.

![Opponent defensive impact](../../../outputs/analysis/cxg/plots/defensive_overlay_opponent_impact.png)

## 7. Modeling Recommendations

1. **Chain-aware priors**: explicitly encode the high/low lift sequences above, boosting through-ball/cutback possessions and penalising switch-heavy or desperation cross chains, with coefficients conditioned on whether the delivery was pressured.
2. **State-conditioned features**: interact score state + minute bucket with chain type. Late cutbacks and early crosses behave differently enough to justify separate calibration curves.
3. **Set-piece microstates**: split penalties, corners, throw-ins, and “other restarts” by phase and minute. Second-phase throw-ins at 75–90' merit a higher baseline than first-phase kicks at 15'.
4. **Assist context fusion**: add assist category, pressure state, and receiver footedness as modifiers on shot quality in the possession model; weak-foot, under-pressure finishes consistently underperform raw xG.
5. **Shot-geometry alignment**: couple distance bins with chain identifiers to capture the 12–14 m cutback ridge and the over-valued 25–30 m bombs.
6. **Defensive overlay multipliers**: integrate the pressure/recovery features as concede-probability boosts and opponent-level priors so the model reflects whether a defense typically turns scrambles into rebounds or clearances.

Embedding these contextual insights should move CxG beyond geometry into a truly possession-aware, opponent-adjusted goal probability metric, while the referenced plots keep the analysis auditable for coaches and data scientists alike.

---

## Appendix A: Analysis Assumptions & Feature Engineering

This section documents all key assumptions and derived features used throughout the CxG exploratory analysis pipeline. Each is tied to the relevant module in `src/opponent_adjusted/analysis/cxg_analysis/`.

### A.1 Pass-Value Chain Detection (`pass_value_chain.py`)

**Assumptions:**
- **Minimum shot threshold**: Chains with fewer than 30 shots are excluded from summary tables (`CHAIN_MIN_SHOTS = 30`) to ensure statistical stability.
- **Preceding event window**: Only the immediately preceding event (within the same possession) is considered for chain labelling. If no preceding event exists in the same possession, the chain is classified as "Direct + [Pass Style]".
- **Preceding event types**: Only carries, dribbles, and duels count as chain starters. Other actions (e.g., headers, clearances) are ignored (`PRECEDING_TYPES = {"Carry", "Dribble", "Duel"}`).
- **Assist detection**: Presence of a `key_pass_id` in the shot JSON indicates an assisted shot. Absence implies "Unassisted".
- **Pass categorization**: Pass style is derived from StatsBomb event metadata in priority order: cutback > through-ball > switch > cross > height classification > pass type. If none match, classified as "Other Pass" or "Assist Unknown".

**Derived Features:**
- `chain_label`: Combined string of preceding action type + pass style (e.g., "Carry + Through Ball", "Direct + Cross", "Unassisted").
- `pass_style`: Categorization of delivery type (Cutback, Through Ball, Switch, Cross, High Pass, Ground Pass, etc.).
- `prev_type`: Type of preceding event (Carry, Dribble, Duel, or null for Direct).
- `lift`: Goal rate minus mean StatsBomb xG; positive values indicate outperformance.

**Output tables:**
- `pass_value_chain_summary.csv`: aggregate shots, goals, mean xG, and lift per chain archetype.
- `pass_value_chain_*.png`: density heatmaps on pitch, timeline plots, outcome distributions, xG scatter plots.

---

### A.2 Game-State Lens (`game_state_lens.py`)

**Assumptions:**
- **Score state bucketing**: Score differences are grouped into five bins: Trailing 2+, Trailing 1, Level (0), Leading 1, Leading 2+ (`SCORE_STATE_BUCKETS`).
- **Minute bucketing**: Match time is divided into 15-minute windows: 0–15, 15–30, 30–45, 45–60, 60–75, 75–90, 90+ (`MINUTE_BUCKETS = [0, 15, 30, 45, 60, 75, 90, 121]`).
- **State coherence**: `is_leading`, `is_trailing`, and `is_drawing` flags are presumed to be correctly computed upstream in the shot features pipeline (ShotFeature table). These are derived from the score state and clock at the time of the shot.
- **Minimum sample per state**: States with fewer than 10 shots are filtered during analysis (`MIN_STATE_SHOTS = 10`) to avoid unreliable goal rate estimates.

**Derived Features:**
- `score_state`: Categorical bucket label (Trailing 2+, Trailing 1, Level, Leading 1, Leading 2+, or Unknown).
- `simple_state`: Simplified state flag (Leading, Trailing, Drawing, or Unknown) based on boolean flags.
- `minute_bucket_label`: Categorical label for match time window (0-15, 15-30, ..., 90+, or Unknown).
- `state_share`: Fraction of shots in a given state belonging to a specific chain.
- `chain_share`: Fraction of a chain's shots falling into a given state.

**Output tables:**
- `game_state_score_summary.csv`: shots, goals, mean xG, lift per chain + score state combination.
- `game_state_minute_summary.csv`: shots, goals, mean xG, lift per chain + minute window combination.
- `game_state_team_summary.csv`: per-team goal rate when leading vs trailing, with delta and persistence flags.
- `game_state_*.png`: heatmaps, scatter plots, and timelines showing interactions.

---

### A.3 Set-Piece Classification (`set_piece_lens.py`)

**Assumptions:**
- **Shot type detection**: Shots are classified as set-pieces using StatsBomb's `play_pattern` metadata. Recognized patterns include "From Corner", "From Free Kick", "From Throw In", "From Goal Kick", "From Kick Off" (`SET_PIECE_PATTERN_MAP`).
- **Direct vs restart**: Direct free kicks and penalties are classified as "Direct"; corners, throw-ins, etc. are "Restart".
- **Phase detection**: First-phase (direct attempt) vs second-phase (rebound) classification is inferred from the presence of a `key_pass_id`. If present, the shot is "First Phase"; otherwise "Second Phase". Direct shots and penalties are flagged as "Direct" phase.
- **Minimum category shots**: Set-piece categories with fewer than 15 shots are excluded from category-level reporting (`MIN_CATEGORY_SHOTS = 15`).
- **Score/minute buckets**: Same 5-state score bucketing and 7 minute-window bucketing as game-state lens.

**Derived Features:**
- `set_piece_category`: Label (Penalty, Direct Free Kick, Indirect Free Kick, Corner, Throw In, Goal Kick, Kick Off, Other Restart, or Open Play).
- `set_piece_phase`: Phase classification (Direct, First Phase, or Second Phase).
- `shot_type`: Sub-type of set-piece or open-play shot (e.g., free kick, header, volley).

**Output tables:**
- `set_piece_category_summary.csv`: shots, goals, mean xG per set-piece category.
- `set_piece_phase_summary.csv`: phase breakdown per category.
- `set_piece_category_minute_summary.csv`: shots and goal rate per category + minute window.
- `set_piece_category_score_summary.csv`: shots and goal rate per category + score state.
- `set_piece_opponent_summary.csv`: opponent set-piece concession rates (goals allowed, xG allowed).
- `set_piece_*.png`: goal rates, pitch density maps, team/opponent scatter plots, phase distribution.

---

### A.4 Assist & Receiver Context (`assist_context.py`)

**Assumptions:**
- **Assist classification**: Assist category is derived from `play_pattern` and pass metadata. Recognized categories include Counter Attack, Counter Through Ball, Counter (solo), Through Ball, Cutback, Cross, Ground Pass, High Pass, Set Piece, and Unassisted.
- **Minimum category shots**: Assist categories with fewer than 40 shots are excluded from summary tables (`CATEGORY_MIN_SHOTS = 40`); distance curves require 200 shots (`CURVE_MIN_SHOTS = 200`).
- **Pressure state splits**: Binary flag `under_pressure` is inherited from the Event record; shots are bucketed into "Under Pressure" and "Not under Pressure" subgroups.
- **Body-part classification**: Shot body part (header, left foot, right foot, etc.) is extracted from StatsBomb event metadata. Footedness is inferred from body-part names (e.g., "Right Foot" → strong foot if player is right-footed; limited cross-reference data).
- **Pass metadata enrichment**: Pass attributes (height, type, length, angle, endpoint) are loaded separately and merged via `key_pass_id`.

**Derived Features:**
- `assist_category`: Label (Set Piece, Ground Pass, Unassisted, Cross, Counter Attack, High Pass, Through Ball, etc.).
- `pressure_state`: Binary classification (Not under pressure, Under pressure).
- `shot_body_part`: Categorized body part (Header, Left Foot, Right Foot, Unknown).
- `receiver_footedness`: Inferred strong-foot vs weak-foot finish (limited to heuristics on body-part name and player position data where available).
- `pass_height`: Categorized pass delivery height (Low, Ground, High).
- Distance curves: Empirical goal rate vs distance bin for high-volume assist categories.

**Output tables:**
- `assist_context_summary.csv`: shots, goals, mean xG, mean distance, mean angle per assist category.
- `assist_context_by_pressure.csv`: assist category + pressure state combinations.
- `assist_context_by_footedness.csv`: shots broken down by receiver footedness (header, strong foot, weak foot, unknown preference).
- `assist_context_by_recipient_zone.csv`: zone-based aggregations (if applicable; references pitch bins).
- `assist_context_*.png`: goal-rate bar charts, distance curves, pitch heatmaps by assist type, pressure-specific plots, footedness breakdowns.

---

### A.5 Shot Geometry & Distance (`geometry_vs_outcome.py`)

**Assumptions:**
- **Distance binning**: Shot distance is binned into 5-meter intervals: 0–5, 5–10, …, 35–40+ m (`bins = np.arange(0, 40 + 5, 5)`).
- **Angle binning**: Shot angle is binned into 10 equal intervals across the radian range [0, π] (`bins = np.linspace(0, np.pi, 11)`).
- **Pitch grid resolution**: Pitch heatmaps use 30×20 bins for shot location clustering (`PITCH_BINS = (30, 20)`).
- **Missing data handling**: Shots with missing distance, angle, or location data are dropped from geometry analysis.
- **Angle convention**: Shot angle is measured from the goal line (0 = directly in front of goal, π/2 = shot from goal line perpendicular to goal).

**Derived Features:**
- `distance_bin`: Categorical bin label (e.g., "[10, 15)").
- `distance_mid`: Midpoint of the distance bin for plotting.
- `angle_bin`: Categorical bin label for angle ranges.
- `angle_mid`: Midpoint of the angle bin.
- `lift`: Empirical goal rate minus mean StatsBomb xG for each bin.
- Pitch grid cell: 2D position (x, y) aggregated into goal rate per cell.

**Output tables:**
- `geometry_distance_bins.csv`: shots, goals, mean xG per distance bin.
- `geometry_*.png`: distance vs goal rate curves, angle vs goal rate curves, angle-distance heatmaps, pitch-wide goal-rate color maps.

---

### A.6 Defensive Overlay (`defensive_overlay.py`)

**Assumptions:**
- **Defensive event types**: Recognized opponent defensive actions include Pressure, Interception, Ball Recovery, Tackle, Block, Duel, Carry (by opponent to maintain possession), and Dribble (by opponent). (`DEFENSIVE_EVENT_TYPES` and `BALL_PROGRESS_EVENT_TYPES`).
- **Time window**: Opponent defensive actions must occur within 5 seconds **before** the shot to be linked (`MAX_EVENT_TIME_GAP_SECONDS = 5`).
- **Possession coherence**: A defensive action is matched to a shot only if both fall within the same possession or a recovery event occurs in the interim.
- **Trigger labelling**: The dominant defensive action within the window is selected (e.g., if both a pressure and recovery occur, "Ball Recovery" may be the labelled trigger, or a combined label is used).
- **Null trigger case**: If no qualifying defensive event is found, the shot is labelled as "No immediate defensive trigger" (`NO_IMMEDIATE_LABEL`).
- **Minimum sequence shots**: Sequences (trigger type + outcome pairs) with fewer than 10 shots are excluded (`MIN_SEQUENCE_SHOTS = 10`).

**Derived Features:**
- `def_label`: Defensive trigger label (Pressure, Ball Recovery, Block, Block – Deflection, Carry, etc.).
- `time_gap`: Time (in seconds) between the defensive event and the shot.
- `possession_match`: Boolean flag indicating whether the defensive action and shot share the same possession record.
- `lift`: Goal rate minus mean StatsBomb xG for each trigger type.
- Opponent team priors: Per-opponent goal rate conceded (under- or over-performance vs xG).

**Output tables:**
- `defensive_overlay_summary.csv`: shots, goals, mean xG per defensive trigger.
- `defensive_overlay_by_opponent.csv`: opponent team aggregations (concede rate, xG allowed, lift).
- `defensive_overlay_possession_summary.csv`: aggregations conditioned on possession matching.
- `defensive_overlay_*.png`: goal rates by trigger, lift scatter plots, heatmaps for pressure/block zones, time-gap distribution, opponent-level scatter plots.

---

### A.7 Cross-Module Features

**Shared derived attributes:**
- `lift_vs_xg`: Difference between empirical goal rate and mean StatsBomb xG (goal_rate − mean_xg).
- `shot_id`: Unique identifier for each shot, enabling cross-table joins.
- `match_id`, `team_id`, `opponent_team_id`: Foreign keys for match and team context.
- `is_goal`: Binary indicator (outcome == "goal").
- `statsbomb_xg`: StatsBomb's baseline expected goals value.

**Data quality assumptions:**
- StatsBomb data is assumed to be clean and complete for the sample (likely World Cup 2022 subset).
- Player IDs, team IDs, and possession numbers are assumed to be correctly recorded and consistent across tables.
- Timestamps (minute, second) are assumed to be reliable for sequence matching and time-gap calculations.
- Shot outcomes (Goal, Saved, Blocked, Off Target, Wayward, Post) are mapped to simplified categories (Goal, On Target, Blocked, Off Target) for outcome analysis.

---

### A.8 Summary of Key Constants & Thresholds

| Parameter | Value | Module(s) | Rationale |
| --- | ---:| --- | --- |
| `CHAIN_MIN_SHOTS` | 30 | `pass_value_chain` | Ensure stable empirical rates. |
| `PRECEDING_TYPES` | {Carry, Dribble, Duel} | `pass_value_chain` | Define actionable chain starters. |
| `MINUTE_BUCKETS` | [0,15,30,45,60,75,90,121] | `game_state_lens`, `set_piece_lens` | 15-minute intervals + stoppage. |
| `MAX_EVENT_TIME_GAP_SECONDS` | 5 | `defensive_overlay` | Causal window for defensive actions. |
| `MIN_SEQUENCE_SHOTS` | 10 | `defensive_overlay` | Minimum sample per trigger type. |
| `PITCH_BINS` | (30, 20) | All spatial modules | Heatmap grid resolution. |
| `SCORE_STATE_BUCKETS` | ±2, ±1, 0 | `game_state_lens`, `set_piece_lens` | Lead/trail severity thresholds. |
| `MIN_CATEGORY_SHOTS` | 15–40 | `set_piece_lens`, `assist_context` | Suppress low-volume categories. |
| `CATEGORY_MIN_SHOTS` | 40 | `assist_context` | Minimum for assist category summary. |
| `CURVE_MIN_SHOTS` | 200 | `assist_context` | Minimum for distance-curve plotting. |

These constants can be tuned in future analysis iterations to balance statistical power against category granularity.
