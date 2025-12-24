# cxA Phase Plan (Sequence-Based)

**Date:** 2025-12-24  
**Status:** Planning (Phase 2: cxA)  
**Scope:** Data ingestion → feature engineering → analysis → modelling methodology  

This plan extends the existing CxG stack by defining **cxA (contextual expected assists)** as a **sequence-based chance-creation** metric. The core principle is:

- An "assist" is not a single event; it is the **last $k$ actions** in a possession leading to a shot.
- The **sequence accumulates threat**; the **final pass converts** that accumulated threat into a shot.
- We build a baseline that is already sequence-aware, then move to enriched + stacked models using submodels.

---

## 1. Definitions

### 1.1 Assist Sequence (pre-shot window)
For each shot event $S$:

- Define $A_S$ as the ordered list of the attacking team’s actions in the same possession immediately preceding the shot, capped at $k$ actions:

$$A_S = [a_{t-k}, \dots, a_{t-1}]$$

where $a_t$ is the shot.

**Constraints (v1 defaults):**
- Same `match_id`
- Same `possession`
- Same attacking `team_id`
- Ordered by event time (period, minute, second) + stable tie-breaker (raw event ordering)

### 1.2 Baseline xA (industry)
Baseline xA is the key-pass definition:

- If a pass is the final pass directly preceding a shot: $xA = \text{CxG}(\text{shot})$ (or provider xG benchmark)
- Else: $xA = 0$

This is used as a benchmark, not the final target.

### 1.3 Baseline xA+ (recommended baseline for this project)
Baseline xA+ upgrades baseline xA to match the project’s neutralization philosophy and sequence framing.

**Shot value:**

$$V(S) = \text{NeutralCxG}(S)$$

**Attribution:** distribute $V(S)$ across the last $k$ actions $A_S$ using weights, then sum the pass credits for each passer.

Two baseline weighting options:

1) **Recency decay (fast/robust):**
- $w_i = \lambda^{\Delta_i}$ where $\Delta_i$ is actions-before-shot
- $\tilde{w}_i = \frac{w_i}{\sum_j w_j}$
- credit: $c(a_i) = \tilde{w}_i \cdot V(S)$

2) **Softmax football-informed weights (still baseline, no ML required):**

$$\text{score}(a)=\alpha_1\Delta xT(a)+\alpha_2\text{progress}(a)+\alpha_3\text{bypassProxy}(a)+\alpha_4\mathbb{1}[\text{is\_final\_pass}]+\alpha_5\text{passTypeWeight}$$

$$\tilde{w}_i = \frac{e^{\text{score}(a_i)}}{\sum_j e^{\text{score}(a_j)}}$$

This preserves the property that total credit equals $V(S)$.

### 1.4 Full cxA (sequence + context + opponent)
Full cxA is modeled as a chain:

$$\text{cxA} = P(\text{complete}\mid X)\times P(\text{shot within }k\mid \text{complete},X)\times E[\text{CxG of resulting shot}\mid \text{shot within }k,X]$$

**v1 simplification:** restrict to completed passes so $P(\text{complete})=1$, then add completion later.

---

## 2. Data Ingestion Requirements

### 2.1 Required fields for pass modelling
From normalized DB tables:
- `events`: match/time ordering, possession id, under_pressure, location_x/y, team_id, player_id
- `passes`: length, angle, height, type, body_part, recipient_player_id, is_cross, is_through_ball, outcome

### 2.2 Critical linkage requirements
We must be able to:
- order events reliably within match/possession
- identify shots in the same possession following a pass

### 2.3 Endpoint locations (strongly recommended)
For sequence/receive-zone features:
- pass end location `pass_end_x`, `pass_end_y` (or equivalent)

If endpoints are not persisted, extract from raw JSON or `extra_attributes` during feature building.

---

## 3. Feature Engineering Plan

We engineer features at two granularities:

### 3.1 Action-level features (for each action in $A_S$)
- Start zone, end zone (where applicable)
- Progress (Δx), centrality, box-entry flags
- Pass family: cross / through ball / cutback proxy / ground pass
- Under-pressure flag

### 3.2 Sequence-level features (aggregates over $A_S$)
- Threat accumulated: $\sum \Delta xT(a)$
- Total progress: $\sum \Delta x(a)$
- Counts by action type (Pass/Carry/Dribble)
- Tempo: gaps between actions; possession length/duration at shot time

### 3.3 Last-pass features (delivery features)
For the last pass in $A_S$ (if present):
- pass type/height/body part, length/angle
- receive zone (end_x/end_y bucket)
- context at the moment of the pass (minute bucket, score_diff, pressure)
- opponent features (global + zone)

### 3.4 Players bypassed (proxy)
True bypass counts require defender locations at pass time (usually unavailable in open event data). v1 uses proxies:
- zone leap / ΔxT magnitude
- progressive distance into high-value zones
- end location inside box / half-space

---

## 4. Analysis (EDA) Deliverables

Add cxA diagnostics analogous to `analysis/cxg_analysis`:
- shot creation rate by pass family and receive zone
- effect of pressure on shot creation
- opponent slices (weak vs strong defenses)
- sequence shape: typical action patterns in last $k$ actions before shots
- sanity checks: are set pieces dominating? are penalties excluded?

---

## 5. Modelling Methodology (Baseline → Enriched → Stacked)

### 5.1 Benchmarks
- **Benchmark 1:** provider xA = StatsBomb xG of resulting shot (key passes)
- **Benchmark 2:** internal baseline xA = CxG of resulting shot (key passes)
- **Benchmark 3:** baseline xA+ = sequence-attributed NeutralCxG

### 5.2 Submodels (priors)
Submodels output calibrated probabilities/logits used as inputs to a meta-learner.

**A) Pass completion model (classifier)**
- Target: `complete`
- Output: $\hat{p}_{comp}$

**B) Key-pass prior (classifier)**
- Target: `is_key_pass` (final pass immediately before shot in same possession)
- Output: $\hat{p}_{key}$

**C) Shot-within-k hazard (classifier)**
- Target: `shot_within_k` (shot occurs within next k actions or t seconds)
- Output: $\hat{p}_{shot}$

**D) Conditional shot quality model (regression or bucket mean)**
- Target: $\text{CxG}(\text{next shot})$ conditional on a shot occurring
- Output: $\widehat{E[\text{CxG}]}$

**E) Specialist gates (optional but recommended)**
- Cross specialist models
- Through-ball specialist models
- Set-piece gate (train separate branches for set pieces vs open play)

**F) Retention / turnover risk (recommended)**
- Target: turnover within h actions (or possession ends shortly)
- Output used to penalize over-valuing low-probability hero balls

### 5.3 Meta-learner (stacking)
Train a final model using:
- raw features (pass/sequence/opponent/context)
- submodel outputs (logits/probs)

This mirrors the existing CxG approach (priors + meta-learner) and improves calibration.

### 5.4 Neutralization (consistent with CxG)
Compute neutral cxA by fixing reference context/opponent:
- score_diff = 0
- minute bucket reference (e.g., 55th minute bucket)
- under_pressure = False
- opponent = average defensive profile

Report:
- `cxA - neutral_cxA`
- `cxA / neutral_cxA`

### 5.5 Validation protocol
- Use `GroupKFold` with `match_id`
- Metrics:
  - for hazard models: log loss, AUC, calibration (ECE)
  - for regression: MAE/RMSE, calibration by bins
  - stability: rank correlation across time windows
  - face validity: top creators by role/team

---

## 6. Practical Defaults (v1)

- Scope: **open play** only
- $k$ (actions before shot): **3** (tune 2–5)
- Shot window for hazard: **within next 3 actions** (or **10 seconds**) after pass completion
- Shot to attribute in-window: **first shot** (simplest, consistent)
- Shot value for creation credit: **NeutralCxG**

---

## 7. Implementation Roadmap (repo-aligned)

1) Ensure pass endpoints are accessible (persist or extract).
2) Build pass/sequence dataset generation:
   - per pass: features + labels (completion, shot_within_k, key pass)
   - per shot: pre-shot sequence aggregates
3) Implement baseline xA and baseline xA+ reports.
4) Train submodels A–D (and optional E/F branches), generate OOF predictions.
5) Train stacked meta-learner, calibrate, and evaluate.
6) Add neutralization + opponent-adjusted cxA metrics.

---

## 8. Open Decisions (to finalize before coding)

- Exact definition of action ordering tie-breaker (raw event ordering field).
- Whether to include set pieces in v1 (recommended: exclude, then add as separate branch).
- Whether the hazard window uses actions, seconds, or both.
- Whether to attribute value only to passes or to all actions (passes + carries).