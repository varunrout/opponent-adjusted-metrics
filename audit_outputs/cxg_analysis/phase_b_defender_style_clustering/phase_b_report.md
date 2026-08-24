# CxG+ Phase B — Defender Playing-Style Clustering

Replaces the deprecated ODI (`nearest_defender_odi`) defender-quality signal with a
defender **style** signal: a K-Means clustering of the *mix* of defensive actions an
individual player performs, attached to each CxG+ shot via its nearest freeze-frame
defender as `nearest_defender_style_archetype`.

- Model version: `cxg_defstyle_kmeans_v1_k4_cxg_defstyle_action_mix_v1_s7`
- Feature set version: `cxg_defstyle_action_mix_v1_s7`
- `data_version` `b0bc9f22dd77c206ddedc1d742893b3bbe64baec`, `silver_schema_version` `statsbomb_silver_v1_2`
- Run date: 2026-08-22

**Two ODI flaws this phase was built to avoid, and how:**

1. *ODI consumed `statsbomb_xg`*, which is circular with this project's own xG
   benchmarking. `statsbomb_xg` is read **nowhere** in this phase — not in
   `analysis/defstyle/`, not in `scripts/materialize_cxg_defender_style_clusters.py`.
   The only table this phase reads that even *has* an `statsbomb_xg` column is
   `cxg_defensive_involvement_v1`, and only its `player_id` column is selected.
2. *ODI was computed from shot-facing "nearest defender" involvements* — a median of
   ~4 observations per player. This phase instead reads each player's **entire event
   history** from `oam_core.events`.

---

## Step 1 — Investigation findings

### `oam_core.events` schema (verified live)

All the columns named in the brief are present and typed as expected:
`event_type_id` (INTEGER), `event_type_name` (STRING), `player_id` (INTEGER),
`team_id` (INTEGER), `position_name` (STRING), `location_x`/`location_y` (FLOAT),
`minute`, `second` (INTEGER), `timestamp` (STRING), `duration` (FLOAT),
`under_pressure`, `counterpress` (BOOLEAN).

**Lineage gotcha:** `oam_core.events` holds 6,470,469 rows, which is *three* copies of
the same 2,156,823-row corpus, one per `silver_schema_version`
(`statsbomb_silver_v1`, `_v1_1`, `_v1_2`). Every query in this phase pins **both**
`data_version` and `silver_schema_version`; without that, all action counts triple.

### Duel sub-type availability — **NOT AVAILABLE**

Duel is used as a **single combined rate**, not split by aerial/ground or won/lost.
This was checked three ways rather than assumed:

- `oam_core.events` has no duel outcome or sub-type column. Its only outcome-ish flat
  columns are `under_pressure`, `counterpress`, `off_camera`, `out`. There is no
  STRUCT, ARRAY-of-STRUCT or JSON column on the table at all (the sole REPEATED field
  is `related_event_ids STRING`).
- There is **no qualifiers side-table and no `duels` table** anywhere in `oam_core`
  (18 tables: `ball_receipts`, `carries`, `competitions`, `disciplinary_events`,
  `dribbles`, `events`, `matches`, `passes`, `players`, `possessions`, `pressures`,
  `shot_freeze_frame_players`, `shots`, `starting_xi_players`, `substitutions`,
  `teams`, `three_sixty_frames`, `three_sixty_players`).
- The Silver contract (`pipelines/silver/contracts.py`) defines no duel table either,
  so the gap is in the Silver projection itself, not just in what got loaded.

This is a genuine gap, not a search failure: sub-typed detail **does** exist for other
families (`passes`, `dribbles`, `pressures`, `disciplinary_events` all carry
`outcome_id`/`outcome_name`). Recovering duel sub-type would require re-ingesting raw
StatsBomb JSON, which is out of scope here. Recorded in code as
`features.DUEL_SUBTYPE_AVAILABLE = False` and pinned by a test.

### Sample-size reality check

Measured live over the 610 split-assigned matches:

| statistic | value |
|---|---|
| players with ≥1 qualifying defensive action | 1,958 |
| total qualifying defensive events | 316,808 |
| median per player | **66** |
| mean per player | 162 |
| p25 / p75 / max | 22 / 167 / 2,078 |

**Correction to the brief:** the task described a median of "~186 relevant defensive
events per player". The measured median is **66**, and 186 does not match the mean
(162) either. The brief's *conclusion* still holds — 66 is an order of magnitude better
than ODI's ~4, so sample size is not the binding constraint — but the number is
recorded honestly rather than restated.

### Precedent and conventions reused

- **Preprocessing** — median `SimpleImputer` → `StandardScaler`, both fitted on the
  **train split only**, per `scripts/materialize_cxg_defensive_profile_clusters.py`
  (Phase 2). Replicated exactly.
- **Split** — the existing match-level split in `oam_analysis.cxg_match_splits_v1`
  (426 train / 92 validation / 92 test matches), produced by
  `scripts/run_cxg_split_analysis.py` and consumed by `cxg_plus_360_model_matrix_v1`.
  No new split invented. Seed `260821`, the repo-wide convention.
- **Nearest defender** — the existing definition materialised in
  `oam_analysis.cxg_defensive_involvement_v1`: from `shot_freeze_frame_players`, rows
  with `teammate = FALSE` and non-null `player_id`, minimum Euclidean distance to the
  shot location, tie-broken by `freeze_frame_player_ordinal`, period-5 shootouts
  excluded. No geometry recomputed.
- **Contracts** — `ColumnContract`/`TableContract` from
  `pipelines/silver/contracts.py`, imported read-only exactly as
  `analysis/defprofile/contracts.py` does.
- **Charts** — local render → GCS upload → **scoped delete-then-insert by `run_id`**
  on the chart registry, per `scripts/render_cxg_feature_eda_appendix.py`. The registry
  is never `CREATE OR REPLACE`d.

### Concurrency check (Phase A)

`git log --oneline -5` and `git status` were run on
`src/opponent_adjusted/features/cxg/contracts.py` before editing, and the file was
re-read immediately before the edit. Phase A's in-progress additions
(`SourceType` widened to include `"opponent_adjusted"`, plus the
`OPPONENT_ADJUSTED_FAMILIES` block) were left **completely untouched**. This phase
appends a *separate* `DEFENDER_STYLE_FAMILIES` block at the end of the file rather
than widening Phase A's candidate tuple and its
`OPPONENT_ADJUSTED_FAMILY_CANDIDATE_COUNTS` dict — minimal conflict surface, and
analytically the right call anyway (see the block's own comment: Phase A's candidates
are per-shot situation signals, this one is a per-player property looked up per shot).

Phase A advanced further *during* this session (adding
`features/cxg/phase_a_geometric.py`, `scripts/materialize_cxg_phase_a_geometric_features.py`,
`tests/features/cxg/test_phase_a_geometric.py`, and modifying
`analysis/odi/contracts.py`). None of those files were touched or committed by this
phase; the Phase B commit names only its own paths.

---

## Step 2 — Feature construction

Seven per-player features, from **all** of the player's events in `oam_core.events`
(not shot-facing rows):

`pressure_share`, `duel_share`, `interception_share`, `clearance_share`,
`block_share`, `foul_committed_share`, `fifty_fifty_share`.

### Normalisation: compositional shares, not per-90 — and why

Each player's vector is the **share** of their qualifying defensive actions falling
into each type (the seven sum to 1). Per-90 was considered and rejected:

- The question is a player's **style** — the *mix* of actions they choose — not how
  much defending they do. Per-90 conflates the two: a 90-minute centre-back and a
  20-minute substitute centre-back with identical habits get different per-90 vectors
  purely from volume and from their team's share of possession.
- Per-90 additionally requires per-player minutes reconstructed from
  `starting_xi_players` + `substitutions`, importing that reconstruction's own
  estimation error (and `analysis/odi/roster.py`'s documented period-boundary
  convention) for no gain on the style question.
- Shares are already scale-free, so volume information is carried purely by the
  sample-size threshold rather than leaking into the features.

The seven shares are linearly dependent (they sum to 1), so vectors lie on a
6-simplex. This is harmless for K-Means — a rank deficiency in the ambient space, not
a degeneracy of Euclidean distance on the simplex — and is preferable to dropping one
category, which would make the geometry depend on the arbitrary choice of reference.

### Sample-size threshold: **30 actions**

Chosen from the observed distribution, not assumed:

- The brief proposed ≥20. At n=20, a mid-frequency share (duel, ~0.14) has binomial
  SE ≈ 0.078 — more than half the mean. At n=30 that falls to ≈0.063, and the dominant
  category (pressure, ~0.60) has SE ≤ 0.09.
- The cost/benefit knee is at 30. Measured CxG+ shot coverage: 92.3% at ≥20,
  **89.8% at ≥30**, 82.7% at ≥50. Raising 20→30 costs 2.5 points; raising 30→50 costs
  a further 7.1 points for a much smaller noise reduction.
- 30 sits at roughly the 35th percentile of the per-player count distribution
  (median 66), excluding the genuinely unobservable tail without discarding ordinary
  squad players.

Players below 30 are **not** clustered and get **no** fallback archetype — NULL with
`style_archetype_null_reason = 'below_min_action_threshold'`.

### Train vs deploy vectors

Because the clustering unit is a **player** but the project's split is by **match**,
"train" is applied at the event level. Each player has two vectors:

- **TRAIN vector** — built only from events in train-split matches. Gates and drives
  the fit (1,197 players clear the threshold on train events alone).
- **DEPLOY vector** — the player's full event history. Scored through the
  *train-fitted* transform at assignment time (1,373 players).

The fit therefore never sees a validation/test-only player, while assignment uses the
best available estimate of each player's style. This is the player-level analogue of
Phase 2's "fit on train rows, predict on all rows".

*Note on imputation:* share vectors are built with a guaranteed non-zero denominator,
so no share is ever NaN and the median imputer is a structural no-op here. It is
retained so this phase's preprocessing is literally the same object graph as Phase 2's,
rather than a lookalike that would silently diverge if a future feature did admit
missingness. A test pins that the imputer fills from **train** medians.

---

## Step 3 — Cluster count determination

K-Means, fitted on the 1,197 train-eligible players, k ∈ 2..8, seed 260821, `n_init=10`.

| k | silhouette (train) | inertia (train) | Δ inertia | cluster sizes | min cluster fraction |
|---|---|---|---|---|---|
| 2 | **0.2606** | 6317.0 | — | 808 / 389 | 0.325 |
| 3 | 0.2348 | 5573.7 | 743.3 | 665 / 367 / 165 | 0.138 |
| **4** | 0.2112 | **4937.8** | **635.9** | 139 / 179 / 538 / 341 | 0.116 |
| 5 | 0.1837 | 4480.0 | 457.8 | 118 / 205 / 333 / 396 / 145 | 0.099 |
| 6 | 0.1822 | 4104.0 | 376.0 | 171 / 201 / 106 / 103 / 293 / 323 | 0.086 |
| 7 | 0.1873 | 3852.8 | 251.2 | 147 / 165 / 108 / 184 / 143 / 106 / 344 | 0.089 |
| 8 | 0.1532 | 3660.6 | 192.2 | 103 / 143 / 142 / 96 / 186 / 262 / 130 / 135 | 0.079 |

**k = 4 chosen.** Being straight about the trade-off: **silhouette does not select
k=4** — it declines monotonically from its maximum at k=2, so on silhouette alone the
answer would be "two clusters". k=4 is chosen on three other grounds:

1. **Elbow.** The inertia drop falls off sharply after k=4 (743 → 636 → **458** →
   376 → 251). k=4 is the last k that buys a large marginal reduction.
2. **Resolution.** k=2 and k=3 under-resolve: at k=2 the split is essentially
   "deep-block defenders vs everyone else", collapsing the duel-led and pressure-led
   behaviours that k=4 separates cleanly and that are positionally distinct (below).
3. **Balance.** At k=4 every cluster holds ≥11.6% of the train cohort. From k=5 the
   minimum falls under 10% and clusters start fragmenting rather than splitting.

The monotone-declining silhouette is itself expected here and is not evidence of "no
structure": the action-mix simplex is a *continuum* of playing styles with dense
interior, not a set of well-separated blobs, so any partition of it scores modestly.
Phase 2 faced exactly the same shape (its silhouette also peaked at the lowest k it
tested, 4 — 0.1506 — and declined thereafter) and also settled on k=4.

No deviation from K-Means was warranted.

---

## Step 4 — Validation

### Stability (`raw/stability.json`)

Three independent perturbations, because they fail for different reasons. All use
**Adjusted Rand Index**, which is invariant to cluster relabelling — it asks whether
the same *partition* is recovered, not whether the same integers were handed out.

| check | what it perturbs | result |
|---|---|---|
| Seed perturbation (5 alternative seeds) | K-Means initialisation | ARI 0.973 – 1.000, **min 0.973** |
| 80% subsample refit (25 draws) | which players are in the train set | **mean 0.943**, min 0.820 |
| Validation-split refit (506 players) | the entire fitting dataset | **ARI 0.865** |

The validation-split check is the strongest: a model refit from scratch on data the
train model never saw carves the same population the same way 86.5% of the way to
perfect agreement. The archetypes are a property of the data, not of the train split.

The stability harness is also tested for the ability to *fail*: `test_clustering.py`
includes a guard asserting that the same check collapses (bootstrap ARI < 0.8) on
isotropic noise, so a high score is informative rather than vacuous.

### Interpretability

Archetype labels are **derived mechanically** from the scaled centroids
(`labels.derive_archetype_labels`), never hand-assigned to a cluster index. The rule:
name the cluster after its single most over-represented action in z-space; if even
that action is under 0.5 SD above the cohort mean, call it muddy.

z-space rather than raw shares is essential — pressure is ~60% of all qualifying
actions, so a raw-share argmax would call *every* cluster a "presser".

Scaled centroids (cohort SDs from the mean; **bold** = dominant):

| cluster | pressure | duel | intercept | clearance | block | foul | 50/50 | label |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.30 | −0.28 | −0.09 | −0.43 | −0.05 | 0.03 | **2.23** | `unresolved_5050_annotation_density` |
| 1 | −0.18 | **1.51** | −0.60 | −0.36 | −0.46 | 0.33 | −0.35 | `duel_dominant_contester` |
| 2 | **0.73** | −0.51 | −0.27 | −0.54 | −0.26 | 0.11 | −0.26 | `high_volume_presser` |
| 3 | −1.18 | 0.12 | 0.77 | **1.22** | 0.67 | −0.36 | −0.32 | `deep_block_clearer` |

Raw centroid shares, cluster sizes and an independent position/competition cross-check:

| cluster | archetype | n (all) | n (train) | headline raw shares | modal position mix |
|---|---|---|---|---|---|
| 3 | **`deep_block_clearer`** | 381 (27.7%) | 341 | clearance 0.206, block 0.099, interception 0.069, pressure 0.433 (lowest) | **93.5% defenders**, 5% mid, 1% fwd |
| 2 | **`high_volume_presser`** | 652 (47.5%) | 538 | pressure 0.690 (highest), clearance 0.039 (lowest) | 53% midfielders, 33% fwd, 14% def |
| 1 | **`duel_dominant_contester`** | 186 (13.5%) | 179 | duel 0.229 (highest), interception 0.024 (lowest) | 53% forwards, 28% def, 19% mid |
| 0 | **muddy** — `unresolved_5050_annotation_density` | 154 (11.2%) | 139 | 50/50 0.0259 (≈10× cohort), everything else near cohort mean | mixed: 48% mid, 27% fwd, 25% def |

Position mix was **not** an input to the clustering — it is an external validity check,
and clusters 1–3 each pass it with a coherent positional identity.

### The muddy cluster — reported honestly, not narrated

**Cluster 0 is not a playing style and is labelled as such.** The evidence:

- Its *only* distinguishing feature is an elevated 50/50 share. Every other action sits
  within 0.43 SD of the cohort mean — it is "an average defender who happens to have
  more 50/50s recorded".
- 50/50 is 0.4% of all qualifying actions, and its **recording rate varies ~6× across
  competitions** in this corpus: 0.21% of qualifying events in competition 55/season 43,
  up to 1.29% in competition 55/season 282. Cluster 0's centroid (2.6%) is roughly
  double even the highest competition rate.
- 91% of its members come from competitions 43 and 55.
- Unlike clusters 1–3, it has **no positional coherence** (48/27/25 mid/fwd/def).

So cluster 0 is substantially tracking *which competition a player appears in*, not how
they defend. This is enforced in code, not just prose: `labels.py` routes any
50/50-dominant centroid to `MUDDY_5050_LABEL` by rule, the profile table carries an
`is_muddy` BOOL, and the rendered chart title is stamped `[MUDDY]`. 521 CxG+ shots
(13.2%) carry this label; any downstream consumer should treat it as "unclassified",
not as a fourth archetype.

**Recommendation for the (out-of-scope) qualification task:** consider re-running with
`fifty_fifty_share` dropped from the feature set and comparing, or folding cluster 0
into a explicit "unclassified" bucket. Not done here — changing the feature set after
seeing the labels would be exactly the kind of post-hoc fitting this phase is trying to
avoid, and the brief specifies 50/50 as an input.

---

## Step 5 & 6 — Deliverables and row-count reconciliation

### Tables

| table | grain | rows |
|---|---|---|
| `oam_analysis.cxg_defender_style_clusters_v1` | one row per player *(internal lookup — `player_id` allowed here only)* | 1,958 |
| `oam_analysis.cxg_defender_style_cluster_profile_v1` | one row per cluster | 4 |
| `oam_features.cxg_defensive_360_features` | +1 column `nearest_defender_style_archetype` | 3,960 (unchanged) |

### Defender coverage

| | count | share |
|---|---|---|
| Players with a non-null archetype | **1,373** | 70.1% |
| Players below the 30-action threshold (NULL + reason) | **585** | 29.9% |
| **Total in the lookup table** | **1,958** | 100% |

Restricted to the 876 distinct players who are actually a nearest defender on at least
one CxG+ shot, coverage is much better: **701 (80.0%)** have an archetype, 159 (18.2%)
are below threshold, 16 (1.8%) have zero qualifying defensive events at all.

### CxG+ shot coverage (`raw/shot_coverage.json`)

| outcome | shots | share |
|---|---|---|
| Non-null `nearest_defender_style_archetype` | **3,557** | **89.8%** |
| NULL — nearest defender below the 30-action threshold | 278 | 7.0% |
| NULL — no freeze-frame defender resolvable | 98 | 2.5% |
| NULL — defender has zero qualifying defensive events | 27 | 0.7% |
| **Total (360-eligible cohort)** | **3,960** | 100% |

Reconciles exactly: 3,557 + 278 + 98 + 27 = 3,960. Verified independently in BigQuery
against the Gold table: 3,557 non-null + 403 null = 3,960 rows, matching the pre-merge
row count (no rows created or lost by the merge).

Assigned archetype distribution across shots: `deep_block_clearer` 1,530,
`high_volume_presser` 1,268, `unresolved_5050_annotation_density` 521,
`duel_dominant_contester` 238.

Null reasons are recorded in the **local audit JSON**, not in the model-facing Gold
table, where the column is simply NULL. **No default or fallback archetype is ever
assigned** — pinned by `test_no_fallback_to_the_largest_cluster`.

### Identity non-exposure

`player_id` appears in `cxg_defender_style_clusters_v1` **only** (internal lookup,
needed to resolve a shot's nearest defender). The Gold column carries the archetype
**string alone** — no `player_id`, no `team_id`, no joinable cluster index.
`cxg_defensive_360_features` had no `player_id`/`team_id` column before this phase
(verified live: its only non-feature columns are `event_id`, `match_id`,
`data_version`, `feature_version`, `materialized_at`) and still has none. Enforced by
contract tests in `tests/analysis/defstyle/test_shot_join.py`.

### Additive merge (no clobbering)

The Gold column is applied by `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` followed by a
targeted `UPDATE` of that one column from a transient staging table (dropped
afterwards). The Gold table is **never** `CREATE OR REPLACE`d, so this cannot destroy
concurrent Phase A column work. The reset-then-apply ordering also means a re-run
cannot leave a stale archetype on a shot that no longer resolves.

---

## Step 7 — Charts

5 charts rendered under `run_id = phase-b-defstyle-20260822T035306Z`:

- One grouped bar chart per cluster — centroid action share vs cohort mean, so a reader
  can see which actions the archetype is actually built on. Muddy clusters are stamped
  `[MUDDY]` in the title.
- One overview chart — every cluster's centroid in scaled z-space, the space the labels
  were actually derived from.

Bar rather than radar, deliberately: the seven shares differ by two orders of magnitude
(pressure ~0.60, 50/50 ~0.005), and a radar's area encoding would render the small
categories invisible while implying a cyclic ordering the taxonomy does not have.

Pipeline: local HTML + PNG render → GCS upload
(`gs://oam-varun-260819-artifacts/analysis/cxg/phase-b-defstyle-20260822T035306Z/defender_style_charts/`)
→ registered in `oam_analysis.cxg_rendered_chart_registry_v1` by **scoped
delete-then-insert on this `run_id` only**. The registry table is created with
`exists_ok=True` and never `CREATE OR REPLACE`d. The run_id is phase-scoped so the
delete can only ever match rows this script itself wrote — verified after the run: the
registry holds this run's 5 rows plus all prior `cxg-analysis-*` runs intact.

---

## What was actually executed against live BigQuery

**Everything in this report was executed for real.** Local ADC credentials for project
`oam-varun-260819` (location `europe-west2`) were available and used. Specifically:

| step | status |
|---|---|
| Schema/table investigation, event-count distributions, threshold sweep, 50/50 competition breakdown | **executed live** (BQ MCP + `google-cloud-bigquery`) |
| `scripts/materialize_cxg_defender_style_clusters.py` full run | **executed live** |
| `oam_analysis.cxg_defender_style_clusters_v1` (1,958 rows) | **written live** |
| `oam_analysis.cxg_defender_style_cluster_profile_v1` (4 rows) | **written live** |
| `ALTER TABLE` + `UPDATE` of `nearest_defender_style_archetype` on `oam_features.cxg_defensive_360_features` | **executed live**, 3,557 rows populated |
| `scripts/render_cxg_defender_style_charts.py` — render, GCS upload, registry insert | **executed live**, 5 charts |
| Post-hoc row-count reconciliation queries | **executed live** |

Nothing in this report is a dry-run result or a projection. Every figure quoted above
came out of an actual query or an actual pipeline run.

---

## Out of scope (deliberately not done)

- Univariate / correlation / PCA / bivariate qualification of
  `nearest_defender_style_archetype` — a separate future task.
- Re-deriving duel sub-type from raw StatsBomb JSON.
- Any modification to `analysis/defprofile/` (Phase 2, frozen) — untouched.

## Files

**New**
- `src/opponent_adjusted/analysis/defstyle/{__init__,features,clustering,labels,contracts,shot_join}.py`
- `scripts/materialize_cxg_defender_style_clusters.py`
- `scripts/render_cxg_defender_style_charts.py`
- `tests/analysis/defstyle/{__init__,test_features,test_clustering,test_shot_join}.py`
- `audit_outputs/cxg_analysis/phase_b_defender_style_clustering/` (this report + `raw/`)

**Modified — left in the working tree, deliberately NOT in the Phase B commit**
- `src/opponent_adjusted/features/cxg/contracts.py` — appended `DEFENDER_STYLE_FAMILIES`
  only; Phase A's in-progress block untouched.

  This edit is **intentionally excluded from the Phase B commit**. Phase A has
  uncommitted work in the same file, and my block is appended directly after theirs, so
  the two form a single contiguous diff hunk that cannot be split. Committing the file
  would either drag Phase A's WIP into a Phase B commit, or (if the hunk were forced
  apart) produce a state where `SourceType` lacks the `"opponent_adjusted"` literal that
  my block's `source_type` uses. Leaving the edit staged-but-uncommitted in the working
  tree preserves both parties' work and lets it land naturally with Phase A's commit.

  Nothing required by the Phase B deliverables depends on this: the two analysis-table
  contracts and the Gold column contract all live in
  `src/opponent_adjusted/analysis/defstyle/contracts.py`, which **is** committed. The
  taxonomy registration is governance metadata only — the pipeline, the tables and the
  Gold column all work without it.
