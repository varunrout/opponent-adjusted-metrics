# Task: CxG+ Defensive Profile Clustering — Phase 2 (Phase 0 + Phase 1 ODI are complete and validated)

## Context

Phase 0 (identity spike) and Phase 1 (ODI pipeline) are complete and validated:
- `audit_outputs/cxg_analysis/odi_defprofile_spike/freeze_frame_identity_report.md`
- `audit_outputs/cxg_analysis/odi_phase1/odi_phase1_validation_report.md`
- Live tables: `oam_analysis.cxg_defensive_involvement_v1` (15,445 rows), `oam_analysis.cxg_odi_features_v1` (3,960 rows)

This task is independent of ODI — it does not need player identity, only positional geometry already derived in existing Gold feature tables (`cxg_defensive_360_features`, `cxg_line_shape_360_features`). Do not touch the ODI code/tables from Phase 1, and do not touch frozen S/E1-E12 feature code or `three_sixty_context.py` F-family derivations.

Follow the repository's existing governance style: typed contracts (`contracts.py` pattern), validation before anything is called done, evidence under `audit_outputs/`, no BigQuery table considered real until row-count/reconciliation checks pass. No model training, calibration, or promotion — this is feature engineering only, same phase discipline as the rest of `oam_analysis`.

## Build: Defensive profile clustering

**1. Feature selection.**
Extract pre-shot defensive shape features per 360-eligible shot (3,960-shot cohort, same population as `cxg_odi_features_v1`) from the existing `cxg_defensive_360_features` and `cxg_line_shape_360_features` Gold tables. Do not recompute geometry from scratch. Select a fixed-size, well-justified subset of shape scalars for clustering input — document the selection reasoning (e.g. why these columns and not others; check `docs/cxg_split_policy_and_parallel_plan.md` and `18_CXG_CHART_ANALYSIS_DECISIONS_LOCKED` content already in the repo/docs for known redundancy clusters within these families, and avoid feeding obviously redundant/near-duplicate columns into the same clustering run — e.g. the defensive_360 redundancy flags already documented: `defenders_within_8m` vs `local_defensive_density`, `actor_space` vs `nearest_defender_distance`). Handle missingness per the existing 360-family null-governance approach — many of these fields have documented eligibility gaps; do not silently impute without flagging the approach in the report.

**2. Train-only cluster fitting (hard requirement).**
Use `oam_analysis.cxg_match_splits_v1` / `cxg_plus_360_model_matrix_v1` to identify train-split shots. Fit clustering **exclusively on train-split shots** — this is not optional, it follows the same split discipline as `docs/cxg_split_policy_and_parallel_plan.md`. Validation and test shots must never influence centroid/model fitting.

**3. Model selection.**
Try k-means (or GMM if you judge it more appropriate — justify the choice) across a reasonable range of k (4-8). Report a cluster-quality metric (e.g. silhouette score) per k on the train split. Propose a specific k with justification — do not pick arbitrarily, and do not just default to the highest silhouette score if a smaller k gives more interpretable/stable clusters; explain the tradeoff you're making.

**4. Assignment.**
Assign every shot in the full 360-eligible cohort (train, validation, and test) a cluster label via nearest-centroid to the train-fit model. Validation/test shots must only ever be assigned, never used to influence the fit.

**5. Interpretation.**
For each cluster, write a human-readable interpretation derived from the actual mean/median feature values per cluster (e.g. "cluster 2: deep low block, high compactness, GK covering near post") — derived from real computed statistics, not guessed or assumed. Include per-cluster shot count and 360-cohort goal-rate to help judge whether clusters carry any visible signal separation (descriptive only — this is not a claim of predictive value, just a look at the data).

**6. Output table.**
`oam_analysis.cxg_defensive_profile_clusters_v1` — typed contract following `contracts.py`, joinable to `cxg_analysis_360_v1`/`cxg_plus_360_model_matrix_v1` on `event_id`. Include cluster label, the split the shot belongs to (train/validation/test, for traceability), and a `cluster_model_version` field for reproducibility (e.g. versioned identifier tied to the k chosen and feature set used, so a future re-fit doesn't silently overwrite without a version bump).

**7. Validation before calling this done.**
- Row-count reconciliation against the 360-eligible cohort (3,960, or explicitly account for any gap the same way Phase 1 did).
- Cluster size distribution — flag if any cluster is degenerate/near-empty (e.g. <2% of shots) and note whether that's expected (rare defensive shape) or a fitting problem.
- The interpretation writeup from step 5.
- Explicit statement of what missingness-handling approach was used and why it doesn't leak test/validation information into the fit.

## What NOT to do

- Do not add ODI or profile-cluster features into any CxG+ candidate model spec or run any train/validate/test model fitting — that's a separate, later phase requiring explicit sign-off.
- Do not touch Phase 1 ODI code/tables.
- Do not touch frozen S/E1-E12 feature code or existing `three_sixty_context.py` F-family derivations.
- Do not fit clustering on anything but the train split.

## Deliverables checklist

- [ ] Feature-selection reasoning documented in the report
- [ ] `oam_analysis.cxg_defensive_profile_clusters_v1` + typed contract
- [ ] k-selection evidence (silhouette or equivalent per k) and justification for chosen k
- [ ] Row-count reconciliation, cluster size distribution, missingness-handling explanation
- [ ] Per-cluster interpretation writeup with real computed statistics
- [ ] No model training, no changes to Phase 1 ODI or frozen feature code

Report back with a summary when complete.
