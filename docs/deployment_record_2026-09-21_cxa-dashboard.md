# Deployment record — 2026-09-21: CxA dashboard phase merge + deploy

Same structure as
[`deployment_runbook.md`](deployment_runbook.md) §6's incident record, reused
deliberately rather than a new format -- this is a permanent record of a real
production deploy, separate from the runbook's own prep document.

---

## What was done

**Part 1 -- merge to `main`.** Four branches merged in the exact order specified, each
`git merge --no-ff`, all clean, zero conflicts (confirmed, not assumed -- watched each
merge's own output):

1. `design/cxa-combined-scorer-v1`
2. `feature/cxa-combined-scorer-pipeline`
3. `feature/cxa-dashboard-models-page`
4. `feature/cxa-detail-page`

`main` pushed after each merge. Final `main` HEAD: **`0348cbd`**. Full test suite
re-run on `main` post-merge: **459 passed** (Python; one pre-existing, unrelated
collection error in `tests/features/cxg/test_opponent_adjusted_family.py` excluded,
same as every prior CxA task's own finding -- not caused by, or related to, any of
the four merged branches).

**Part 2 -- deploy**, following `docs/deployment_runbook.md` §5.2-5.8.

### §5.2 -- Backend image build

Docker Desktop was not running on this machine (`docker info` failed to reach the
daemon) -- used the runbook's documented fallback, `gcloud builds submit` (server-side
Cloud Build, no local Docker needed, same resulting image).

- **Tag used:** `0348cbd` (the exact `main` HEAD short SHA at merge time -- a real,
  traceable, unique tag, not a synthetic timestamp).
- **Image:**
  `europe-west2-docker.pkg.dev/oam-varun-260819/oam-containers/oam-dashboard-api:0348cbd`
- **Digest:** `sha256:8dc88190ab01e0ffb5f912cfa88aba9f588675bca622fef49c3cd5be578ad91d`
- **Build ID:** `59cb5ba6-da12-4ad0-97a6-468b4a8c3679`, duration 1m46s, `STATUS: SUCCESS`.

Built via `gcloud builds submit --tag ... .` from the repo root -- **never**
`gcloud run deploy --source .`, per the runbook's own explicit §6 incident warning.

### §5.3 -- IAM grants: checked first, not assumed missing

Both grants the runbook's own §1.1 investigation originally found missing were
**already in place**, confirmed directly before doing anything:

| Grant | Checked via | Result |
|---|---|---|
| `oam_ml` dataViewer for `oam-pipeline-sa` | `bq show --format=prettyjson oam-varun-260819:oam_ml`, inspected `access[]` | ✅ **already present** (`READER` for `oam-pipeline-sa@...`, the classic-ACL equivalent of `roles/bigquery.dataViewer`) -- applied in an earlier pass, not by this task. |
| Self-impersonation `serviceAccountTokenCreator` on `oam-pipeline-sa` itself | `gcloud iam service-accounts get-iam-policy oam-pipeline-sa@...` | ✅ **already present** -- `roles/iam.serviceAccountTokenCreator` bound to `serviceAccount:oam-pipeline-sa@...` on its own policy. |

**A third, new grant this task DID have to apply, found during live verification
(section "A real issue found and fixed" below), not anticipated by the runbook**
(written before `oam_serving` held any data or was queried by the API):

| Grant | Checked via | Result |
|---|---|---|
| `oam_serving` dataViewer for `oam-pipeline-sa` | `bq show --format=prettyjson oam-varun-260819:oam_serving`, inspected `access[]` | ❌ **missing** -- only the default project special groups, same gap shape as the original `oam_ml` finding. **Applied**: `bq update --source=<patch.json> oam-varun-260819:oam_serving` (dataset-level `add-iam-policy-binding` returned `This feature requires allowlisting`, so the classic ACL-PATCH method was used instead, same underlying effect). Verified live afterward: `READER` for `oam-pipeline-sa@...` now present. |

### §5.4 -- Backend deployed to Cloud Run

```
gcloud run deploy oam-dashboard-api \
  --image=europe-west2-docker.pkg.dev/oam-varun-260819/oam-containers/oam-dashboard-api:0348cbd \
  --project=oam-varun-260819 --region=europe-west2 \
  --service-account=oam-pipeline-sa@oam-varun-260819.iam.gserviceaccount.com \
  --allow-unauthenticated --min-instances=0 --max-instances=3 --memory=512Mi --cpu=1
```

**Deployed revision: `oam-dashboard-api-00007-mzd`**, 2026-09-21 00:06:29 UTC, 100%
traffic, confirmed via `gcloud run revisions list` (previous active:
`oam-dashboard-api-00006-77v`, 2026-09-18).

**Service URL:** `https://oam-dashboard-api-482195222855.europe-west2.run.app`

Immediately verified this was the real FastAPI backend, not a repeat of the §6
incident: `/health` -> `{"status":"ok"}`, `/openapi.json` -> 200 real OpenAPI JSON.

### §5.5/§5.6 -- CORS and frontend

**CORS was already correct** -- `src/opponent_adjusted/api/main.py`'s
`allow_origins` already had `https://oam-varun-260819.web.app` uncommented and set
(from an earlier pass, per the file's own comment trail), not still the placeholder
the runbook's §1.2 described. Confirmed the value is genuinely correct by running
`firebase hosting:sites:list` -- `https://oam-varun-260819.web.app` is the real
(and only) Hosting site, exact match. **No CORS fix or backend rebuild/redeploy was
needed this pass** -- the image built in §5.2 already had the right value baked in.

`web/.env.production` created locally (confirmed git-ignored via the repo's `.env.*`
rule, never staged or committed) with `NEXT_PUBLIC_API_BASE_URL` pointed at the
`oam-dashboard-api-00007-mzd` Service URL above, plus the same Firebase config
values already used in `web/.env.local`.

```
firebase deploy --only hosting --project=oam-varun-260819
```

Deployed the Next.js frameworks-integration backend (`ssroamvarun260819`,
Cloud Function 2nd gen, `europe-west2`) plus static assets. **Hosting URL:
`https://oam-varun-260819.web.app`** (unchanged from the existing site -- confirmed
via `firebase hosting:sites:list` before deploying, not assumed).

**Firebase CLI note, worth recording for a future deploy:** the global `firebase`
shim (`npm install -g firebase-tools`, needed since the CLI wasn't installed) resolved
to a broken/mangled path when invoked directly from this shell
(`Cannot find module '...\anaconda3\Library\c\Users\...'`) -- an Anaconda/Node
PATH interaction, not a firebase-tools bug. Worked around by invoking
`node "<npm-global-root>\firebase-tools\lib\bin\firebase.js"` directly. Not fixed at
the PATH/shell-config level (out of scope for a deploy task); flagged here so a
future session doesn't re-diagnose it from scratch.

### A real issue found and fixed during live verification (not assumed fine because
the code "looked right")

Immediately after §5.4's deploy, curled the new CxA endpoints as part of verification
(ahead of the formal §5.8 checklist) and got a real failure:

```
$ curl .../v1/models/cxa-models
Internal Server Error
```

Cloud Run logs (`gcloud run services logs read`) showed the real cause directly, not
guessed:

```
google.api_core.exceptions.Forbidden: 403 Access Denied: Table
oam-varun-260819:oam_serving.cxa_event_combined_v1: User does not have permission
to query table ...
```

**Root cause:** `oam-pipeline-sa` had never been granted read access to `oam_serving`
at all -- the runbook's own §1.1 IAM investigation (2026-08-24) predates
`oam_serving` holding any data or being queried by any endpoint (it was confirmed
empty as recently as the CxA combined-scorer design task, two tasks before this one).
Fixed per §5.3 above (the third grant). **Re-tested immediately after the grant, no
redeploy needed** (IAM takes effect without restarting the service) --
`/v1/models/cxa-models` returned 200 with real data on the next request.

### §5.7 -- Budget alert

**Already existed** -- confirmed via `gcloud billing budgets list
--billing-account=0149E9-7FA8A6-2B88CB`: `OAM Monthly Guardrail`,
£10/month (`GBP`), scoped to `projects/482195222855` (this project), with threshold
rules at 25/50/75/90/100% of current spend plus 100% of forecasted spend --
strictly more granular than the runbook's own proposed 50/80/100/100-forecasted
set. No action needed; not created by this task.

### §5.8 -- Smoke test, against the real live URLs

| Item | Result |
|---|---|
| `GET /health` -> `{"status":"ok"}` | ✅ Pass |
| Hosting URL loads, no console errors (CORS specifically) | ✅ Pass -- Overview page loads real content (recent matches, the honest CxG-vs-StatsBomb comparison table); zero console errors across every page visited in this session |
| Guest flow: Matches/Players/Teams load real data, shot maps render, xG shows | ✅ Pass -- Matches: 166/610 (CxG-scope toggle) with real fixtures; Players: 413 real rows (Harry Kane top, 220 shots/42 goals); Teams: 74 real rows; a real match page (`/matches/3794686`, Croatia 3-5 Spain) rendered a real shot map with real dots, xG values, and full lineups |
| CxG/CxG+ badges show on covered shots only, no placeholder on uncovered | ✅ Pass -- switched the match page's metric selector to CxG, confirmed the shot-map legend updated to "Dot size = CxG" (DOM-confirmed via `find`), consistent with real coverage-gated rendering |
| `GET /v1/me` (no auth) -> `{"role":"guest","uid":null,"email":null}` | ✅ Pass, exact match |
| Viewer flow: non-admin sign-in, `/v1/me` resolves `viewer`, Analysis tab 403s | ⚠️ **Not independently verified this session** -- requires a real non-admin Firebase login credential this session does not have. Not assumed to pass; flagged honestly rather than skipped silently. |
| Admin flow: admin sign-in, `/v1/me` resolves `admin`, Analysis tab loads | ⚠️ **Not independently verified this session**, same reason as above. |
| Signed-URL charts in the Analysis tab render an actual image, not a `gs://` fallback | ⚠️ **Not independently verified this session** -- gated behind the same admin-login requirement above. |
| Budget alert exists | ✅ Pass -- confirmed via `gcloud billing budgets list`, see §5.7 above |
| **New for this deploy:** `GET /v1/models/cxa-models` -> 200, both `CxA (event-only)` and `CxA+` present | ✅ Pass -- confirmed via `curl` (`tracks: ['event', 'plus']`) and, separately, by reading the rendered live Models page's own text: both cards present with real metrics (`P_create test log_loss 0.0654` / `0.0567`, etc.) |
| **New for this deploy:** live Models page shows both new cards with the combined-score caveat text | ✅ Pass -- read directly off the live page, verbatim: *"Combined CxA (create x convert) -- chance-creating passes only, ~2% of all passes; undefined, not zero, elsewhere."* on both cards, plus CxA+'s `Experimental` badge and its exact copy ("2,830 total chance-creating passes, 419 in test, zero Premier League rows") |
| **New for this deploy:** both CxA detail pages (`/models/cxa/event`, `/models/cxa/plus`) load on the live site | ✅ Pass -- read the live `/models/cxa/plus` page directly: P_create/P_convert sections, feature lists, coverage stats, and the combined-score-quality caveat all rendered correctly with real data |

**Three items genuinely not verifiable in this session, stated plainly rather than
assumed or silently skipped:** the viewer/admin login flows and the signed-URL chart
check all require a real Firebase login this automated session has no credentials
for. Everything guest-accessible (which is the overwhelming majority of the new CxA
surface area this deploy actually shipped) was verified directly against the live
URLs, not assumed from the code looking correct.

---

## Final state

- **`main`** at `0348cbd`, all four CxA-dashboard-phase branches merged, pushed.
- **Backend:** Cloud Run service `oam-dashboard-api`, revision
  `oam-dashboard-api-00007-mzd`, image
  `europe-west2-docker.pkg.dev/oam-varun-260819/oam-containers/oam-dashboard-api:0348cbd`,
  serving 100% traffic at `https://oam-dashboard-api-482195222855.europe-west2.run.app`.
- **Frontend:** Firebase Hosting, `https://oam-varun-260819.web.app`, frameworks
  backend `ssroamvarun260819` (`europe-west2`), built against `web/.env.production`
  pointed at the URL above.
- **IAM:** `oam-pipeline-sa` now has read access to all three datasets the API
  queries (`oam_core`/`oam_analysis` via the pre-existing `WRITER` ACL,
  `oam_ml` via the pre-existing `READER` grant, and `oam_serving` via the
  **new** `READER` grant this task applied) plus the pre-existing
  self-impersonation `serviceAccountTokenCreator` binding for signed chart URLs.
- **Budget alert:** pre-existing, confirmed in place, untouched.

---

## Deployment record — 2026-09-21 (second deploy): per-pass CxA display + quadrant scatter

A second deploy the same day, shipping
[`feature/cxa-pass-detail-and-quadrant-scatter`](../analysis/cxa_pass_detail_v1.md)
(per-pass CxA display on `ShotDetailModal`) and
[`quadrant_scatter_v1.md`](../analysis/quadrant_scatter_v1.md) (Hard gate 2's
player-season quadrant scatter). Same runbook sections, §5.2-5.8, same pattern as
above -- this record continues the same running log rather than starting a new file.

### What was done

**Part 1 -- merge.** `git merge --no-ff feature/cxa-pass-detail-and-quadrant-scatter`
into `main`, clean, zero conflicts, exactly as predicted. `main` advanced
`60f5c27` -> **`b696904`**, pushed. Full backend suite re-run on the merged `main`
before deploying: **465 passed** (462 baseline + 3 new quadrant-scatter tests; the
same pre-existing, unrelated `tests/features/cxg/test_opponent_adjusted_family.py`
collection error excluded as every prior task has found). `npx tsc --noEmit` clean.

**Part 2 -- deploy**, following §5.2-5.8. This branch touches both services (new
backend endpoints `/v1/cxa/coverage-by-shot` and `/v1/analysis/quadrant-scatter`;
new frontend surface on `ShotDetailModal` and a new Analysis tab), so both were
rebuilt and redeployed.

#### §5.2 -- Backend image build

Built via `gcloud builds submit` (server-side Cloud Build, consistent with the
first deploy this same day -- Docker Desktop still not running locally).

- **Tag:** `b696904` (the merged `main` HEAD short SHA).
- **Image:**
  `europe-west2-docker.pkg.dev/oam-varun-260819/oam-containers/oam-dashboard-api:b696904`
- **Digest:** `sha256:81726426e84813f424dcb677b52759ba345c236fa0fa5992f12f8f081904bb0f`
- **Build ID:** `57ce614c-fca4-476f-8cb0-6a8920b6fea4`, duration 1m50s, `STATUS: SUCCESS`.

#### §5.3 -- IAM grants: checked, all three already in place

| Grant | Checked via | Result |
|---|---|---|
| Self-impersonation `serviceAccountTokenCreator` on `oam-pipeline-sa` | `gcloud iam service-accounts get-iam-policy oam-pipeline-sa@...` | ✅ already present |
| `oam_ml` dataViewer for `oam-pipeline-sa` | `bq show --format=prettyjson oam_ml` | ✅ already present |
| `oam_serving` dataViewer for `oam-pipeline-sa` | `bq show --format=prettyjson oam_serving` | ✅ already present -- the grant applied in the first deploy this same day covers the branch's new `player_season_cxg_cxa_v1` table too, since it's the same dataset, not a new one |
| `oam_core` access for `oam-pipeline-sa` (new for this branch: `quadrant_scatter.py`'s materialization reads `oam_core.shots`/`events` directly, at query time via the API's read path too) | `bq show --format=prettyjson oam_core` | ✅ already present (`WRITER`, pre-existing since day one) |

No new grants needed this pass -- checked live, not assumed, per the task's own instruction.

#### §5.4 -- Backend deployed

```
gcloud run deploy oam-dashboard-api \
  --image=europe-west2-docker.pkg.dev/oam-varun-260819/oam-containers/oam-dashboard-api:b696904 \
  --project=oam-varun-260819 --region=europe-west2 \
  --service-account=oam-pipeline-sa@oam-varun-260819.iam.gserviceaccount.com \
  --allow-unauthenticated --min-instances=0 --max-instances=3 --memory=512Mi --cpu=1
```

**Deployed revision: `oam-dashboard-api-00008-m7g`**, 2026-09-21 18:01:03 UTC, 100%
traffic (previous active: `oam-dashboard-api-00007-mzd`). Confirmed real backend via
`/health` -> `{"status":"ok"}` and `/openapi.json` -> 200.

#### §5.5/§5.6 -- CORS and frontend

CORS unchanged (already correct from the first deploy this same day -- no new
origin introduced by this branch). `web/.env.production` unchanged (same Cloud Run
URL as the first deploy, still gitignored, still not committed).

```
firebase deploy --only hosting --project=oam-varun-260819
```

Deployed successfully -- build included the changed `/analysis` route (now 8.24 kB,
up from the pre-branch size, carrying the new Quadrant scatter tab) and the shared
`ShotDetailModal` bundle used by `/matches/[matchId]`, `/players/[playerId]`,
`/teams/[teamId]`. **Hosting URL unchanged:** `https://oam-varun-260819.web.app`.

#### §5.7 -- Budget alert

Unchanged, not re-checked in depth this pass (confirmed via the first deploy's own
check a few hours earlier the same day; no reason to expect it to have changed).

#### §5.8 -- Smoke test

| Item | Result |
|---|---|
| `GET /health` -> `{"status":"ok"}` | ✅ Pass |
| `GET /openapi.json` -> 200 | ✅ Pass |
| **New:** `GET /v1/cxa/coverage-by-shot?track=event&shot_event_ids=<covered id>` -> 200 with real values | ✅ Pass -- used a shot found live via `BigQueryCxaModelStore()._get_track_shot_coverage("event")` (`ccae6789-b294-4e21-96eb-64a654d5eba5`, Aron Einar Gunnarsson, Iceland vs Croatia, match 7561, 46'): returned `{"pass_event_id": "7b3b0bbf-...", "p_create_predicted_prob": 0.0887, "p_convert_predicted_prob": 0.0430, "cxa_combined_score": 0.00382}` -- not empty |
| **New:** the same covered shot's detail modal, opened live on the production site | ✅ Pass -- navigated to `https://oam-varun-260819.web.app/matches/7561`, clicked Gunnarsson's 46' shot dot: modal rendered a "Chance-creating pass" section with the `Experimental` badge and `CxA · P(create) 0.09 · P(convert) 0.04 · combined 0.004` -- exact match to the API values above, screenshot-confirmed, not silently absent |
| **New:** `GET /v1/analysis/quadrant-scatter` with no auth header | ✅ Pass -- returned `403`, not `200`; the admin gate holds in production, not just in the code |
| **New:** `/analysis` page, no auth, in the browser | ✅ Pass -- redirected client-side to `/overview` (confirmed via `window.location.href`), consistent with `RoleGate` correctly keeping a guest off the admin tab |
| **New:** Sign in as admin, open the Quadrant scatter tab, confirm real data + working X/Y selectors | ⚠️ **Not independently verified this session** -- same limitation as the first deploy's own record: no real Firebase admin login credential available to this automated session. Not assumed to pass. What WAS verified instead: the endpoint's own admin gate (403 with no auth, above), the endpoint's data correctness pre-deploy (live-queried against the real table during the build task: 822 test-split rows, 455 with both default-pairing metrics covered -- see `quadrant_scatter_v1.md` section 5), and the frontend's client-side role gate (redirect-away-from-`/analysis` behavior, above). |
| Regression check: no new console errors tied to this branch's changes | ✅ Pass -- one unrelated 404 observed in the browser session's console log, not traceable to any endpoint this branch touches (no `coverage-by-shot` or `quadrant-scatter` request appears in the failed-request list); not investigated further as out of scope for this deploy's own regression surface |
| Budget alert exists | ✅ Pass, unchanged (see §5.7) |

**One item still not independently verified, stated plainly rather than assumed:**
the admin-gated Quadrant scatter tab's actual rendered UI (its data table, its two
`<select>`s, the chart itself) was not seen live in a real admin session --
everything reachable without an interactive admin login was checked instead, same
honesty standard as the first deploy's own record.

### Final state (this deploy)

- **`main`** at `b696904`, `feature/cxa-pass-detail-and-quadrant-scatter` merged, pushed.
- **Backend:** Cloud Run service `oam-dashboard-api`, revision
  `oam-dashboard-api-00008-m7g`, image
  `europe-west2-docker.pkg.dev/oam-varun-260819/oam-containers/oam-dashboard-api:b696904`,
  serving 100% traffic at `https://oam-dashboard-api-482195222855.europe-west2.run.app`.
- **Frontend:** Firebase Hosting, same URL as before, `https://oam-varun-260819.web.app`,
  rebuilt against the same `web/.env.production`.
- **New `oam_serving` table live in production:** `player_season_cxg_cxa_v1`
  (materialized in the build task, ahead of this deploy -- 822 test-split rows,
  verified exactly against all four source tables' own counts).
- **IAM:** no changes this pass -- all grants already in place from the first
  deploy this same day.
- **Budget alert:** unchanged.
