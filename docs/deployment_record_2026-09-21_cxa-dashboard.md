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
