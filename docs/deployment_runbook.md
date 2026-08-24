# Deployment runbook — OAM dashboard on GCP

Backend (FastAPI) on Cloud Run, frontend (Next.js) on Firebase Hosting, both
in `oam-varun-260819`. Free default URLs (no custom domain), £10/month budget
alert, `oam-pipeline-sa` reused as the Cloud Run runtime service account, no
staging environment — this ships straight to production.

**This document only prepares artifacts and instructions.** Every command
below is written for *you* (Varun) to run from your own shell. Nothing in
this repo change executes a `gcloud`/`firebase` deploy, creates a GCP
resource, or grants an IAM role — same boundary as the earlier GCS-signing
IAM binding.

All investigation below was run read-only against the real project
(`oam-varun-260819`) on 2026-08-24 — nothing here is guessed.

---

## 1. Investigation findings

### 1.1 `oam-pipeline-sa`'s IAM — what it has, what it's missing

Current bindings, checked directly (`gcloud projects get-iam-policy`, BigQuery
dataset ACLs, the bucket's IAM policy, and the service account's own IAM
policy):

| Resource | Binding | Status |
|---|---|---|
| Project `oam-varun-260819` | `roles/bigquery.jobUser` | ✅ already granted |
| Project `oam-varun-260819` | `roles/run.developer` | ✅ already granted (not needed for the *running* service, harmless) |
| Dataset `oam_core` | dataset-level `WRITER` (explicit ACL entry) | ✅ already granted — more than the API needs (read-only), but sufficient |
| Dataset `oam_analysis` | dataset-level `WRITER` (explicit ACL entry) | ✅ already granted — same as above |
| Dataset `oam_ml` | **no entry for `oam-pipeline-sa` at all** — only `projectOwners`/`projectWriters`/`projectReaders` special groups and Varun's own user `OWNER` | ❌ **missing.** `oam-pipeline-sa` is not a member of any of those special groups (it only has `bigquery.jobUser`/`run.developer` at the project level, neither of which grants BigQuery dataset read). Both the CxG coverage endpoint (`cxg_coverage.py`, queries `oam_ml.cxg_event_v3_predictions`/`cxg_plus_v3_predictions`) and the Analysis tab (`bigquery_analysis_store.py`, queries `oam_ml.*_metrics`/`*_coefficients`) will fail in production without this. |
| Bucket `gs://oam-varun-260819-artifacts` | `roles/storage.objectAdmin` | ✅ already granted — this is what makes a *generated* signed URL actually resolve (GCS checks the signer's real object permissions when the URL is used, not just at signing time), so no bucket change needed |
| Service account `oam-pipeline-sa` (IAM policy on itself) | `roles/iam.serviceAccountTokenCreator` for `user:varun.rout898@gmail.com` only | ❌ **missing the self-grant.** `gcs_signing.py` impersonates `SIGNING_SERVICE_ACCOUNT_EMAIL` (`oam-pipeline-sa` itself) from whatever identity Application Default Credentials resolves to. Locally that's your user OAuth token (already granted, from the earlier session). On Cloud Run, ADC resolves to `oam-pipeline-sa`'s own metadata-server credentials — so the calling identity and the target identity become the *same* service account, and GCP still requires `serviceAccountTokenCreator` to be granted to a principal **on itself** for self-impersonation signBlob calls to succeed. This binding does not exist today. Without it, every rendered-chart signed URL in the Analysis tab will silently fall back to `None` (by design — `sign_gcs_uri` degrades gracefully) rather than erroring, so this would ship as a quiet feature regression, not a crash. |

Exact commands (not run — for you to execute):

```bash
# 1. oam_ml read access for oam-pipeline-sa (dataset-level, read-only —
#    matches least-privilege; WRITER isn't needed since the API only reads).
bq add-iam-policy-binding \
  --member="serviceAccount:oam-pipeline-sa@oam-varun-260819.iam.gserviceaccount.com" \
  --role="roles/bigquery.dataViewer" \
  oam-varun-260819:oam_ml

# 2. Self-impersonation grant for GCS signed-URL generation from Cloud Run.
gcloud iam service-accounts add-iam-policy-binding \
  oam-pipeline-sa@oam-varun-260819.iam.gserviceaccount.com \
  --member="serviceAccount:oam-pipeline-sa@oam-varun-260819.iam.gserviceaccount.com" \
  --role="roles/iam.serviceAccountTokenCreator" \
  --project=oam-varun-260819
```

Note on `bq`: on this machine `bq` fails with `python3.14: command not found`
(the Cloud SDK's bundled `bq.cmd` picked up a stray Python 3.14 on PATH that
doesn't have the right packages). If you hit the same error, set
`CLOUDSDK_PYTHON` to a real Python 3.11 install first, e.g.:
`export CLOUDSDK_PYTHON="C:\Users\USER\AppData\Local\Programs\Python\Python311\python.exe"`
(this fixed it when verifying the command above).

**What did *not* need a grant, contrary to the original assumption:**
Firebase Admin's read path (`dependencies.py`'s `verify_id_token` call, used
by `/v1/me`) doesn't call any Firebase Admin API and doesn't need any IAM
role — it only verifies the JWT signature against Google's public certs and
reads the `role` custom claim already embedded in the token. No
`roles/firebase.*` grant is needed for this to work in Cloud Run.

### 1.2 CORS — what's there today

`main.py`'s `CORSMiddleware` currently allows exactly one origin:
`http://localhost:3000`. I've added a commented-out placeholder line for the
production Firebase Hosting origin (`src/opponent_adjusted/api/main.py`) —
left commented because **the real hosting URL depends on the hosting site ID,
which doesn't exist until you run `firebase deploy` for the first time.** It
defaults to `<firebase-project-id>.web.app` (so `oam-varun-260819.web.app`
here) unless you create a non-default hosting site — confirm the actual value
with `firebase hosting:sites:list` after the first deploy, then uncomment and
fix that line, then redeploy the backend. Until that's done, the frontend's
calls to the API will be blocked by CORS in production (they'll work from
`localhost:3000` only).

### 1.3 Frontend hosting fit — Firebase Hosting web-frameworks integration, not a separate Cloud Run

Checked directly rather than assumed:

- No `app/**/route.ts` API routes, no `middleware.ts`, no server actions, no
  `next.config.js` output override (`reactStrictMode` only) — so this isn't a
  from-a-server-heavy app.
- **But** `npm run build`'s own output shows `/matches/[matchId]`,
  `/players/[playerId]`, and `/teams/[teamId]` as `ƒ (Dynamic — server-
  rendered on demand)`, not `○ (Static)`. Even though each of those pages is
  a `"use client"` component doing its own `fetch()` against the FastAPI
  backend at runtime (no server-side data fetching, no SSR data need), there's
  no `generateStaticParams` for them — Next can't know every possible
  match/player/team ID at build time, so it still needs a live Node process
  to serve the page shell for an arbitrary ID at request time. A plain static
  export (`next export` / Hosting's classic "just serve `out/`" mode) would
  break these three routes outright.
- Conclusion: this needs **Firebase Hosting's Next.js web-frameworks
  integration** (`firebase.json` with a `"source"` pointing at the Next app,
  no `"public"` key) — *not* a hand-rolled Cloud Run service running `next
  start`. The frameworks integration auto-detects the dynamic routes and
  deploys just those to a Cloud Functions (2nd gen, Cloud Run-backed)
  backend automatically, while everything static goes to Hosting's CDN. This
  is the actually-correct middle ground for this app: less to operate than a
  second manual Cloud Run service, while still handling the dynamic routes
  that plain static hosting can't.
- Firebase CLI isn't installed on this machine (`firebase: command not
  found`) — you'll need `npm install -g firebase-tools` (any recent version;
  the web-frameworks integration has been stable, not experimental, for a
  while, but confirm with `firebase --version` — if it's old enough to still
  require it, run `firebase experiments:enable webframeworks` first).
- One more thing worth flagging, found while checking whether this project
  is even Firebase-enabled: `gcloud services list --enabled` shows
  `firebase.googleapis.com` and `firebasehosting.googleapis.com` both
  `ENABLED` at the Service Usage layer, but a direct authenticated call to
  the Firebase Management API from this session got back
  `SERVICE_DISABLED`/`PERMISSION_DENIED`. That's inconsistent, and I can't
  fully resolve it read-only (most likely an OAuth-scope quirk of the token
  this session's `gcloud` produced, not a real state problem — but I can't
  rule out "this GCP project was never actually registered as a Firebase
  project" either). **Run `firebase projects:list` yourself after installing
  the CLI as the real check** — if `oam-varun-260819` isn't listed, you'll
  need to add Firebase to the project first (Firebase console → "Add
  Firebase to an existing GCP project", or `firebase projects:addfirebase
  oam-varun-260819`) before `firebase init hosting` will work. Given the app
  already uses Firebase Auth client-side against this same project,
  Firebase is almost certainly already attached — but confirm rather than
  assume.

`firebase.json` and `.firebaserc` are already prepared at the repo root (see
below) — `"source": "web"` so `firebase deploy` builds and deploys the
Next.js app in `web/` without needing to `cd` there first.

---

## 2. Config that's already environment-driven vs. hardcoded

**Frontend** — already fully env-driven, nothing to change:
- `NEXT_PUBLIC_API_BASE_URL` (`web/lib/api.ts`) — defaults to
  `http://localhost:8000` if unset; **must** be set to the real Cloud Run URL
  before the production build.
- `NEXT_PUBLIC_FIREBASE_API_KEY`, `_AUTH_DOMAIN`, `_PROJECT_ID`,
  `_STORAGE_BUCKET`, `_MESSAGING_SENDER_ID`, `_APP_ID` (`web/lib/firebase.ts`)
  — already guarded to degrade to guest-only if unset; for production you
  want the real values so Firebase Auth actually works.
- These are Next.js build-time env vars (`NEXT_PUBLIC_*` gets inlined at
  `next build`), so they need to exist in `web/.env.production` (already
  gitignored via `web/.gitignore`'s `.env*.local` pattern — note
  `.env.production` isn't `*.local`, so **do not commit real values into
  `web/.env.production`**; keep it as a local-only file you create before
  each deploy, the same way `.env.local` already works for dev) or in the
  shell environment when `firebase deploy` runs the build.

**Backend** — needs **zero** env vars set on Cloud Run for this deploy:
- `PROJECT = "oam-varun-260819"` and the dataset names (`oam_core`,
  `oam_analysis`, `oam_ml`) are hardcoded constants across `bigquery_store.py`
  / `bigquery_analysis_store.py` / `cxg_coverage.py`, not env-driven. That's
  fine functionally here — you're deploying to the exact project those
  constants already point at — but it does mean this code can't be pointed
  at a different GCP project without an edit. **Flagging, not changing**:
  making `PROJECT` read from an `OAM_PROJECT` env var (falling back to the
  current hardcoded value) would be a small, low-risk follow-up if you ever
  want a second environment, but it's out of scope for "ship straight to
  production once" and touches application logic, which this pass
  deliberately avoids.
- `FIREBASE_SERVICE_ACCOUNT_PATH` must **not** be set in the Cloud Run
  service config (see §3 below) — its absence is what makes
  `dependencies.py` fall through to `credentials.ApplicationDefault()`.
- `$PORT` is injected by Cloud Run automatically; the Dockerfile's `CMD`
  already reads it (`${PORT:-8080}`), never hardcode it.

---

## 3. Firebase Admin credentials — confirmed

`dependencies.py`'s `_init_firebase_admin()` already does the right thing:
`credentials.ApplicationDefault()` when `FIREBASE_SERVICE_ACCOUNT_PATH` is
unset. On Cloud Run, Application Default Credentials resolve to the
attached runtime service account (`oam-pipeline-sa`) automatically via the
metadata server — no JSON key file needs to exist anywhere, in the image or
otherwise. **Do not set `FIREBASE_SERVICE_ACCOUNT_PATH` as a Cloud Run env
var** — leaving it unset in the service config is the correct production
configuration, not an oversight to fix.

---

## 4. Artifacts prepared in this repo

- [`Dockerfile`](../Dockerfile) — multi-stage build (Poetry install into a
  venv in the builder stage, slim runtime copying just the venv + `src/`),
  non-root user, reads `$PORT`.
- [`.dockerignore`](../.dockerignore) — excludes `web/`, `tests/`, caches,
  `docs/`, `infra/`, anything credential-shaped.
- [`firebase.json`](../firebase.json) / [`.firebaserc`](../.firebaserc) —
  Hosting web-frameworks config pointing at `web/`, region `europe-west2`
  (matches the BigQuery datasets' location), `maxInstances: 3` on the
  frameworks backend.
- `src/opponent_adjusted/api/main.py` — added the commented CORS placeholder
  for the production Hosting origin (see §1.2).

---

## 5. Ordered runbook

Run everything from the repo root unless noted. Region is `europe-west2`
throughout, matching where the BigQuery datasets and the existing
`oam-containers` Artifact Registry repo already live.

### 5.1 Prerequisites (one-time)

```bash
# Firebase CLI isn't installed on this machine yet.
npm install -g firebase-tools
firebase login
firebase projects:list   # confirm oam-varun-260819 is listed (see §1.3)

# Docker Desktop (or another local Docker) must be running for the image
# build below. Confirm:
docker info
```

### 5.2 Build and push the backend image

The `oam-containers` Artifact Registry repo already exists in
`europe-west2` — reuse it rather than creating a new one.

```bash
gcloud auth configure-docker europe-west2-docker.pkg.dev

docker build -t europe-west2-docker.pkg.dev/oam-varun-260819/oam-containers/oam-dashboard-api:latest .

docker push europe-west2-docker.pkg.dev/oam-varun-260819/oam-containers/oam-dashboard-api:latest
```

### 5.3 IAM grants (from §1.1 — run before or right after the first deploy;

the service will start either way, but `oam_ml`-backed endpoints and signed
chart URLs won't work correctly until these are in place)

```bash
bq add-iam-policy-binding \
  --member="serviceAccount:oam-pipeline-sa@oam-varun-260819.iam.gserviceaccount.com" \
  --role="roles/bigquery.dataViewer" \
  oam-varun-260819:oam_ml

gcloud iam service-accounts add-iam-policy-binding \
  oam-pipeline-sa@oam-varun-260819.iam.gserviceaccount.com \
  --member="serviceAccount:oam-pipeline-sa@oam-varun-260819.iam.gserviceaccount.com" \
  --role="roles/iam.serviceAccountTokenCreator" \
  --project=oam-varun-260819
```

### 5.4 Deploy the backend to Cloud Run

```bash
gcloud run deploy oam-dashboard-api \
  --image=europe-west2-docker.pkg.dev/oam-varun-260819/oam-containers/oam-dashboard-api:latest \
  --project=oam-varun-260819 \
  --region=europe-west2 \
  --service-account=oam-pipeline-sa@oam-varun-260819.iam.gserviceaccount.com \
  --allow-unauthenticated \
  --min-instances=0 \
  --max-instances=3 \
  --memory=512Mi \
  --cpu=1
```

- `--allow-unauthenticated`: the API serves guest-role Explore-zone content
  with no login required (per the design spec's roles table), so it needs to
  be publicly reachable — this is the intended access model, not an
  oversight.
- `--min-instances=0`: no idle cost between visits — the whole point of
  Cloud Run for a low-traffic personal dashboard under a £10/month cap.
- `--max-instances=3`: a hard ceiling on concurrent billed instances. 3 is
  conservative for a single-visitor-at-a-time dashboard with no expected
  concurrent load — it caps worst-case cost (e.g., a traffic spike or a
  runaway retry loop) at 3x one instance's cost rather than leaving it
  unbounded, while still giving headroom for a handful of simultaneous
  visitors before anyone gets queued.

Capture the deployed URL from the command's output (`Service URL:` line) —
you'll need it for §5.5 and §5.6.

### 5.5 Fix CORS with the real Hosting origin, redeploy

After §5.6's first `firebase deploy`, get the real Hosting URL:

```bash
firebase hosting:sites:list
```

Edit `src/opponent_adjusted/api/main.py`, uncomment and correct the
`allow_origins` placeholder line to the real
`https://<site-id>.web.app` value, then repeat §5.2 and §5.4 to rebuild,
repush, and redeploy the backend with the fix.

### 5.6 Deploy the frontend to Firebase Hosting

```bash
# Create web/.env.production locally (NOT committed) with the real values:
#   NEXT_PUBLIC_API_BASE_URL=<the Cloud Run URL from §5.4>
#   NEXT_PUBLIC_FIREBASE_API_KEY=...
#   NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN=...
#   NEXT_PUBLIC_FIREBASE_PROJECT_ID=...
#   NEXT_PUBLIC_FIREBASE_STORAGE_BUCKET=...
#   NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID=...
#   NEXT_PUBLIC_FIREBASE_APP_ID=...
# (same values as web/.env.local, just pointed at the production API URL)

firebase deploy --only hosting --project=oam-varun-260819
```

This builds `web/` (via the frameworks integration) and deploys it. Run this
once before §5.5 to learn the real Hosting URL, then again after the CORS
fix redeploys the backend.

### 5.7 Set the £10/month budget alert

Billing account for this project, checked directly:
`0149E9-7FA8A6-2B88CB` (currency: GBP, so `10GBP` needs no conversion).

```bash
gcloud billing budgets create \
  --billing-account=0149E9-7FA8A6-2B88CB \
  --display-name="OAM dashboard — £10/month cap" \
  --budget-amount=10GBP \
  --filter-projects=projects/oam-varun-260819 \
  --threshold-rule=percent=0.5 \
  --threshold-rule=percent=0.8 \
  --threshold-rule=percent=1.0 \
  --threshold-rule=percent=1.0,basis=forecasted-spend
```

This emails the billing account's admins (default recipients) at 50%, 80%,
100% of actual spend and 100% of *forecasted* spend for the month — it's an
alert, not a hard spend cap (GCP budgets don't stop billing on their own);
if you want an actual cutoff, that needs a separate Cloud Function wired to
the budget's Pub/Sub notification, which is out of scope here.

### 5.8 Smoke test against the real live URLs

- [ ] `GET https://<cloud-run-url>/health` → `{"status": "ok"}`
- [ ] Open the Hosting URL, confirm the Overview page loads with no console
      errors (check for CORS failures specifically — the #1 likely first
      failure mode from §1.2/§5.5's ordering)
- [ ] **Guest flow**: browse Matches/Players/Teams with no login; confirm
      shot maps render, xG shows, CxG/CxG+ badges show on covered shots only
      (no placeholder on uncovered ones)
- [ ] `GET /v1/me` with no `Authorization` header → `{"role": "guest", "uid":
      null, "email": null}`
- [ ] **Viewer flow**: sign in with a non-admin Firebase account, confirm
      `/v1/me` resolves `role: "viewer"` and the Analysis tab stays
      inaccessible (403)
- [ ] **Admin flow**: sign in with the admin account, confirm `/v1/me`
      resolves `role: "admin"` and the Analysis tab loads
- [ ] **Signed-URL charts**: in the Analysis tab, confirm at least one
      rendered chart shows an actual image/iframe (a signed HTTPS URL), not
      a fallback raw `gs://` path — this is the one most likely to silently
      regress per §1.1's `serviceAccountTokenCreator` finding
- [ ] Confirm the budget alert exists: `gcloud billing budgets list
      --billing-account=0149E9-7FA8A6-2B88CB`
