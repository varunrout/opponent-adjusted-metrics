"""Short-lived signed URL generation for gs:// rendered-chart artifacts.

Signing a GCS object URL from Application Default Credentials that lack a
private key (a user's own `gcloud auth application-default login` token,
or a Compute Engine/Cloud Run attached service account without an explicit
self-impersonation grant) requires the IAM Credentials API's `signBlob`,
reached here via `google.auth.impersonated_credentials`. That call needs
`roles/iam.serviceAccountTokenCreator` granted to the calling identity ON
the target service account specifically — plain project Owner/Editor does
NOT include this by default.

Confirmed against this project (oam-varun-260819) rather than assumed:
this environment's ADC is `varun.rout898@gmail.com`'s own user OAuth
token (via `gcloud auth application-default login`), which has
`roles/owner` at the project level. Attempting to impersonate either
`oam-pipeline-sa@oam-varun-260819.iam.gserviceaccount.com` or the default
Compute Engine service account both failed with:

    PERMISSION_DENIED: Permission 'iam.serviceAccounts.getAccessToken'
    denied on resource (or it may not exist).

`gcloud iam service-accounts get-iam-policy` on both service accounts
returned an empty policy — no one has serviceAccountTokenCreator bound on
either, and project-level Owner does not implicitly grant it (a
deliberate GCP security boundary against privilege escalation via service
account tokens). Signing will not work in this environment until that
IAM binding is granted explicitly, e.g.:

    gcloud iam service-accounts add-iam-policy-binding \\
      oam-pipeline-sa@oam-varun-260819.iam.gserviceaccount.com \\
      --member="user:varun.rout898@gmail.com" \\
      --role="roles/iam.serviceAccountTokenCreator"

(or the equivalent grant to whatever identity actually runs the API in a
given deployment — a Cloud Run service's own attached service account
would need this granted on itself, or on SIGNING_SERVICE_ACCOUNT_EMAIL if
signing as a different identity).

This module does not attempt to route around that constraint — it fails
individual signing attempts and returns None, by design, so a chart whose
signing fails just falls back to the existing raw gs:// text display
rather than taking the whole /v1/analysis/charts endpoint down.
"""

from __future__ import annotations

import datetime
import logging
import re

import google.auth
from google.auth import impersonated_credentials
from google.auth.transport import requests as google_auth_requests
from google.cloud import storage  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

PROJECT = "oam-varun-260819"
SIGNING_SERVICE_ACCOUNT_EMAIL = "oam-pipeline-sa@oam-varun-260819.iam.gserviceaccount.com"
SIGNED_URL_EXPIRATION = datetime.timedelta(minutes=15)

_GCS_URI_PATTERN = re.compile(r"^gs://(?P<bucket>[^/]+)/(?P<blob>.+)$")

_signing_credentials = None
_signing_init_attempted = False


def _init_signing_credentials():
    """Lazily build impersonated credentials capable of signBlob; never raises.

    Mirrors the lazy-init-that-degrades-gracefully pattern already used by
    dependencies.py's _init_firebase_admin and bigquery_store.py's _client.
    """
    global _signing_credentials, _signing_init_attempted
    if _signing_init_attempted:
        return _signing_credentials
    _signing_init_attempted = True
    try:
        source_credentials, _ = google.auth.default()
        target_credentials = impersonated_credentials.Credentials(
            source_credentials=source_credentials,
            target_principal=SIGNING_SERVICE_ACCOUNT_EMAIL,
            target_scopes=["https://www.googleapis.com/auth/cloud-platform"],
            lifetime=300,
        )
        target_credentials.refresh(google_auth_requests.Request())
        _signing_credentials = target_credentials
        logger.info("GCS URL signing credentials initialized via %s", SIGNING_SERVICE_ACCOUNT_EMAIL)
    except Exception:
        logger.warning(
            "GCS URL signing unavailable — could not impersonate %s (likely missing "
            "roles/iam.serviceAccountTokenCreator on that service account for the "
            "calling identity); charts will fall back to raw gs:// paths",
            SIGNING_SERVICE_ACCOUNT_EMAIL,
            exc_info=True,
        )
        _signing_credentials = None
    return _signing_credentials


def sign_gcs_uri(gcs_uri: str | None) -> str | None:
    """Return a short-lived (15 min) signed HTTPS URL for a gs:// URI.

    Returns None — never raises — if the URI is missing/malformed, if
    signing credentials aren't available, or if the signing call itself
    fails for any reason. Callers should treat None as "fall back to
    showing the raw URI," not as an error to propagate.
    """
    if not gcs_uri:
        return None
    match = _GCS_URI_PATTERN.match(gcs_uri)
    if not match:
        logger.warning("Not a gs:// URI, skipping signing: %s", gcs_uri)
        return None

    credentials = _init_signing_credentials()
    if credentials is None:
        return None

    try:
        client = storage.Client(project=PROJECT, credentials=credentials)
        bucket = client.bucket(match.group("bucket"))
        blob = bucket.blob(match.group("blob"))
        return blob.generate_signed_url(
            version="v4",
            expiration=SIGNED_URL_EXPIRATION,
            method="GET",
        )
    except Exception:
        logger.warning("Failed to sign GCS URI %s", gcs_uri, exc_info=True)
        return None
