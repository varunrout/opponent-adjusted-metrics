"""One-off: set the Firebase `role` custom claim on a user (default: admin).

This is an operational script, not part of the app's runtime — it exists so
promoting an account to admin is a deliberate, explicit action taken by a
human, not something the API or web app does automatically (see
src/opponent_adjusted/api/dependencies.py's get_role(), which only *reads*
this claim).

SAFETY: this script is dry-run by default. Without --confirm-live, it only
prints what it would do — resolved Firebase project, target uid/email, and
the claim that would be set — and makes NO Firebase Admin API calls (no
get_user, no get_user_by_email, no set_custom_user_claims). The actual write
only happens when --confirm-live is passed explicitly. This exists because
an earlier run of this script, meant only to verify its logic, executed
against the real project by accident (Application Default Credentials
happened to be live in that environment) — dry-run-by-default makes that
class of mistake structurally impossible instead of relying on remembering
not to pass real arguments.

Usage:
    # Dry run (default) — prints what would happen, touches nothing:
    poetry run python scripts/set_admin_claim.py --email you@example.com

    # Actually perform the write:
    poetry run python scripts/set_admin_claim.py --email you@example.com --confirm-live
    poetry run python scripts/set_admin_claim.py --uid <firebase-uid> --confirm-live
    poetry run python scripts/set_admin_claim.py --email you@example.com --role viewer --confirm-live

Firebase Admin initializes the same way the dashboard API does: via
Application Default Credentials, with FIREBASE_SERVICE_ACCOUNT_PATH as a
local-dev override pointing at a service account JSON file. Unlike the API,
this script does NOT degrade gracefully if Firebase Admin can't initialize —
a one-off operational tool that silently no-ops on a misconfigured
environment is worse than one that fails loudly, so any init error just
propagates and the script exits non-zero. (Firebase Admin still initializes
in dry-run mode, since resolving/printing the project id needs it — that's
local SDK/credential setup, not a call to any Firebase Auth API endpoint.)

IMPORTANT — Firebase ID tokens cache custom claims. Setting the claim here
takes effect on the account immediately, but any ID token the user already
holds (e.g. from an existing browser session) still carries the OLD claims
until it's refreshed. The affected user must sign out and back in, or
force-refresh their token client-side (`user.getIdToken(true)`), before
GET /v1/me reflects the new role.
"""

from __future__ import annotations

import argparse
import os
import sys

import firebase_admin
from firebase_admin import auth as firebase_auth
from firebase_admin import credentials

VALID_ROLES = ("guest", "viewer", "admin")


def _init_firebase_admin() -> firebase_admin.App:
    """Initialize Firebase Admin; raises on failure, deliberately not guarded.

    Mirrors the credential-selection logic in
    src/opponent_adjusted/api/dependencies.py's _init_firebase_admin(), minus
    the try/except that makes the API degrade to guest-only — this script has
    no useful fallback if it can't authenticate, so it should just fail.

    This alone does not call any Firebase Auth API — it only sets up local
    SDK/credential state. It runs in both dry-run and --confirm-live modes.
    """
    service_account_path = os.environ.get("FIREBASE_SERVICE_ACCOUNT_PATH")
    cred = (
        credentials.Certificate(service_account_path)
        if service_account_path
        else credentials.ApplicationDefault()
    )
    return firebase_admin.initialize_app(cred)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Set the Firebase 'role' custom claim on a user (one-off, operational)."
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--uid", help="Firebase user UID to promote.")
    target.add_argument("--email", help="Firebase user email to promote (resolved to a UID).")
    parser.add_argument(
        "--role",
        default="admin",
        choices=VALID_ROLES,
        help="Role to set on the user (default: admin).",
    )
    parser.add_argument(
        "--confirm-live",
        action="store_true",
        help=(
            "Actually perform the write against the real Firebase project. "
            "Without this flag (the default), the script only prints what it "
            "would do and makes no Firebase Admin API calls at all."
        ),
    )
    return parser.parse_args()


def _describe_target(args: argparse.Namespace) -> str:
    return f"uid={args.uid}" if args.uid else f"email={args.email}"


def _print_dry_run(args: argparse.Namespace, project_id: str | None) -> None:
    print("DRY RUN — no write performed, pass --confirm-live to actually set this claim")
    print(f"  Firebase project: {project_id or '<could not resolve project id from credentials>'}")
    print(f"  Target:           {_describe_target(args)}")
    print(f"  Would set claim:  role={args.role!r}")
    print(
        "  (existing custom claims on the account, if any, would be preserved — "
        "not shown here since reading them requires a live call, which dry-run skips)"
    )


def main() -> None:
    args = parse_args()
    app = _init_firebase_admin()
    project_id = getattr(app, "project_id", None)

    if not args.confirm_live:
        _print_dry_run(args, project_id)
        return

    user = (
        firebase_auth.get_user(args.uid)
        if args.uid
        else firebase_auth.get_user_by_email(args.email)
    )

    existing_claims = dict(user.custom_claims or {})
    updated_claims = {**existing_claims, "role": args.role}
    firebase_auth.set_custom_user_claims(user.uid, updated_claims)

    # Read back from Firebase to confirm the claim actually took, rather than
    # trusting that set_custom_user_claims() not raising means it worked.
    refreshed = firebase_auth.get_user(user.uid)
    confirmed_role = (refreshed.custom_claims or {}).get("role")

    if confirmed_role != args.role:
        print(
            f"FAILED: expected role={args.role!r} on {user.uid}, but read back {confirmed_role!r}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"OK: {user.uid} ({user.email or 'no email'}) now has role={confirmed_role!r}")
    print(
        "Note: this user must sign out and back in (or force-refresh their ID "
        "token) before GET /v1/me reflects the new role — Firebase ID tokens "
        "cache claims until refreshed."
    )


if __name__ == "__main__":
    main()
