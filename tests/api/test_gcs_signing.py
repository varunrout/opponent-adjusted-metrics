"""Tests for gcs_signing.py's short-lived signed URL generation.

Mirrors test_bigquery_client_singleton.py's mocking pattern — no real GCS
credentials, IAM impersonation, or network access required.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from opponent_adjusted.api import gcs_signing


@pytest.fixture(autouse=True)
def reset_signing_state():
    """The signing-credentials cache is a module-level singleton; isolate tests."""
    gcs_signing._signing_credentials = None
    gcs_signing._signing_init_attempted = False
    yield
    gcs_signing._signing_credentials = None
    gcs_signing._signing_init_attempted = False


def test_sign_gcs_uri_returns_none_for_missing_uri():
    assert gcs_signing.sign_gcs_uri(None) is None
    assert gcs_signing.sign_gcs_uri("") is None


def test_sign_gcs_uri_returns_none_for_non_gcs_uri():
    assert gcs_signing.sign_gcs_uri("https://example.com/not-a-gcs-uri") is None


def test_sign_gcs_uri_returns_none_when_credentials_unavailable():
    with patch.object(gcs_signing, "_init_signing_credentials", return_value=None):
        assert gcs_signing.sign_gcs_uri("gs://bucket/path/file.html") is None


def test_sign_gcs_uri_returns_signed_url_on_success():
    fake_credentials = MagicMock()
    fake_blob = MagicMock()
    fake_blob.generate_signed_url.return_value = "https://storage.googleapis.com/signed-url"
    fake_bucket = MagicMock()
    fake_bucket.blob.return_value = fake_blob
    fake_client = MagicMock()
    fake_client.bucket.return_value = fake_bucket

    with (
        patch.object(gcs_signing, "_init_signing_credentials", return_value=fake_credentials),
        patch.object(gcs_signing.storage, "Client", return_value=fake_client) as mock_client_cls,
    ):
        url = gcs_signing.sign_gcs_uri("gs://my-bucket/some/nested/path.html")

    assert url == "https://storage.googleapis.com/signed-url"
    mock_client_cls.assert_called_once_with(
        project=gcs_signing.PROJECT, credentials=fake_credentials
    )
    fake_client.bucket.assert_called_once_with("my-bucket")
    fake_bucket.blob.assert_called_once_with("some/nested/path.html")
    fake_blob.generate_signed_url.assert_called_once_with(
        version="v4",
        expiration=gcs_signing.SIGNED_URL_EXPIRATION,
        method="GET",
    )


def test_sign_gcs_uri_returns_none_when_signing_raises():
    """A single chart's signing failure must degrade to None, not raise."""
    fake_credentials = MagicMock()
    fake_bucket = MagicMock()
    fake_bucket.blob.side_effect = Exception("boom")
    fake_client = MagicMock()
    fake_client.bucket.return_value = fake_bucket

    with (
        patch.object(gcs_signing, "_init_signing_credentials", return_value=fake_credentials),
        patch.object(gcs_signing.storage, "Client", return_value=fake_client),
    ):
        url = gcs_signing.sign_gcs_uri("gs://my-bucket/some/path.html")

    assert url is None


def test_init_signing_credentials_is_cached_after_first_attempt():
    """_init_signing_credentials should only actually attempt impersonation once."""
    with patch.object(gcs_signing.impersonated_credentials, "Credentials") as mock_creds_cls:
        mock_creds_cls.return_value = MagicMock()
        with patch.object(gcs_signing.google.auth, "default", return_value=(MagicMock(), "proj")):
            first = gcs_signing._init_signing_credentials()
            second = gcs_signing._init_signing_credentials()

    assert first is second
    mock_creds_cls.assert_called_once()
