"""Tests for the StatsBomb HTTP source boundary."""

from urllib.error import URLError

from opponent_adjusted.ingestion import statsbomb_source
from opponent_adjusted.ingestion.statsbomb_source import (
    StatsBombSource,
    _load_json_url,
    build_raw_base_url_for_ref,
    fetch_json_with_retries,
    is_valid_source_ref,
)


def test_source_decodes_json_payload_from_mocked_bytes(monkeypatch):
    calls = []

    def fake_fetch(url: str) -> bytes:
        calls.append(url)
        return b'[{"competition_id": 43}]'

    monkeypatch.setattr(statsbomb_source, "_fetch", fake_fetch)

    assert _load_json_url("https://example.test/data/competitions.json") == [{"competition_id": 43}]
    assert calls == ["https://example.test/data/competitions.json"]


def test_source_uses_decode_boundary_for_competitions(monkeypatch):
    seen = []

    def fake_loader(url: str):
        seen.append(url)
        return [{"competition_id": 99}]

    monkeypatch.setattr(statsbomb_source, "_load_json_url", fake_loader)
    source = StatsBombSource(base_url="https://example.test/data")
    assert source.get_competitions() == [{"competition_id": 99}]
    assert seen == ["https://example.test/data/competitions.json"]


def test_source_retries_transient_failures(monkeypatch):
    attempts = 0
    sleeps = []

    def flaky_loader(_url: str):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise URLError("temporary failure")
        return [{"match_id": 7}]

    monkeypatch.setattr(statsbomb_source.time, "sleep", sleeps.append)

    result = fetch_json_with_retries(
        "https://example.test/events/7.json",
        fetch_json=flaky_loader,
    )

    assert result == [{"match_id": 7}]
    assert attempts == 3
    assert sleeps == [0.8, 1.6]


def test_source_returns_none_after_terminal_failure(monkeypatch):
    monkeypatch.setattr(statsbomb_source.time, "sleep", lambda _seconds: None)

    def failed_loader(_url: str):
        raise URLError("permanent failure")

    assert (
        fetch_json_with_retries(
            "https://example.test/matches/43/3.json",
            retries=2,
            fetch_json=failed_loader,
        )
        is None
    )


def test_source_ref_url_is_pinned_and_validated():
    source_ref = "b0bc9f22dd77c206ddedc1d742893b3bbe64baec"
    assert is_valid_source_ref(source_ref) is True
    assert (
        build_raw_base_url_for_ref(source_ref)
        == f"https://raw.githubusercontent.com/statsbomb/open-data/{source_ref}/data"
    )


def test_source_ref_validation_rejects_invalid_sha():
    assert is_valid_source_ref("master") is False
