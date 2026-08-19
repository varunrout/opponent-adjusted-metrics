from __future__ import annotations

import json

import pyarrow as pa

from opponent_adjusted.pipelines.silver.contracts import (
    CONTRACTS,
    SILVER_SCHEMA_VERSION,
    table_arrow_schema,
    write_contract_json,
)


def test_contract_version_and_required_tables_present(tmp_path):
    assert SILVER_SCHEMA_VERSION == "statsbomb_silver_v1"
    for table in ["events", "shots", "possessions", "three_sixty_frames", "three_sixty_players"]:
        assert table in CONTRACTS

    out = tmp_path / "contract.json"
    path = write_contract_json(out)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["silver_schema_version"] == SILVER_SCHEMA_VERSION
    assert any(t["table"] == "events" for t in payload["tables"])


def test_events_schema_related_ids_is_repeated_string():
    schema = table_arrow_schema("events")
    field = schema.field("related_event_ids")
    assert pa.types.is_list(field.type)
    assert pa.types.is_string(field.type.value_type)
