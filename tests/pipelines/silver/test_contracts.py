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
    assert SILVER_SCHEMA_VERSION == "statsbomb_silver_v1_2"
    for table in [
        "events",
        "starting_xi_players",
        "shots",
        "possessions",
        "three_sixty_frames",
        "three_sixty_players",
    ]:
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


def test_starting_xi_players_schema_and_key():
    table = CONTRACTS["starting_xi_players"]
    assert table.key == ["event_id", "lineup_ordinal", "data_version", "silver_schema_version"]
    assert [column.name for column in table.columns] == [
        "event_id",
        "match_id",
        "competition_id",
        "season_id",
        "event_index",
        "team_id",
        "team_name",
        "formation",
        "lineup_ordinal",
        "player_id",
        "player_name",
        "position_id",
        "position_name",
        "jersey_number",
        "data_version",
        "silver_schema_version",
    ]
