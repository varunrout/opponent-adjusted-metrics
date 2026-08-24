import pytest

from opponent_adjusted.analysis.defstyle.contracts import (
    CXG_DEFENDER_STYLE_CLUSTER_PROFILE_V1,
    CXG_DEFENDER_STYLE_CLUSTERS_V1,
    FORBIDDEN_GOLD_COLUMNS,
    GOLD_STYLE_ARCHETYPE_COLUMN,
    GOLD_TARGET_TABLE,
)
from opponent_adjusted.analysis.defstyle.features import (
    NULL_REASON_BELOW_THRESHOLD,
    NULL_REASON_NO_DEFENDER,
    NULL_REASON_NOT_CLUSTERED,
    STYLE_FEATURES,
)
from opponent_adjusted.analysis.defstyle.shot_join import (
    ShotArchetype,
    coverage_summary,
    resolve_shot_archetype,
)

LOOKUP = {
    11: "deep_block_clearer",
    22: "high_volume_presser",
    33: None,  # in the cluster table but below the sample-size threshold
}


def test_assigns_the_nearest_defenders_archetype():
    result = resolve_shot_archetype(11, LOOKUP)
    assert result == ShotArchetype("deep_block_clearer", None)


def test_no_defender_is_null_with_reason():
    result = resolve_shot_archetype(None, LOOKUP)
    assert result.archetype is None
    assert result.null_reason == NULL_REASON_NO_DEFENDER


def test_below_threshold_defender_is_null_with_reason_not_a_default():
    result = resolve_shot_archetype(33, LOOKUP)
    assert result.archetype is None
    assert result.null_reason == NULL_REASON_BELOW_THRESHOLD


def test_unknown_defender_is_null_with_its_own_distinct_reason():
    result = resolve_shot_archetype(999, LOOKUP)
    assert result.archetype is None
    assert result.null_reason == NULL_REASON_NOT_CLUSTERED


def test_no_fallback_to_the_largest_cluster():
    # Every unresolvable case must be NULL -- never the modal archetype.
    for defender in (None, 33, 999):
        assert resolve_shot_archetype(defender, LOOKUP).archetype is None


def test_archetype_and_null_reason_are_mutually_exclusive():
    with pytest.raises(ValueError):
        ShotArchetype("deep_block_clearer", NULL_REASON_NO_DEFENDER)
    with pytest.raises(ValueError):
        ShotArchetype(None, None)


def test_coverage_summary_reconciles_to_the_total():
    resolutions = [
        resolve_shot_archetype(d, LOOKUP) for d in (11, 22, 11, None, 33, 999, 999)
    ]
    summary = coverage_summary(resolutions)
    assert summary["total"] == 7
    assert summary["assigned"] == 3
    assert summary[NULL_REASON_NO_DEFENDER] == 1
    assert summary[NULL_REASON_BELOW_THRESHOLD] == 1
    assert summary[NULL_REASON_NOT_CLUSTERED] == 2
    nulls = sum(summary[r] for r in (
        NULL_REASON_NO_DEFENDER, NULL_REASON_BELOW_THRESHOLD, NULL_REASON_NOT_CLUSTERED
    ))
    assert summary["assigned"] + nulls == summary["total"]


def test_null_reasons_are_distinct():
    reasons = {NULL_REASON_NO_DEFENDER, NULL_REASON_BELOW_THRESHOLD, NULL_REASON_NOT_CLUSTERED}
    assert len(reasons) == 3


# --- contract / exposure guards ---


def test_gold_column_is_a_nullable_string_archetype_only():
    assert GOLD_STYLE_ARCHETYPE_COLUMN.name == "nearest_defender_style_archetype"
    assert GOLD_STYLE_ARCHETYPE_COLUMN.arrow_type == "string"
    assert GOLD_STYLE_ARCHETYPE_COLUMN.nullable is True
    assert GOLD_TARGET_TABLE == "cxg_defensive_360_features"


def test_gold_column_exposes_no_identity():
    assert FORBIDDEN_GOLD_COLUMNS == {"player_id", "team_id"}
    for forbidden in FORBIDDEN_GOLD_COLUMNS:
        assert forbidden not in GOLD_STYLE_ARCHETYPE_COLUMN.name


def test_player_id_lives_only_in_the_internal_lookup_table():
    lookup_columns = {c.name for c in CXG_DEFENDER_STYLE_CLUSTERS_V1.columns}
    profile_columns = {c.name for c in CXG_DEFENDER_STYLE_CLUSTER_PROFILE_V1.columns}
    assert "player_id" in lookup_columns
    assert FORBIDDEN_GOLD_COLUMNS.isdisjoint(profile_columns)
    assert "team_id" not in lookup_columns


def test_cluster_table_keys_and_style_features_present():
    assert CXG_DEFENDER_STYLE_CLUSTERS_V1.key == ["player_id", "cluster_model_version"]
    columns = {c.name for c in CXG_DEFENDER_STYLE_CLUSTERS_V1.columns}
    assert set(STYLE_FEATURES).issubset(columns)


def test_profile_table_carries_every_centroid_and_a_muddy_flag():
    columns = {c.name for c in CXG_DEFENDER_STYLE_CLUSTER_PROFILE_V1.columns}
    for feature in STYLE_FEATURES:
        assert f"{feature}_centroid" in columns
        assert f"{feature}_z" in columns
    assert "is_muddy" in columns
    assert CXG_DEFENDER_STYLE_CLUSTER_PROFILE_V1.key == ["cluster_label", "cluster_model_version"]


def test_cluster_label_is_nullable_so_below_threshold_stays_unassigned():
    by_name = {c.name: c for c in CXG_DEFENDER_STYLE_CLUSTERS_V1.columns}
    assert by_name["cluster_label"].nullable is True
    assert by_name["style_archetype"].nullable is True
    assert by_name["style_archetype_null_reason"].nullable is True
