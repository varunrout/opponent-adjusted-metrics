"""Test configuration loading."""

from opponent_adjusted.config import settings


def test_settings_load():
    """Test that settings load correctly."""
    assert settings.database_url is not None
    assert settings.data_root is not None
    assert settings.log_level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


def test_competition_filters():
    """Test competition filters configuration."""
    assert len(settings.competitions) == 4
    assert any(c["name"] == "FIFA World Cup" and c["season"] == "2018" for c in settings.competitions)
    assert any(c["name"] == "UEFA Euro" and c["season"] == "2020" for c in settings.competitions)


def test_zone_definitions():
    """Test zone definitions configuration."""
    zones = settings.zone_definitions
    assert "A" in zones
    assert "B" in zones
    assert "C" in zones
    assert "D" in zones
    assert "E" in zones
    assert "F" in zones


def test_pitch_dimensions():
    """Test pitch dimensions configuration."""
    assert settings.pitch_length == 120.0
    assert settings.pitch_width == 80.0
    assert settings.goal_center_x == 120.0
    assert settings.goal_center_y == 40.0


def test_neutralization_context():
    """Test neutralization context configuration."""
    ctx = settings.neutralization_context
    assert ctx["score_diff_at_shot"] == 0
    assert ctx["minute"] == 55
    assert ctx["under_pressure"] is False
