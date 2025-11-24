"""Test geometric feature calculations."""

import math
import pytest
from opponent_adjusted.features.geometry import (
    calculate_distance,
    calculate_shot_angle,
    calculate_centrality,
    calculate_distance_to_goal_line,
    assign_zone,
    calculate_all_geometry_features,
)


def test_calculate_distance():
    """Test distance calculation."""
    # Shot from penalty spot (108, 40)
    distance = calculate_distance(108, 40)
    assert abs(distance - 12.0) < 0.1


def test_calculate_shot_angle():
    """Test shot angle calculation."""
    # Shot from center, straight on
    angle = calculate_shot_angle(108, 40)
    assert angle > 0  # Should have some angle
    
    # Shot from far away should have smaller angle
    angle_far = calculate_shot_angle(60, 40)
    angle_close = calculate_shot_angle(110, 40)
    assert angle_far < angle_close


def test_calculate_centrality():
    """Test centrality calculation."""
    # Shot from center
    centrality = calculate_centrality(40)
    assert centrality == 0
    
    # Shot from side
    centrality = calculate_centrality(50)
    assert centrality == 10
    
    # Shot from other side
    centrality = calculate_centrality(30)
    assert centrality == 10


def test_calculate_distance_to_goal_line():
    """Test distance to goal line."""
    dist = calculate_distance_to_goal_line(108)
    assert dist == 12
    
    dist = calculate_distance_to_goal_line(100)
    assert dist == 20


def test_assign_zone():
    """Test zone assignment."""
    # Zone A: Close central
    zone = assign_zone(10, 5)
    assert zone == "A"
    
    # Zone B: Close wide
    zone = assign_zone(10, 15)
    assert zone == "B"
    
    # Zone C: Mid central
    zone = assign_zone(15, 5)
    assert zone == "C"
    
    # Zone D: Mid wide
    zone = assign_zone(15, 15)
    assert zone == "D"
    
    # Zone E: Far central
    zone = assign_zone(25, 5)
    assert zone == "E"
    
    # Zone F: Far wide
    zone = assign_zone(25, 15)
    assert zone == "F"


def test_calculate_all_geometry_features():
    """Test calculation of all geometry features."""
    features = calculate_all_geometry_features(108, 40)
    
    assert "shot_distance" in features
    assert "shot_angle" in features
    assert "centrality" in features
    assert "distance_to_goal_line" in features
    assert "zone_id" in features
    
    assert features["distance_to_goal_line"] == 12
    assert features["centrality"] == 0
    assert features["zone_id"] in ["A", "B", "C", "D", "E", "F"]
