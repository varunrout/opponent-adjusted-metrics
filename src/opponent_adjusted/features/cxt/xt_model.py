"""Expected Threat (xT) model for pass value calculation.

Implements a 12x8 xT grid based on Karun Singh's methodology.
The grid divides the pitch into zones and assigns threat values
based on the probability of scoring from each zone.

StatsBomb coordinates: 120x80 yards (0,0 is attacking left)
"""

from __future__ import annotations

import logging
from typing import Tuple, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Standard xT grid dimensions (Karun Singh style)
XT_GRID_X = 12  # Columns (along pitch length)
XT_GRID_Y = 8   # Rows (along pitch width)

# StatsBomb pitch dimensions
PITCH_LENGTH = 120.0
PITCH_WIDTH = 80.0

# Pre-computed xT values based on historical shot conversion rates
# These values represent the probability of scoring from each zone
# Rows: y-zones (0=left touchline, 7=right touchline)
# Cols: x-zones (0=own goal, 11=opponent goal)
# Values from open-source xT implementations (similar to Karun Singh's)
XT_GRID = np.array([
    [0.00638303, 0.00779616, 0.00844854, 0.00977659, 0.01126267, 0.01248344, 0.01473596, 0.0174506 , 0.02122129, 0.02756312, 0.03485072, 0.0379259 ],
    [0.00750072, 0.00878589, 0.00942382, 0.0105949 , 0.01214719, 0.0138454 , 0.01611813, 0.01870347, 0.02401521, 0.0337291 , 0.0462449 , 0.05796000],
    [0.00609239, 0.00751141, 0.00872654, 0.01015174, 0.01248164, 0.01526558, 0.01869138, 0.02440067, 0.03490333, 0.05582126, 0.09480495, 0.13649532],
    [0.00625679, 0.00727291, 0.00829022, 0.00976936, 0.01251414, 0.01565719, 0.02073714, 0.02975722, 0.04744587, 0.08052886, 0.14653592, 0.25191519],
    [0.00612392, 0.0072446 , 0.00834982, 0.00984326, 0.01248706, 0.01573634, 0.0207515 , 0.02985173, 0.04747056, 0.08044485, 0.14667808, 0.2515406 ],
    [0.00606703, 0.0073493 , 0.00866198, 0.01014534, 0.01246318, 0.01527361, 0.01863814, 0.02433562, 0.03470189, 0.05541019, 0.09387358, 0.1357498 ],
    [0.00743598, 0.00873459, 0.00936573, 0.01058093, 0.01216362, 0.01385029, 0.01605012, 0.01863184, 0.0239342 , 0.03369837, 0.04615992, 0.05763928],
    [0.00630029, 0.00774052, 0.00844279, 0.00980329, 0.01123563, 0.01246948, 0.01472585, 0.01739875, 0.02115354, 0.02751304, 0.03482048, 0.03794799],
])


def get_zone(x: float, y: float) -> Tuple[int, int]:
    """Get the xT grid zone indices for a pitch location.
    
    Args:
        x: X coordinate (0-120, left to right in attacking direction)
        y: Y coordinate (0-80, bottom to top)
        
    Returns:
        Tuple of (x_zone, y_zone) indices into XT_GRID
    """
    # Clamp to pitch boundaries
    x = np.clip(x, 0, PITCH_LENGTH - 0.001)
    y = np.clip(y, 0, PITCH_WIDTH - 0.001)
    
    # Calculate zone indices
    x_zone = int(x / PITCH_LENGTH * XT_GRID_X)
    y_zone = int(y / PITCH_WIDTH * XT_GRID_Y)
    
    # Clamp to valid range
    x_zone = min(x_zone, XT_GRID_X - 1)
    y_zone = min(y_zone, XT_GRID_Y - 1)
    
    return x_zone, y_zone


def get_xt_value(x: float, y: float) -> float:
    """Get the xT value at a pitch location.
    
    Args:
        x: X coordinate (0-120)
        y: Y coordinate (0-80)
        
    Returns:
        xT value (probability of scoring from that zone)
    """
    x_zone, y_zone = get_zone(x, y)
    return float(XT_GRID[y_zone, x_zone])


def get_xt_delta(
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
) -> float:
    """Calculate xT change from a pass.
    
    Positive values indicate threat-increasing passes (toward goal).
    Negative values indicate threat-decreasing passes (backward).
    
    Args:
        start_x, start_y: Pass origin coordinates
        end_x, end_y: Pass destination coordinates
        
    Returns:
        Change in xT (end_xT - start_xT)
    """
    start_xt = get_xt_value(start_x, start_y)
    end_xt = get_xt_value(end_x, end_y)
    return end_xt - start_xt


def add_xt_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add xT features to a passes DataFrame.
    
    Expects columns: start_x, start_y, end_x, end_y
    
    Adds columns:
        - start_xt: xT value at pass origin
        - end_xt: xT value at pass destination
        - xt_delta: Change in xT (end - start)
        - start_zone_x: X zone index (0-11)
        - start_zone_y: Y zone index (0-7)
        - end_zone_x: X zone index (0-11)
        - end_zone_y: Y zone index (0-7)
        
    Args:
        df: DataFrame with pass coordinates
        
    Returns:
        DataFrame with xT columns added
    """
    logger.info(f"Adding xT features to {len(df):,} rows...")
    
    df = df.copy()
    
    # Vectorized xT calculation
    start_x = df["start_x"].fillna(0).values
    start_y = df["start_y"].fillna(0).values
    end_x = df["end_x"].fillna(0).values
    end_y = df["end_y"].fillna(0).values
    
    # Calculate zone indices
    df["start_zone_x"] = np.clip((start_x / PITCH_LENGTH * XT_GRID_X).astype(int), 0, XT_GRID_X - 1)
    df["start_zone_y"] = np.clip((start_y / PITCH_WIDTH * XT_GRID_Y).astype(int), 0, XT_GRID_Y - 1)
    df["end_zone_x"] = np.clip((end_x / PITCH_LENGTH * XT_GRID_X).astype(int), 0, XT_GRID_X - 1)
    df["end_zone_y"] = np.clip((end_y / PITCH_WIDTH * XT_GRID_Y).astype(int), 0, XT_GRID_Y - 1)
    
    # Lookup xT values using zone indices
    df["start_xt"] = XT_GRID[df["start_zone_y"].values, df["start_zone_x"].values]
    df["end_xt"] = XT_GRID[df["end_zone_y"].values, df["end_zone_x"].values]
    df["xt_delta"] = df["end_xt"] - df["start_xt"]
    
    # For incomplete passes, xT delta should be negative (lost possession)
    if "is_complete" in df.columns:
        # Incomplete passes lose the threat at start position
        df.loc[df["is_complete"] == False, "xt_delta"] = -df.loc[df["is_complete"] == False, "start_xt"]
        df.loc[df["is_complete"] == False, "end_xt"] = 0.0
    
    logger.info(f"  Mean xT delta: {df['xt_delta'].mean():.6f}")
    logger.info(f"  Positive xT passes: {(df['xt_delta'] > 0).sum():,} ({(df['xt_delta'] > 0).mean()*100:.1f}%)")
    
    return df


def get_zone_name(x_zone: int, y_zone: int) -> str:
    """Get a human-readable zone name.
    
    Args:
        x_zone: X zone index (0-11)
        y_zone: Y zone index (0-7)
        
    Returns:
        Zone name like "Defensive Third - Left", "Attacking Box - Center"
    """
    # X zones: thirds of the pitch
    if x_zone < 4:
        x_name = "Defensive Third"
    elif x_zone < 8:
        x_name = "Middle Third"
    else:
        x_name = "Attacking Third"
    
    # Attacking box (x >= 10 and y in [2,5])
    if x_zone >= 10 and 2 <= y_zone <= 5:
        x_name = "Attacking Box"
    
    # Y zones: width
    if y_zone < 2:
        y_name = "Left Wing"
    elif y_zone < 4:
        y_name = "Left Half"
    elif y_zone < 6:
        y_name = "Right Half"
    else:
        y_name = "Right Wing"
    
    # Central for middle y zones
    if 3 <= y_zone <= 4:
        y_name = "Central"
    
    return f"{x_name} - {y_name}"


def get_pitch_zone(x: float) -> str:
    """Get pitch third from x coordinate.
    
    Args:
        x: X coordinate (0-120)
        
    Returns:
        "Defensive", "Middle", or "Attacking"
    """
    if x < 40:
        return "Defensive"
    elif x < 80:
        return "Middle"
    else:
        return "Attacking"
