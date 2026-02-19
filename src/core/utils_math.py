# -*- coding: utf-8 -*-
"""
Core Mathematical Utilities.
----------------------------
Provides specialized mathematical operations for biomechanical analysis:
1. Fast temporal searching (Binary Search).
2. Spatial geometry (3D Euclidean distances between markers).
3. Statistical distance metrics for gait consistency.

Architecture: src/core/utils_math.py
"""

import numpy as np
import pandas as pd
import bisect
from typing import Optional, List, Union


def first_greater_or_equal(series: Union[pd.Series, List, np.ndarray], value: float) -> Optional[int]:
    """
    Finds the index of the first element in a sorted series that is >= value.
    Uses binary search (bisect) for O(log n) efficiency.

    This is primarily used to synchronize events (like Initial Contact) with 
    continuous time-series data.

    Args:
        series: A sorted sequence of numbers (e.g., timestamps).
        value: The target value to search for.

    Returns:
        Optional[int]: The index of the first match, or None if no such value exists.
    """
    # bisect_left returns the leftmost insertion point to maintain order
    index = bisect.bisect_left(series, value)
    return index if index < len(series) else None


def compute_3d_distance(static_df: pd.DataFrame, point_a: str, point_b: str) -> float:
    """
    Calculates the average 3D Euclidean distance between two markers.
    
    Commonly used for:
    - Estimating Leg Length (Hips to Ankle).
    - Marker gap validation.
    - Calculating stride length from foot markers.

    

    Args:
        static_df (pd.DataFrame): DataFrame with columns suffix ' X', ' Y', ' Z'.
        point_a (str): Prefix name of the first marker (e.g., 'R_ICT').
        point_b (str): Prefix name of the second marker (e.g., 'R_FAL').

    Returns:
        float: The mean distance across all frames in the static trial.
    """
    # List required axes
    axes = ['X', 'Y', 'Z']
    
    # Coordinate extraction with verification
    try:
        pos_a = np.stack([static_df[f"{point_a} {axis}"] for axis in axes], axis=0)
        pos_b = np.stack([static_df[f"{point_b} {axis}"] for axis in axes], axis=0)
    except KeyError as e:
        raise KeyError(f"❌ Marker columns not found in static trial: {e}")

    # Vectorized Euclidean distance calculation: sqrt((x2-x1)^2 + (y2-y1)^2 + (z2-z1)^2)
    distances = np.linalg.norm(pos_b - pos_a, axis=0)
    
    # We return the mean to account for slight marker vibrations during the static pose
    return float(np.mean(distances))