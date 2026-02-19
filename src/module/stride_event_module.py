# -*- coding: utf-8 -*-
"""
MoCap Cycle Segmentation Wrapper Module.
---------------------------------------
Handles the detection of movement cycles for different sports (Running & Cycling)
using marker-based motion capture data. It acts as a bridge between raw marker 
trajectories and specific event-detection algorithms.

Architecture: src/module/stride_event_module.py
"""

import numpy as np
import pandas as pd
from typing import Literal, get_args, Union, List, Optional

# --- Internal Specialized Detection Algorithms ---
from src.external.running_stance_event import detect_running_stance, detect_stance_cut_errors
from src.external.cycling_crank_event import detect_top_crank_position
from src.core.utils_processing import filter_signal

# Type alias for clarity
TimeSeries = Union[np.ndarray, pd.Series, List[float]]


def cut_cycle_sport_mocap(
    data: Union[pd.DataFrame, np.ndarray, list],
    sport: Literal["running", "cycling"] = "running",
    right_foot_col: Optional[str] = None,
    left_foot_col: Optional[str] = None,
    fs: float = 300.0,
    tolerance: int = 7,
    seuil: float = 0.3
) -> pd.DataFrame:
    """
    Segments raw MoCap data into discrete movement cycles.

    Args:
        data (Union): Input data containing marker trajectories.
        sport (str): 'running' for IC/TO detection or 'cycling' for Top Dead Center.
        right_foot_col (str): Column name for the right side marker (e.g., 'R_BP Z').
        left_foot_col (str): Column name for the left side marker.
        fs (float): Sampling frequency in Hz.
        tolerance (int): Temporal window (samples) for peak/event validation.
        seuil (float): Amplitude threshold for event detection.

    Returns:
        pd.DataFrame: Sorted events with timestamps, indices, and foot side.
    """
    
    # --- 1. Validation of Sport Parameter ---
    if sport not in get_args(Literal["running", "cycling"]):
        options = get_args(Literal["running", "cycling"])
        raise ValueError(
            f"❌ Unsupported sport: '{sport}'. Available options: {', '.join(options)}."
        )
    
    # --- 2. Data Type Normalization ---
    if isinstance(data, list):
        data = pd.DataFrame(data)
    elif isinstance(data, np.ndarray):
        if data.ndim == 1:
            data = pd.DataFrame({right_foot_col or "signal": data})
        elif data.ndim == 2:
            data = pd.DataFrame(data)
        else:
            raise ValueError("❌ Data must be a 1D or 2D array.")
    elif not isinstance(data, pd.DataFrame):
        raise TypeError("❌ Data must be a pandas DataFrame, list, or numpy array.")

    # --- 3. Column Integrity Verification ---
    if right_foot_col is None and left_foot_col is None:
        raise ValueError("❌ Error: You must specify at least one marker column: "
                         "`right_foot_col` or `left_foot_col`.")

    missing_cols = [c for c in [right_foot_col, left_foot_col] if c and c not in data.columns]
    if missing_cols:
        raise ValueError(f"❌ Marker columns missing in data: {', '.join(missing_cols)}")

    # --- 4. Sport-Specific Processing ---
    
    # A. RUNNING: Detects Initial Contact (IC) and Toe Off (TO)
    if sport == "running":
        print(f"--- Segmenting Running Cycles ({fs}Hz) ---")
        
        results = []
        if right_foot_col:
            df_r = detect_running_stance(
                data[right_foot_col], fs=fs, foot="right", 
                tolerance=tolerance, threshold=seuil
            )
            results.append(df_r)

        if left_foot_col:
            df_l = detect_running_stance(
                data[left_foot_col], fs=fs, foot="left", 
                tolerance=tolerance, threshold=seuil
            )
            results.append(df_l)

        # Merge and sort by time
        df_events = pd.concat(results, ignore_index=True)
        df_events.sort_values(by="IC_time", inplace=True)
        df_events.reset_index(drop=True, inplace=True)
        
        # Biomechanical consistency check
        if right_foot_col and left_foot_col:
            detect_stance_cut_errors(
                df_events, data, sport, fs, 
                right_foot=right_foot_col, left_foot=left_foot_col
            )

        return df_events

    # B. CYCLING: Detects Top Dead Center (Crank high point)
    elif sport == "cycling":
        print(f"--- Segmenting Cycling Cycles ({fs}Hz) ---")
        
        results = []
        if right_foot_col:
            df_r = detect_top_crank_position(
                crank_signal=data[right_foot_col], fs=fs, foot="right"
            )
            results.append(df_r)
            
        if left_foot_col:
            df_l = detect_top_crank_position(
                crank_signal=data[left_foot_col], fs=fs, foot="left"
            )
            results.append(df_l)
            
        df_events = pd.concat(results, ignore_index=True)
        # Cycling uses 'indice' (sample index) for synchronization
        df_events.sort_values(by="indice", inplace=True)
        df_events.reset_index(drop=True, inplace=True)
        
        if right_foot_col and left_foot_col:
            detect_stance_cut_errors(
                df_events, data, sport, fs, 
                right_foot=right_foot_col, left_foot=left_foot_col
            )
                
        return df_events