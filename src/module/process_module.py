# -*- coding: utf-8 -*-
"""
Processing Module: Signal Preprocessing, Cycle Segmentation, and Normalization.
------------------------------------------------------------------------------
Handles the end-to-end transformation of raw data into segmented gait cycles.
- EMG: Demeaning, Bandpass, Rectification, and Lowpass (Envelope extraction).
- Kinematics: Joint angle filtering and anatomical reordering.
- Normalization: Temporal resampling to 101 points (0-100% gait cycle).

Architecture: src/module/process_module.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Literal, List, Union, Optional

# --- Internal Core Imports ---
from src.core.utils_processing import (
    signal_preprocessing,
    extract_signal_subset,
    normalize_cycles,
    split_into_cycles
)
from config import load_parameters, get_file_paths, VERBOSE

# === ANATOMICAL UTILS ===

def filter_and_sort_angles(angle_df: pd.DataFrame) -> pd.DataFrame: 
    """
    Selects specific joint angles and enforces an anatomical order: 
    Hip -> Knee -> Ankle (Proximal to Distal).
    
    Args:
        angle_df (pd.DataFrame): Raw angle data.
        
    Returns:
        pd.DataFrame: Sorted and filtered angles.
    """
    filtered_columns = []
    
    # 1. Extraction of relevant columns (Right leg primary focus)
    for col in angle_df.columns:
        if "Right" in col and any(joint in col for joint in ["Hip", "Knee", "Ankle"]):
            # Exclude Z-axis for Knee/Ankle if irrelevant for the study
            if (("Knee" in col or "Ankle" in col) and col.endswith("Z")):
                continue
            filtered_columns.append(col)

    # 2. Custom Sorting (Hip -> Knee -> Ankle)
    priority = {"Hip": 0, "Knee": 1, "Ankle": 2}
    
    filtered_columns.sort(key=lambda x: (
        next((priority[k] for k in priority if k in x), 99), 
        x  # Secondary sort by Axis (X, Y)
    ))

    return angle_df[filtered_columns]

# ====== MAIN MODULE ======

def processing_module(
    try_id: str, 
    signal_type: Literal["emg", "angle"],
    cycle: bool = False,
    foot: Literal["right", "left"] = "right"
) -> Union[pd.DataFrame, List[pd.DataFrame]]:
    """
    Standardized pipeline for trial processing.
    
    Args:
        try_id (str): Trial identifier.
        signal_type (str): 'emg' or 'angle'.
        cycle (bool): If True, segments data into individual gait cycles.
        foot (str): Side to extract if 'cycle' is True.

    Returns:
        Union: Full processed signal or a list of segmented, normalized cycles.
    """
    
    params = load_parameters()
    paths = get_file_paths(try_id)
    
    # --- 1. DATA LOADING & SPECIFIC PREPROCESSING ---
    if signal_type == "angle":
        data_df = pd.read_csv(paths["angle_df"])
        data_df = filter_and_sort_angles(data_df)
        fs = params["fs_cam"]
            
        # Kinematic processing: Lowpass filtering only (Standard: 6-20Hz)
        preprocessed = signal_preprocessing(
            data_df, 
            fs=fs,
            demean=False, 
            rectif=None, 
            f_low=0, f_high=0, 
            lowpass_freq=20, lowpass_order=4, 
            normalize=True,
            norm_type="minmax"
        )
        
    elif signal_type == "emg":
        data_df = pd.read_csv(paths["emg_df"])
        fs = params["fs_emg"]

        # EMG Envelope extraction: Bandpass -> Rectify -> Lowpass
        # 
        preprocessed = signal_preprocessing(
            data_df, 
            fs=fs,
            demean=True,
            rectif="fullwave",
            f_low=50, f_high=450,
            lowpass_freq=20,
            normalize=True,
            norm_type="max",
            force_positive=True
        )
         
    # --- 2. CYCLE SEGMENTATION & NORMALIZATION ---
    if cycle:
        # Locate the event timestamps (Initial Contact / Toe Off)
        event_path = Path(paths["running_event"]) / f"event_cycle_{try_id}.csv"
        
        if not event_path.exists():
            raise FileNotFoundError(f"❌ Event file missing: {event_path}. Run Stance_event.py first.")
            
        event_df = pd.read_csv(event_path)
        signal_duration = len(data_df) / fs

        # Filter events within signal bounds and by foot side
        event_df = event_df[
            (event_df["IC_time"] >= 0) & 
            (event_df["TC_time"] <= signal_duration) &
            (event_df["foot"] == foot)
        ].copy()

        if event_df.empty:
            raise ValueError(f"❌ No valid {foot} cycles found in trial {try_id}")

        event_df["cycle_id"] = np.arange(1, len(event_df) + 1)
        event_df = event_df.rename(columns={"IC_time": "IC", "TC_time": "TC"})
        
        preprocessed["Time"] = data_df.index / fs
        
        # 
        
        # Subset extraction based on config limits (stride_start / stride_count)
        subset = extract_signal_subset(
            preprocessed, 
            event_df, 
            max_cycles=params["stride_count"], 
            start_cycle=params["stride_start"], 
            time_column="Time"
        )      
        
        # Temporal Normalization (Standardizing to 101 points: 0% to 100% of cycle)
        normalized = normalize_cycles(
            signal_dict={"cycles": subset["cycles"], "data": subset["data"]},
            divisions=[100, 100], # Resample to 100 intervals (101 points)
            time_column="Time"
        )
        
        return split_into_cycles(normalized)
    
    return preprocessed