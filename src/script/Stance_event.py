# -*- coding: utf-8 -*-
"""
Gait Event Detection Script (Initial Contact & Toe Off).
-------------------------------------------------------
Detects gait events using Motion Capture (MoCap) data from markers. 
It identifies Initial Contact (IC) and Toe Off (TO) times for both 
right and left feet to define gait cycles.

Architecture: src/script/Stance_event.py
"""

import os
import sys
import pandas as pd
from pathlib import Path

# --- Path Management ---
# Navigating from 'src/script/' up to project root
root_path = Path(__file__).resolve().parent.parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

# --- Local Project Imports ---
import config
from config import get_file_paths, load_parameters, VERBOSE
from src.core.utils_processing import filter_signal
from src.module.stride_event_module import cut_cycle_sport_mocap


def get_all_try_ids():
    """
    Scans the data directory to list all available trial IDs based on EMG files.
    
    Returns:
        list: Sorted list of trial IDs (e.g., ['P11_0', 'P12_7']).
    """
    # Use config paths to locate the raw data folder
    paths = get_file_paths("temp_id") 
    data_dir = Path(paths["emg_df"]).parent 
    
    try_ids = []
    if not data_dir.exists():
        if VERBOSE: 
            print(f"❌ Data directory not found: {data_dir}")
        return try_ids

    for file in os.listdir(data_dir):
        if file.endswith("_emg.csv"):
            try_id = file.replace("_emg.csv", "")
            try_ids.append(try_id)
            
    return sorted(try_ids)


def process_and_save(try_id):
    """
    Loads MoCap data, filters signals, detects events, and saves results.
    
    Args:
        try_id (str): Trial identifier (e.g., 'P11_0').
    """
    if VERBOSE:
        print(f"\n🚀 Detecting gait events for: {try_id}...")

    paths = get_file_paths(try_id)
    params = load_parameters()
    
    # --- DATA LOADING ---
    if not Path(paths["dot_df"]).exists():
        raise FileNotFoundError(f"MoCap file missing for {try_id}: {paths['dot_df']}")
        
    dot_df = pd.read_csv(paths["dot_df"])
    
    # --- SIGNAL PROCESSING ---
    # Signal filtering (Lowpass 20Hz as per standard biomechanics procedure)
    running_filt = filter_signal(
        dot_df, 
        params["fs_cam"], 
        filter_type="lowpass", 
        cutoff=20
    )

    # --- EVENT DETECTION ---
    # Calling the specialized module for cycle cutting
    # Ensure marker names (e.g., 'R_BP Z') match your CSV headers
    running_event_df = cut_cycle_sport_mocap(
        running_filt, 
        sport="running", 
        right_foot_col="R_BP Z", 
        left_foot_col="L_BP Z", 
        fs=params["fs_cam"],
        tolerance=10, 
        seuil=0.3
    )
    
    # --- SAVING RESULTS ---
    save_path = Path(paths["running_event"])
    save_path.mkdir(parents=True, exist_ok=True)
    
    output_filename = f"event_cycle_{try_id}.csv"
    running_event_df.to_csv(save_path / output_filename, index=False)
    
    if VERBOSE:
        print(f"✅ Events successfully saved: {output_filename}")


def main():
    """
    Main entry point for gait event detection.
    """
    params = load_parameters()
    
    participant_id = params.get("participant_id")
    condition = params.get("condition")

    # Mode Selection: Single Trial vs. Bulk Processing
    if participant_id is None or condition is None:
        if VERBOSE:
            print("🌀 Automatic Mode: Processing all files found in data_example.")
            
        all_trials = get_all_try_ids()
        
        if not all_trials:
            print("❌ No valid files found. Check your data_example folders.")
            return

        for try_id in all_trials:
            try:
                process_and_save(try_id)
            except Exception as e:
                print(f"❌ Error during trial {try_id}: {e}")
    else:
        # Targeted mode for a specific participant/condition
        target_id = f"{participant_id}_{condition}"
        try:
            process_and_save(target_id)
        except Exception as e:
            print(f"❌ Error during trial {target_id}: {e}")


if __name__ == "__main__":
    main()