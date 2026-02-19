# -*- coding: utf-8 -*-
"""
Running Stride Style Analysis Script.
------------------------------------------------------------------
Calculates spatio-temporal parameters for running gait, including:
- Duty Factor (DF)
- Stance and Flight phase durations
- Normalized Stride Length (using leg length from static trials)

This script processes gait events and correlates them with participant 
morphology and treadmill speed.

Architecture: src/script/Stride_type.py
"""

import os
import sys
from pathlib import Path
import pandas as pd

# --- Path Management ---
# Navigating from 'src/script/' up to project root
root_path = Path(__file__).resolve().parent.parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

# --- Local Project Imports ---
import config
from config import get_file_paths, load_parameters, VERBOSE
from src.module.stride_type_module import stride_style_compute


def get_all_try_ids():
    """
    Scans the data directory to list all available trial IDs.
    
    Returns:
        list: List of trial IDs (e.g., ['P11_0', 'P12_7']).
    """
    # Use config paths to locate the raw data folder
    paths = get_file_paths("temp_id") 
    # Logic: find the directory containing EMG or Angle files to identify trials
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
    Loads gait events, computes stride parameters, and saves the results.
    
    Args:
        try_id (str): Trial identifier (e.g., 'P11_0').
    """
    if VERBOSE: 
        print(f"\n🚀 Analyzing stride style for: {try_id}")

    paths = get_file_paths(try_id)
    
    # --- DATA LOADING ---
    # Load detected gait events (Contact/Release times)
    event_path = Path(paths["running_event"]) / f"event_cycle_{try_id}.csv"
    if not event_path.exists():
        raise FileNotFoundError(f"Event file missing: {event_path}")

    event_df = pd.read_csv(event_path)
    speed_df = pd.read_csv(paths["speed_df"], sep=";") 
    static_df = pd.read_csv(paths["static_df"])
    
    # --- METADATA PREPARATION ---
    # Parse ID: "P11_0" -> participant="P11", condition_val=0
    parts = try_id.split("_")
    participant = parts[0]
    condition_val = int(parts[1]) 

    # --- CALCULATIONS VIA MODULE ---
    # We process right and left legs separately then merge
    
    # Right Leg
    _, style_df_right = stride_style_compute(
        participant=participant, 
        condition=condition_val, 
        stance_df=event_df, 
        speed_df=speed_df, 
        static_df=static_df,
        mode="unipedal_right"
    )
    
    # Left Leg
    _, style_df_left = stride_style_compute(
        participant=participant, 
        condition=condition_val, 
        stance_df=event_df, 
        speed_df=speed_df, 
        static_df=static_df,
        mode="unipedal_left"
    )
    
    # Merge and sort by time to maintain chronological order
    full_style_df = pd.concat([style_df_right, style_df_left], ignore_index=True)
    full_style_df = full_style_df.sort_values(by="IC_time").reset_index(drop=True)
    
    # --- SAVING RESULTS ---
    save_dir = Path(paths["running_style"]) 
    save_dir.mkdir(parents=True, exist_ok=True)
    
    output_filename = f"stride_type_{try_id}.csv"
    full_style_df.to_csv(save_dir / output_filename, index=False)
    
    if VERBOSE:
        avg_df = full_style_df['DutyFactor'].mean()
        print(f"✅ Stride style saved: {output_filename}")
        print(f"   📏 Mean Duty Factor: {avg_df:.2f}%")


def main():
    """
    Main entry point for stride analysis.
    """
    params = load_parameters()
    
    participant_id = params.get("participant_id")
    condition = params.get("condition")

    # Mode Selection
    if participant_id is None or condition is None:
        if VERBOSE: 
            print("🌀 Automatic Mode: Processing all available trials.")
        
        all_trials = get_all_try_ids()
        for try_id in all_trials:
            try:
                process_and_save(try_id)
            except Exception as e:
                print(f"❌ Error during trial {try_id}: {e}")
    else:
        # Targeted single trial
        target_id = f"{participant_id}_{condition}"
        try:
            process_and_save(target_id)
        except Exception as e:
            print(f"❌ Error during trial {target_id}: {e}")


if __name__ == "__main__":
    main()