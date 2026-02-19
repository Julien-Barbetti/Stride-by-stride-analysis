# -*- coding: utf-8 -*-
"""
Signal Processing and Cycle Segmentation Script (EMG & Angles).
--------------------------------------------------------------
This script processes raw biomechanical signals:
1. Filters signals (Bandpass for EMG, Lowpass for Angles).
2. Segments continuous data into individual gait cycles based on detected events.
3. Normalizes cycles to 101 time points (0-100% of gait cycle).
4. Saves the structured data as .pkl files for synergy extraction.

Architecture: src/script/Process_cycle.py
"""

import os
import sys
import pickle
import warnings
from pathlib import Path
import pandas as pd

# --- Path Management ---
# Navigating from 'src/script/' up to project root
root_path = Path(__file__).resolve().parent.parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

# --- Local Project Imports ---
import config
from config import get_file_paths, load_parameters, WARNING, VERBOSE
from src.module.process_module import processing_module

# --- Warning Configuration ---
if not WARNING: 
    warnings.simplefilter(action='ignore', category=FutureWarning)


def get_all_try_ids():
    """
    Scans the EMG data directory to list all available trial IDs.
    
    Returns:
        list: Sorted list of trial IDs (e.g., ['P11_0', 'P12_7']).
    """
    # Using config to find the source data directory
    paths = get_file_paths("temp_id") 
    data_dir = Path(paths["emg_df"]).parent 
    
    try_ids = []
    if not data_dir.exists():
        if VERBOSE: 
            print(f"❌ Source directory not found: {data_dir}")
        return try_ids

    for file in os.listdir(data_dir):
        if file.endswith("_emg.csv"):
            try_id = file.replace("_emg.csv", "")
            try_ids.append(try_id)
            
    return sorted(try_ids)


def process_and_save(try_id, foot="right"):
    """
    Processes both EMG and Kinematic signals into segmented cycles.
    
    Args:
        try_id (str): Trial identifier (e.g., 'P11_0').
        foot (str): Target foot side for segmentation ('right' or 'left').
    """
    if VERBOSE: 
        print(f"\n🚀 Processing gait cycles for trial: {try_id} ({foot.capitalize()} side)")

    paths = get_file_paths(try_id)

    # --- EMG CYCLE PROCESSING ---
    try:
        emg_cycles = processing_module(
            try_id=try_id, 
            signal_type="emg", 
            cycle=True, 
            foot=foot
        )
        
        save_path_emg = Path(paths["emg_cycle_data"])
        save_path_emg.mkdir(parents=True, exist_ok=True)
        filename_emg = f"emg_cycle_{try_id}.pkl"
        
        with open(save_path_emg / filename_emg, "wb") as f:
            pickle.dump(emg_cycles, f)
            
        if VERBOSE: 
            print(f"✅ EMG cycles processed and saved: {filename_emg}")
            
    except Exception as e:
        if VERBOSE: 
            print(f"⚠️ EMG Processing error for {try_id}: {e}")

    # --- KINEMATIC (ANGLE) CYCLE PROCESSING ---
    try:
        angle_cycles = processing_module(
            try_id=try_id, 
            signal_type="angle", 
            cycle=True, 
            foot=foot
        )
        
        save_path_angle = Path(paths["angle_cycle_data"])
        save_path_angle.mkdir(parents=True, exist_ok=True)
        filename_angle = f"angle_cycle_{try_id}.pkl"
        
        with open(save_path_angle / filename_angle, "wb") as f:
            pickle.dump(angle_cycles, f)
            
        if VERBOSE: 
            print(f"✅ ANGLE cycles processed and saved: {filename_angle}")
            
    except Exception as e:
        if VERBOSE: 
            print(f"⚠️ ANGLE Processing error for {try_id}: {e}")


def main():
    """
    Main entry point for cycle processing.
    """
    params = load_parameters()
    
    participant_id = params.get("participant_id")
    condition = params.get("condition")

    # Mode Selection: Bulk vs Targeted
    if participant_id is None or condition is None:
        if VERBOSE: 
            print("🌀 Automatic Mode: Scanning all available data...")
            
        all_trials = get_all_try_ids()
        
        if not all_trials:
            if VERBOSE: print("❌ No trials found to process.")
            return

        for try_id in all_trials:
            try:
                process_and_save(try_id)
            except Exception as e:
                if VERBOSE: 
                    print(f"❌ Failed to process {try_id}: {e}")
    else:
        # Targeted single-trial processing
        target_id = f"{participant_id}_{condition}"
        process_and_save(target_id)


if __name__ == "__main__":
    main()