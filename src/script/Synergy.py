# -*- coding: utf-8 -*-
"""
Muscle and Kinematic Synergy Extraction Script (NMF Decomposition).
------------------------------------------------------------------
This script automates the extraction of synergies from EMG and Joint Angle data
using Non-Negative Matrix Factorization (NMF). It can process specific 
participants or automatically scan the data directory for all available trials.

Architecture: src/script/Synergy.py
"""

import os
import sys
import pickle
import warnings
from pathlib import Path
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

# --- Path Management ---
# Add project root to sys.path for absolute imports from 'src' and 'config'
root_path = Path(__file__).resolve().parent.parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

# --- Local Project Imports ---
import config
from config import get_file_paths, load_parameters, VERBOSE, WARNING
from src.module.synergy_module import synergy_module

# --- Warning Configuration ---
if not WARNING: 
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.simplefilter(action='ignore', category=FutureWarning)


def get_all_try_ids():
    """
    Scans the EMG data directory to list all available trial IDs.
    
    Returns:
        list: List of strings containing trial IDs (e.g., ['P11_0', 'P11_7']).
    """
    # Use config paths to find the EMG folder
    paths = get_file_paths("temp_id") 
    emg_dir = Path(paths["emg_df"]).parent 
    
    try_ids = []
    if not emg_dir.exists():
        if VERBOSE: 
            print(f"❌ Data directory not found: {emg_dir}")
        return try_ids

    for file in os.listdir(emg_dir):
        if file.endswith("_emg.csv"):
            try_id = file.replace("_emg.csv", "")
            try_ids.append(try_id)
            
    return sorted(try_ids)


def process_and_save(try_id, foot="right"):
    """
    Executes NMF decomposition and saves results as .pkl files.
    
    Args:
        try_id (str): Trial identifier.
        foot (str): Target foot side ('right' or 'left').
    """
    if VERBOSE: 
        print(f"\n🚀 Computing synergies for trial: {try_id}...")

    paths = get_file_paths(try_id)

    # --- EMG Synergies Extraction ---
    try:
        emg_synergies = synergy_module(
            try_id,
            signal_type="emg",
            foot=foot,
            method="cycle", 
            n_synergies=None, # Auto-detection via VAF
            vaf_target=0.9, 
            channel_vaf_threshold=0.70
        )
        
        save_path = Path(paths["synergy_emg"])
        save_path.mkdir(parents=True, exist_ok=True)
        filename = f"synergy_emg_{try_id}.pkl"
        
        with open(save_path / filename, "wb") as f:
            pickle.dump(emg_synergies, f)
            
        if VERBOSE: 
            print(f"✅ EMG Synergies saved: {filename}")
            
    except Exception as e:
        if VERBOSE: 
            print(f"⚠️ EMG Synergy error for {try_id}: {e}")

    # --- Kinematic (Angle) Synergies Extraction ---
    try:
        angle_synergies = synergy_module(
            try_id,
            signal_type="angle",
            foot=foot,
            method="cycle", 
            n_synergies=None, 
            vaf_target=0.9, 
            channel_vaf_threshold=0.70
        )
        
        save_path_angle = Path(paths["synergy_angle"])
        save_path_angle.mkdir(parents=True, exist_ok=True)
        filename_angle = f"synergy_angle_{try_id}.pkl"
        
        with open(save_path_angle / filename_angle, "wb") as f:
            pickle.dump(angle_synergies, f)
            
        if VERBOSE: 
            print(f"✅ ANGLE Synergies saved: {filename_angle}")
            
    except Exception as e:
        if VERBOSE: 
            print(f"⚠️ ANGLE Synergy error for {try_id}: {e}")


def main():
    """
    Main execution flow. Loads parameters and runs processing.
    """
    params = load_parameters()
    
    participant_id = params.get("participant_id")
    condition = params.get("condition")

    # Mode Selection: Single Trial vs. Bulk Processing
    if participant_id is None or condition is None:
        if VERBOSE: 
            print("🌀 Automatic Mode: Scanning all available data...")
            
        all_try_ids = get_all_try_ids()
        for try_id in all_try_ids:
            try:
                process_and_save(try_id)
            except Exception as e:
                if VERBOSE: 
                    print(f"❌ Process failed for {try_id}: {e}")
    else:
        # Targeted processing
        try_id = f"{participant_id}_{condition}"
        process_and_save(try_id)


if __name__ == "__main__":
    main()