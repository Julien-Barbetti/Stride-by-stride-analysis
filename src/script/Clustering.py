# -*- coding: utf-8 -*-
"""
Synergy Clustering Script (Inter-Participant Analysis).
-------------------------------------------------------
This script performs hierarchical clustering on extracted synergies to:
1. Identify representative muscle and kinematic patterns across a population.
2. Group individual synergies based on spatial/temporal similarity.
3. Filter out specific participants or outliers.
4. Save the global cluster models for further visualization and interpretation.

Architecture: src/script/Clustering.py
"""

import sys
import pickle
from pathlib import Path
import pandas as pd

# --- Path Management ---
# Navigating from 'src/script/' up to project root
root_path = Path(__file__).resolve().parent.parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

# --- Local Project Imports ---
import config
from config import get_file_paths, VERBOSE, SHOW_PLOTS
from src.module.clustering_module import clustering_module 

def main():
    """
    Main execution flow for inter-participant clustering.
    """
    # Load standardized paths from central config
    paths = get_file_paths()
    
    # Toggle plots based on config settings
    plot_enabled = True if SHOW_PLOTS else False

    if VERBOSE:
        print("\n🚀 Starting Global Clustering Analysis...")

    # --- EMG CLUSTERING ---
    # Grouping muscle synergies
    if VERBOSE:
        print("--- Processing EMG Clusters ---")
        
    emg_clustered_data = clustering_module(
        signal_type="emg",
        participant_filter=None,
        fusion="M",              # Method for merging clusters
        max_overlap=0.05,        # Similarity threshold
        doublon=False,           # Prevent multiple synergies from same cycle in one cluster
        plot=plot_enabled,
        participants_to_exclude={"P05", "P09", "P20"} # Exclusion list (outliers)
    )

    # Save EMG Results
    save_path_emg = Path(paths["clustered_emg"])
    save_path_emg.mkdir(parents=True, exist_ok=True)
    filename_emg = "clustered_emg.pkl"
    
    with open(save_path_emg / filename_emg, "wb") as f:
        pickle.dump(emg_clustered_data, f)
        
    if VERBOSE:
        print(f"✅ EMG Clusters successfully saved: {filename_emg}")


    # --- KINEMATIC (ANGLE) CLUSTERING ---
    # Grouping joint angle patterns
    if VERBOSE:
        print("\n--- Processing Angle (Kinematic) Clusters ---")

    angle_clustered_data = clustering_module(
        signal_type="angle",
        participant_filter=None,
        fusion="M",
        max_overlap=0.05,
        doublon=False,
        plot=plot_enabled,
        participants_to_exclude={"P05", "P09", "P20"}
    )

    # Save Angle Results
    save_path_angle = Path(paths["clustered_angle"])
    save_path_angle.mkdir(parents=True, exist_ok=True)
    filename_angle = "clustered_angle.pkl"
    
    with open(save_path_angle / filename_angle, "wb") as f:
        pickle.dump(angle_clustered_data, f)
        
    if VERBOSE:
        print(f"✅ ANGLE Clusters successfully saved: {filename_angle}")


if __name__ == "__main__":
    main()