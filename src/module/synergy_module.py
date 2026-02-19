# -*- coding: utf-8 -*-
"""
Synergy Extraction Module (NMF).
-------------------------------
Coordinates data processing and Non-Negative Matrix Factorization (NMF) 
to extract muscle or kinematic synergies. Supports both cycle-by-cycle 
analysis and mean cycle decomposition.

Architecture: src/module/synergy_module.py
"""

import numpy as np
import pandas as pd
from typing import List, Literal, Tuple, Optional, Dict, Union

# --- Project Imports ---
# Note: Relative imports are preferred within the src/ package structure
from src.core.nnmf_engine import nmf_per_cycle, nmf_decomposition
from src.module.process_module import processing_module 
from config import VERBOSE


def compute_cycle_mean(cycle_list: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Calculates the temporal mean (average) across multiple gait cycles.

    Args:
        cycle_list (List[pd.DataFrame]): List of DataFrames (each: time_points x channels).

    Returns:
        pd.DataFrame: A single DataFrame representing the average cycle.

    Raises:
        ValueError: If the list is empty or cycles have inconsistent lengths.
    """
    if not cycle_list:
        raise ValueError("The cycle list is empty.")

    # Validation: all cycles must have the same length (standardized to 101 points)
    n_rows = cycle_list[0].shape[0]
    if not all(df.shape[0] == n_rows for df in cycle_list):
        raise ValueError("Consistency Error: All cycles must have the same number of time points.")

    # Stack into 3D array: (n_cycles, n_time_points, n_channels)
    stack = np.stack([df.values for df in cycle_list], axis=0)

    # Calculate mean across the cycle axis (axis 0)
    mean_array = np.mean(stack, axis=0)

    return pd.DataFrame(mean_array, columns=cycle_list[0].columns)


def synergy_module(
    try_id: str,
    signal_type: Literal["angle", "emg"] = "emg",
    foot: Literal["right", "left"] = "right",
    method: Literal["cycle", "mean"] = "cycle", 
    n_synergies: Optional[int] = None, 
    vaf_target: float = 0.9, 
    channel_vaf_threshold: float = 0.70
) -> Union[List[Dict], Dict]:
    """
    Main orchestration function for synergy extraction.
    
    1. Fetches segmented cycles via process_module.
    2. Executes NMF based on the chosen method (per cycle or global mean).

    Args:
        try_id (str): Trial identifier (e.g., 'P11_0').
        signal_type (str): Type of signal to process ('emg' or 'angle').
        foot (str): Side to analyze ('right' or 'left').
        method (str): 'cycle' to run NMF on each cycle, 'mean' for average cycle NMF.
        n_synergies (int, optional): Fixed number of synergies. If None, uses VAF target.
        vaf_target (float): Variance Accounted For target (e.g., 0.9 for 90%).
        channel_vaf_threshold (float): Minimum VAF required for every individual channel.

    Returns:
        Union[List[Dict], Dict]: NMF results (Weights and Activations).
    """
    
    # --- Step 1: Data Acquisition ---
    # Retrieve segmented cycles (usually 0-100% normalized)
    data_cycles = processing_module(
        try_id=try_id, 
        signal_type=signal_type,
        cycle=True,
        foot=foot
    )
    
    # --- Step 2: NMF Decomposition ---
    if method == "cycle":
        # Apply NMF to every individual cycle (returns a list of results)
        if VERBOSE:
            print(f"DEBUG: Running NMF on {len(data_cycles)} individual cycles.")
            
        result = nmf_per_cycle(
            data_cycles, 
            n_synergies=n_synergies,
            vaf_target=vaf_target, 
            channel_vaf_threshold=channel_vaf_threshold,
            verbose=VERBOSE
        )
    
    elif method == "mean":
        # Compute the average cycle first
        if VERBOSE:
            print("DEBUG: Computing mean cycle before NMF decomposition.")
            
        mean_cycle = compute_cycle_mean(data_cycles)
        
        # Apply NMF to the single mean cycle (returns a single result dict)
        result = nmf_decomposition(
            mean_cycle,
            n_synergies=n_synergies,
            vaf_target=vaf_target,
            channel_vaf_threshold=channel_vaf_threshold
        )
    else:
        raise ValueError(f"Unknown method: {method}. Choose 'cycle' or 'mean'.")
        
    return result