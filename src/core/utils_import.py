# -*- coding: utf-8 -*-
"""
Module: Data Import Utilities
-----------------------------
Handles loading of muscle synergy and kinematic data from pickle files 
and manages participant-specific speed calculations.
"""

import os
import pickle
import re
from typing import List, Literal, Tuple, Optional, Dict

import numpy as np
import pandas as pd


def find_running_speed(participant_id: str, condition: int, speed_df: pd.DataFrame) -> float:
    """
    Retrieves the running speed for a specific participant and condition.
    
    Args:
        participant_id: ID of the participant.
        condition: Slope condition (0 for 0%, 7 for 7%).
        speed_df: DataFrame containing speed mappings for participants.
        
    Returns:
        float: Speed in meters per second (m/s).
    """
    column = "Condition 0%" if condition == 0 else "Condition 7%"
    speed_kmh = speed_df.loc[speed_df["Participants"] == participant_id, column].values[0]
    
    # Conversion from km/h to m/s
    speed_mps = speed_kmh / 3.6
    return speed_mps


def load_synergy_list_legacy(directory: str) -> List[Tuple[str, str, any]]:
    """
    Loads all `.pkl` files in the specified directory into a list with metadata.
    Used for legacy EMG synergy structures.

    Args:
        directory: Path to the directory containing `.pkl` files.

    Returns:
        List[Tuple]: List of tuples containing (participant, condition, data).
    """
    synergy_data = []

    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory {directory} does not exist.")
    
    pkl_files = [f for f in os.listdir(directory) if f.endswith('.pkl')]

    if not pkl_files:
        print("⚠️ No .pkl files found in the directory.")
        return synergy_data

    for file_name in pkl_files:
        # Pattern for legacy synergy naming
        match = re.match(r"synergy_list_(\w+)_(\w+)_\d+\.\d\.pkl", file_name)
        if match:
            participant, condition = match.groups()
            file_path = os.path.join(directory, file_name)
            with open(file_path, 'rb') as file:
                data = pickle.load(file)
                synergy_data.append((participant, condition, data))
                print(f"✅ Loaded: {file_name}")
    
    print(f"\n✨ {len(synergy_data)} files loaded from {directory}.")
    return synergy_data


def load_synergy_angle_list(directory: str) -> List[Tuple[str, str, any]]:
    """
    Loads all kinematic synergy `.pkl` files in the specified directory.

    Args:
        directory: Path to the directory containing `.pkl` files.

    Returns:
        List[Tuple]: List of tuples containing (participant, condition, data).
    """
    synergy_data = []

    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory {directory} does not exist.")
    
    pkl_files = [f for f in os.listdir(directory) if f.endswith('.pkl')]

    if not pkl_files:
        print("⚠️ No .pkl files found in the directory.")
        return synergy_data

    for file_name in pkl_files:
        match = re.match(r"synergy_angle_(\w+)_(\w+)_\d+\.\d\.pkl", file_name)
        if match:
            participant, condition = match.groups()
            file_path = os.path.join(directory, file_name)
            with open(file_path, 'rb') as file:
                data = pickle.load(file)
                synergy_data.append((participant, condition, data))
                print(f"✅ Loaded: {file_name}")
    
    print(f"\n✨ {len(synergy_data)} files loaded from {directory}.")
    return synergy_data


def load_synergy_list(
    directory: str, 
    participant_filter: Optional[str] = None,
    signal_type: Literal["angle", "emg"] = "emg",
    verbose: bool = False
) -> pd.DataFrame:
    """
    Loads synergy files and flattens them into a structured Pandas DataFrame.
    This is the primary loader for modern analysis pipelines.
    
    Args:
        directory: Path to the data directory.
        participant_filter: Specific Participant ID to load (optional).
        signal_type: Type of data to extract ('emg' or 'angle').
        verbose: If True, prints status messages during loading.

    Returns:
        pd.DataFrame: Processed synergy data with metadata and components.
    """
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")

    records = []
    pkl_files = [f for f in os.listdir(directory) if f.endswith('.pkl')]

    if not pkl_files:
        if verbose: print("⚠️ No .pkl files found.")
        return pd.DataFrame()

    pattern = {
        "emg": r"synergy_emg_(\w+)_(\w+).pkl",
        "angle": r"synergy_angle_(\w+)_(\w+).pkl"
    }.get(signal_type)

    if pattern is None:
        raise ValueError("signal_type must be 'angle' or 'emg'")

    for file_name in pkl_files:
        match = re.match(pattern, file_name)
        if not match:
            continue

        participant_id, condition = match.groups()
        if participant_filter and participant_id != participant_filter:
            continue

        with open(os.path.join(directory, file_name), 'rb') as file:
            synergy_cycles = pickle.load(file)

        for cycle_data in synergy_cycles:
            cycle_index = cycle_data.get('cycle_index', -1)
            M = np.array(cycle_data['M'])
            P = np.array(cycle_data['P'])
            V = cycle_data.get('V', None)
            Vr = cycle_data.get('Vr', None)

            # Flattening synergies: one row per individual synergy component
            for i in range(M.shape[0]):
                records.append({
                    'participant': participant_id,
                    'condition': condition,
                    'cycle': cycle_index,
                    'spatial_synergy': M[i],
                    'temporal_synergy': P[:, i],
                    'signal_original': V,
                    'signal_reconstructed': Vr
                })

        if verbose: print(f"✅ Loaded: {file_name}")

    df = pd.DataFrame(records)
    if verbose: print(f"\n✨ {len(df)} total synergy rows loaded from {directory}.")
    return df