# -*- coding: utf-8 -*-
"""
Spatiotemporal Stride Parameter Calculation Module.
--------------------------------------------------
This module calculates key gait parameters for running analysis:
1. Temporal phases: Stance duration, Flight duration, and Cycle time.
2. Duty Factor (DF): Percentage of the cycle spent in contact with the ground.
3. Normalized Stride Length: Stride length adjusted by leg length (dimensionless).

Architecture: src/module/stride_type_module.py
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Literal

# --- Project Imports ---
from src.core.utils_math import compute_3d_distance
from src.core.utils_import import find_running_speed


def running_phase_compute(stance: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the duration of gait phases from contact events.

    Args:
        stance (pd.DataFrame): DataFrame containing 'IC_time' (Initial Contact) 
                               and 'TC_time' (Toe-Off) events.

    Returns:
        pd.DataFrame: Durations of stance, flight, and total cycle.
    """
    data_phase = []
    # Loop through cycles (from one IC to the next IC)
    for cycle in range(len(stance["IC_time"]) - 1):
        # Time from Toe-Off to next Initial Contact
        t_fly = stance["IC_time"].iloc[cycle + 1] - stance["TC_time"].iloc[cycle]
        
        # Time from Initial Contact to Toe-Off
        t_stance = stance["TC_time"].iloc[cycle] - stance["IC_time"].iloc[cycle]
        
        # Total time for one complete stride
        t_cycle = stance["IC_time"].iloc[cycle + 1] - stance["IC_time"].iloc[cycle]
        
        data_phase.append({
            "IC_time": stance["IC_time"].iloc[cycle],
            "T_stance": t_stance,
            "T_fly": t_fly,
            "T_cycle": t_cycle
        })
        
    return pd.DataFrame(data_phase)


def stride_style_compute(
    stance_df: pd.DataFrame, 
    mode: Literal["unipedal_right", "unipedal_left", "bipedal"] = "bipedal", 
    participant: Optional[str] = None, 
    condition: Optional[int] = None, 
    speed_df: Optional[pd.DataFrame] = None, 
    static_df: Optional[pd.DataFrame] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Main function to compute spatiotemporal running parameters.

    Args:
        stance_df (pd.DataFrame): Event data (IC/TC times and foot side).
        mode (str): Analysis mode ('unipedal_right', 'unipedal_left', or 'bipedal').
        participant (str): Participant ID for speed retrieval.
        condition (int): Experimental condition for speed retrieval.
        speed_df (pd.DataFrame): Global speed reference table.
        static_df (pd.DataFrame): Static trial data for leg length calculation.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: The processed dataframe (returned twice 
                                           for compatibility with older scripts).
    """

    def process(df: pd.DataFrame) -> pd.DataFrame:
        # 1. Temporal Phases Calculation
        phase_df = running_phase_compute(df)
        
        # 2. Duty Factor (%)
        # Formula: (Stance Duration / Total Cycle Duration) * 100
        phase_df["DutyFactor"] = (phase_df["T_stance"] / phase_df["T_cycle"]) * 100
        
        # 3. Leg Length Calculation from Static Trial
        # Distance between Greater Trochanter (ICT/Hip) and Lateral Malleolus (FAL/Ankle)
        leg_r = compute_3d_distance(static_df, "R_ICT", "R_FAL")
        leg_l = compute_3d_distance(static_df, "L_ICT", "L_FAL")
        
        # Average leg length converted from mm to meters
        avg_leg_length_m = ((leg_r + leg_l) / 2) / 1000
        
        # 4. Speed Retrieval (m/s)
        v_mps = find_running_speed(participant, condition, speed_df)
        
        # 5. Normalized Stride Length (L_norm)
        # Formula: (Speed * Cycle_Time) / Leg_Length
        phase_df["normalized_stride_length"] = (v_mps * phase_df["T_cycle"]) / avg_leg_length_m

        # Add Metadata
        if participant: phase_df["Participant"] = participant
        if condition is not None: phase_df["Condition"] = condition
        
        # Return organized columns
        cols = ["IC_time", "DutyFactor", "normalized_stride_length", "Participant", "Condition"]
        return phase_df[cols]

    # --- Foot Filtering Logic ---
    if mode == "unipedal_right":
        curr_df = stance_df[stance_df["foot"] == "right"].reset_index(drop=True)
    elif mode == "unipedal_left":
        curr_df = stance_df[stance_df["foot"] == "left"].reset_index(drop=True)
    else:
        # Sort chronologically for bipedal analysis
        curr_df = stance_df.sort_values("IC_time").reset_index(drop=True)

    if curr_df.empty:
        raise ValueError(f"No data found for mode: {mode}. Check your input dataframe.")

    final_result = process(curr_df)
    
    return final_result, final_result