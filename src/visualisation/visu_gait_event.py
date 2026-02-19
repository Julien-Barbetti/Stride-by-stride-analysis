# -*- coding: utf-8 -*-
"""
Visualization: Raw EMG & Gait Events
------------------------------------
Superimposes gait cycle events (IC/TC) over raw EMG signals.
Useful for validating segmentation accuracy and visual inspection of data.

Architecture: src/visualisation/visu_events_raw.py
"""

import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- 1. Robust Path Setup ---
# Goes up 2 levels from src/visualisation to reach Project Root
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

# --- 2. Configuration ---
FS = 2000           # Sampling Frequency (Hz)
PARTICIPANT = "P11"
CONDITION = "0"
MUSCLES_TO_PLOT = ["Q2_ST", "A3_VM"]
NB_CYCLES = 10      # Number of gait cycles to display
PRE_BUFFER = 0.2    # Time (seconds) to show before the first Initial Contact (IC)

# File Paths
EVENT_FILE = ROOT_DIR / "results" / "Events" / f"event_cycle_{PARTICIPANT}_{CONDITION}.csv"
DATA_FILE = ROOT_DIR / "data_example" / "EMG" / f"{PARTICIPANT}_{CONDITION}_emg.csv"

def plot_superimposed_muscles():
    """Plots raw EMG data with vertical lines for gait cycle events."""
    
    # Check file existence
    if not EVENT_FILE.exists():
        print(f"❌ Event file missing: {EVENT_FILE}")
        return
    if not DATA_FILE.exists():
        print(f"❌ Data file missing: {DATA_FILE}")
        return
    
    # --- 3. Data Loading ---
    df_ev = pd.read_csv(EVENT_FILE)
    df_data = pd.read_csv(DATA_FILE)
    
    # Filter for Right Foot events (standardizing for synchronization)
    df_right = df_ev[df_ev['foot'] == 'right'].reset_index(drop=True)
    
    if df_right.empty:
        print(f"⚠️ No 'right' foot events found in {EVENT_FILE.name}.")
        return

    time_axis = np.arange(len(df_data)) / FS
    
    # --- 4. Plotting ---
    plt.figure(figsize=(15, 7))
    
    # Colors for muscle signals
    colors = ['#333333', '#999999', '#D9534F', '#5BC0DE'] 
    
    for i, muscle in enumerate(MUSCLES_TO_PLOT):
        if muscle in df_data.columns:
            color = colors[i % len(colors)]
            plt.plot(time_axis, df_data[muscle], label=f"EMG {muscle}", 
                     color=color, alpha=0.8, lw=1.2)
        else:
            print(f"⚠️ Warning: Muscle '{muscle}' not found in data columns.")

    # --- 5. Event Overlays ---
    # Limit to requested number of cycles to avoid cluttering
    cycles_to_draw = df_right.head(NB_CYCLES)
    
    for i, row in cycles_to_draw.iterrows():
        is_first = (i == 0)
        # IC: Green vertical lines
        plt.axvline(x=row['IC_time'], color='forestgreen', lw=2, alpha=0.8, 
                    label='Initial Contact (R)' if is_first else "")
        # TC: Red dashed vertical lines
        plt.axvline(x=row['TC_time'], color='crimson', lw=2, ls='--', alpha=0.8, 
                    label='Terminal Contact (R)' if is_first else "")
    
    # --- 6. Aesthetic & Dynamic Zoom ---
    plt.title(f"EMG & Event Synchronization - {PARTICIPANT} (Cond {CONDITION})", 
              fontsize=14, fontweight='bold')
    plt.xlabel("Time (s)")
    plt.ylabel("EMG Amplitude (V or µV)")
    
    # Define horizontal limits based on the requested cycles
    t_start = df_right['IC_time'].iloc[0] - PRE_BUFFER
    
    # End limit: last TC of the requested count (or maximum available)
    idx_limit = min(NB_CYCLES - 1, len(df_right) - 1)
    t_end = df_right['TC_time'].iloc[idx_limit] + 0.3 # Added a small margin
    
    plt.xlim(t_start, t_end)

    # Legend and grid
    plt.legend(loc='upper right', frameon=True, shadow=True, fontsize='small')
    plt.grid(True, axis='both', linestyle='--', alpha=0.4)
    plt.tight_layout()
    
    plt.show()

if __name__ == "__main__":
    plot_superimposed_muscles()