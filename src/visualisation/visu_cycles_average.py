# -*- coding: utf-8 -*-
"""
Visualization: Cycle Profiling
-----------------------------
Plots the average profiles (Mean +/- SD) of EMG or Kinematic signals for a 
specific participant and condition.

Architecture: src/visualisation/visu_cycle.py
"""

import sys
import pickle
from pathlib import Path
from typing import Optional, List

import numpy as np
import matplotlib.pyplot as plt

# --- 1. Robust Path Setup ---
# Goes up 2 levels from src/visualisation to reach Project Root
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

# --- 2. Configuration ---
TYPE_DATA = "EMG"  # "EMG" or "ANGLE"
PARTICIPANT = "P12"
CONDITION = "7"

# Muscle/Angle Names for labeling
NAMES_EMG = ["ST", "RF", "BF", "VL", "GM", "SOL", "PL", "TA", "VM", "Gmax", "GL"]
NAMES_ANGLES = ["ANK-X", "ANK-Y", "HIP-X", "HIP-Y", "HIP-Z", "KNEE-X", "KNEE-Y", "KNEE-Z"]

# Correct path to Cycle data: results/Cycles/TYPE/file.pkl
FILE_PATH = ROOT_DIR / "results" / "Cycles" / TYPE_DATA / f"{TYPE_DATA.lower()}_cycle_{PARTICIPANT}_{CONDITION}.pkl"

def plot_cycles_average():
    """Loads cycle data and plots mean +/- SD for each channel."""
    
    if not FILE_PATH.exists():
        print(f"❌ File not found: {FILE_PATH}")
        print(f"💡 Check if the 'results/Cycles/{TYPE_DATA}' folder contains the file.")
        return
    
    try:
        with open(FILE_PATH, 'rb') as f:
            content = pickle.load(f)
    except Exception as e:
        print(f"❌ Error loading pickle: {e}")
        return
    
    # --- 3. Data Extraction ---
    # Handles both dict format {'data': [...]} or raw list format
    if isinstance(content, dict) and 'data' in content:
        raw_data = content['data']
    else:
        raw_data = content

    # Convert to numpy matrix: (n_cycles, time_points, n_channels)
    try:
        all_cycles = np.array([c['V'] if isinstance(c, dict) else c for c in raw_data])
    except Exception as e:
        print(f"❌ Error formatting data matrix: {e}")
        return

    n_channels = all_cycles.shape[2]
    
    # Select appropriate labels
    labels = NAMES_EMG if TYPE_DATA.upper() == "EMG" else NAMES_ANGLES

    # --- 4. Statistics ---
    mean_profile = np.mean(all_cycles, axis=0)
    std_profile = np.std(all_cycles, axis=0)
    x_percent = np.linspace(0, 100, mean_profile.shape[0])

    # --- 5. Plotting ---
    n_cols = 3 if TYPE_DATA == "EMG" else 2
    n_rows = (n_channels + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3.5 * n_rows), sharex=True)
    
    # Flatten axes array for easy iteration
    axes_flat = axes.flatten() if n_channels > 1 else [axes]

    for i in range(n_channels):
        ax = axes_flat[i]
        
        # Plot Mean (Dark Grey)
        ax.plot(x_percent, mean_profile[:, i], color='#333333', lw=2, label='Mean')
        
        # Plot SD (Shadow)
        ax.fill_between(x_percent, 
                         mean_profile[:, i] - std_profile[:, i], 
                         mean_profile[:, i] + std_profile[:, i], 
                         color='#333333', alpha=0.15)
        
        # Labeling
        name = labels[i] if i < len(labels) else f"Ch {i+1}"
        ax.set_title(name, fontsize=12, fontweight='bold', color='#2c3e50')
        
        # Aesthetics
        ax.set_xlim(0, 100)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.spines[['top', 'right']].set_visible(False)
        
        # Axis labels logic
        if i % n_cols == 0:
            unit = "Amplitude (mV)" if TYPE_DATA == "EMG" else "Angle (°)"
            ax.set_ylabel(unit)
        if i >= n_channels - n_cols:
            ax.set_xlabel("% of Gait Cycle")

    # Remove unused subplots
    for j in range(n_channels, len(axes_flat)):
        fig.delaxes(axes_flat[j])

    plt.suptitle(f"AVERAGE {TYPE_DATA} PROFILE - {PARTICIPANT} (Condition {CONDITION})", 
                 fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    # Optional: Save the figure
    # save_path = ROOT_DIR / "results" / "Figures" / f"average_{TYPE_DATA.lower()}_{PARTICIPANT}_{CONDITION}.png"
    # plt.savefig(save_path)
    
    plt.show()

if __name__ == "__main__":
    plot_cycles_average()