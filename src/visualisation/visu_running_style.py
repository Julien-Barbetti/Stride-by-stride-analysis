# -*- coding: utf-8 -*-
"""
Visualization: Running Style Analysis
------------------------------------
Plots Duty Factor (%) against Normalized Stride Length.
Visualizes individual strides (scatter) and participant centroids (markers).

Architecture: src/visualisation/visu_running_style.py
"""

import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Robust Path Setup ---
# Goes up 2 levels from src/visualisation to reach Project Root
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

# --- 2. Configuration ---
# Path to Running Style CSVs: results/Running_style/
RESULT_DIR = ROOT_DIR / "results" / "Running_style"

def load_all_running_styles() -> Optional[pd.DataFrame]:
    """Scans the result directory and concatenates all stride_type CSVs."""
    all_files = list(RESULT_DIR.glob("stride_type_*.csv"))
    
    if not all_files:
        print(f"❌ No files found in {RESULT_DIR}")
        print("💡 Ensure you have run the running style analysis script first.")
        return None
    
    data_list = []
    for f in all_files:
        try:
            df = pd.read_csv(f)
            # Ensure categorical consistency
            df['Participant'] = df['Participant'].astype(str)
            df['Condition'] = df['Condition'].astype(str)
            data_list.append(df)
        except Exception as e:
            print(f"⚠️ Error loading {f.name}: {e}")
    
    return pd.concat(data_list, ignore_index=True)

def plot_running_style(df: pd.DataFrame):
    """Generates a scatter plot of running styles with participant centroids."""
    plt.figure(figsize=(14, 8))
    
    # --- 3. Global Scatter (Individual Strides) ---
    sns.scatterplot(
        data=df, 
        x='normalized_stride_length', 
        y='DutyFactor', 
        hue='Condition', 
        marker='o',
        alpha=0.2,       # Lower alpha to highlight centroids
        palette='viridis',
        s=30,
        edgecolor=None
    )

    # --- 4. Centroids and Markers Setup ---
    participants = sorted(df['Participant'].unique())
    unique_conditions = sorted(df['Condition'].unique())
    
    # Marker map for participants to distinguish them
    marker_list = ['X', 's', 'D', 'P', '^', 'v', '<', '>']
    part_marker_map = {p: marker_list[i % len(marker_list)] for i, p in enumerate(participants)}
    
    # Color map for conditions
    colors = sns.color_palette('viridis', len(unique_conditions))
    cond_color_map = dict(zip(unique_conditions, colors))

    # --- 5. Centroid Plotting & Annotation ---
    for p in participants:
        for c in unique_conditions:
            subset = df[(df['Participant'] == p) & (df['Condition'] == c)]
            if not subset.empty:
                c_x = subset['normalized_stride_length'].mean()
                c_y = subset['DutyFactor'].mean()
                
                # Plot Large Centroid Symbol
                plt.scatter(
                    c_x, c_y, 
                    s=200, 
                    color=cond_color_map[c],
                    edgecolor='black',
                    linewidth=1.2,
                    marker=part_marker_map[p],
                    zorder=10
                )
                
                # Dynamic text annotation with small offset
                plt.text(
                    c_x + 0.002, 
                    c_y + 0.3, 
                    p, 
                    fontsize=9, 
                    fontweight='bold', 
                    zorder=11
                )

    # --- 6. Plot Aesthetics ---
    plt.title("Running Style: Duty Factor vs Normalized Stride Length", fontsize=15, fontweight='bold')
    plt.xlabel("Normalized Stride Length (V * T_cycle / Leg_Length)", fontsize=12)
    plt.ylabel("Duty Factor (%)", fontsize=12)
    
    # Visual Reference: Running vs Walking threshold (Duty Factor 50%)
    plt.axhline(y=50, color='black', linestyle='--', alpha=0.2, label="Walking Threshold")
    plt.grid(True, linestyle=':', alpha=0.6)

    # Custom Legend Handling
    # We only show Condition colors in the main legend to keep it clean
    handles, labels = plt.gca().get_legend_handles_labels()
    # Logic: take only the first N items matching unique conditions
    plt.legend(handles[:len(unique_conditions)], unique_conditions, 
               title="Slope Conditions", bbox_to_anchor=(1.02, 1), loc='upper left')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    df_all = load_all_running_styles()
    if df_all is not None:
        plot_running_style(df_all)