# -*- coding: utf-8 -*-
"""
Visualization: Cluster Profiling
-------------------------------
Plots the top N dominant clusters for EMG or Kinematic synergies.
Displays mean temporal activations (left) and mean muscle weights (right).

Architecture: src/visualisation/visu_cluster.py
"""

import sys
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- 1. Robust Path Setup ---
# Goes up 2 levels from src/visualisation to reach the Project Root
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

# --- 2. Configuration ---
TYPE_DATA = "EMG"  # "EMG" or "ANGLE"
N_TOP_CLUSTERS = 4 

# Muscle/Angle Names for labeling
NAMES_EMG = ["ST", "RF", "BF", "VL", "GM", "SOL", "PL", "TA", "VM", "Gmax", "GL"]
NAMES_ANGLES = ["HIP-X", "HIP-Y", "HIP-Z", "KNEE-X", "KNEE-Y", "ANK-X", "ANK-Y"]

# Path to the clustered data (located in ROOT/results/...)
FILE_PATH = ROOT_DIR / "results" / "Clustering" / TYPE_DATA / f"clustered_{TYPE_DATA.lower()}.pkl"

def load_clustered_data() -> Optional[pd.DataFrame]:
    """Loads the pickled DataFrame containing clustered synergy data."""
    if not FILE_PATH.exists():
        print(f"❌ Clustering file not found at: {FILE_PATH}")
        print("💡 Make sure to run Clustering.py before visualization.")
        return None
    
    try:
        with open(FILE_PATH, 'rb') as f:
            df = pickle.load(f)
        
        if not isinstance(df, pd.DataFrame):
            print("⚠️ Loaded data is not a DataFrame. Check save format.")
            return None
            
        print(f"✅ Successfully loaded {len(df)} synergy cycles for visualization.")
        return df
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return None

def plot_top_clusters(df: pd.DataFrame, n_top: int = 4):
    """
    Identifies the most frequent clusters and plots their average profiles.
    
    Args:
        df: The clustered synergy DataFrame.
        n_top: Number of most frequent clusters to display.
    """
    if 'cluster_label' not in df.columns:
        print(f"❌ Error: 'cluster_label' column missing. Columns: {df.columns.tolist()}")
        return

    # 1. Identify dominant clusters
    cluster_counts = df['cluster_label'].value_counts()
    top_cluster_ids = cluster_counts.head(n_top).index.tolist()
    
    n_plot = len(top_cluster_ids)
    fig, axes = plt.subplots(n_plot, 2, figsize=(14, 3 * n_plot), squeeze=False)
    
    labels_x = NAMES_EMG if TYPE_DATA == "EMG" else NAMES_ANGLES

    for i, cluster_id in enumerate(top_cluster_ids):
        cluster_data = df[df['cluster_label'] == cluster_id]
        n_members = len(cluster_data)
        
        # --- LEFT: MEAN TEMPORAL ACTIVATION ('temporal_module') ---
        ax_temp = axes[i, 0]
        # Stack 1D arrays to calculate mean/std
        temp_stack = np.stack(cluster_data['temporal_module'].values)
        mean_temp = np.mean(temp_stack, axis=0)
        std_temp = np.std(temp_stack, axis=0)
        x_axis = np.linspace(0, 100, len(mean_temp))
        
        ax_temp.plot(x_axis, mean_temp, color='firebrick', lw=2.5)
        ax_temp.fill_between(x_axis, mean_temp - std_temp, mean_temp + std_temp, 
                             color='firebrick', alpha=0.15)
        ax_temp.set_title(f"Cluster {cluster_id}: Temporal Profile (n={n_members})", fontweight='bold')
        ax_temp.set_xlim(0, 100)
        ax_temp.set_ylabel("Amplitude (A.U.)")
        if i == n_plot - 1:
            ax_temp.set_xlabel("% of Gait Cycle")

        # --- RIGHT: MEAN SPATIAL WEIGHTS ('spatial_module') ---
        ax_spat = axes[i, 1]
        spat_stack = np.stack(cluster_data['spatial_module'].values)
        mean_spat = np.mean(spat_stack, axis=0)
        std_spat = np.std(spat_stack, axis=0)
        
        bars = ax_spat.bar(range(len(mean_spat)), mean_spat, yerr=std_spat, 
                           color='steelblue', alpha=0.8, capsize=4)
        ax_spat.set_xticks(range(len(mean_spat)))
        ax_spat.set_xticklabels(labels_x[:len(mean_spat)], rotation=45, ha="right", fontsize=9)
        ax_spat.set_title(f"Cluster {cluster_id}: Muscle Weights", fontweight='bold')
        ax_spat.set_ylim(0, max(1.1, np.max(mean_spat + std_spat)))

        # Visual cleanup
        for ax in [ax_spat, ax_temp]:
            ax.spines[['top', 'right']].set_visible(False)
            ax.grid(axis='y', linestyle=':', alpha=0.5)

    plt.suptitle(f"DOMINANT {TYPE_DATA} SYNERGY CLUSTERS", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    df_clustered = load_clustered_data()
    if df_clustered is not None:
        plot_top_clusters(df_clustered, n_top=N_TOP_CLUSTERS)