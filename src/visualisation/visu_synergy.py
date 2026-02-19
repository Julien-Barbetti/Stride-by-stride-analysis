# -*- coding: utf-8 -*-
"""
Visualization: Synergy Cycle-by-Cycle Comparison
-----------------------------------------------
Plots temporal activations (P) and spatial weights (M) for multiple 
individual cycles to assess NMF extraction consistency.

Architecture: src/visualisation/visu_synergy_comparison.py
"""

import sys
import pickle
from pathlib import Path
from typing import Optional, List

import numpy as np
import matplotlib.pyplot as plt

# --- 1. Robust Path Setup ---
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

# --- 2. Configuration ---
TYPE_DATA = "EMG"  # "EMG" or "ANGLE"
PARTICIPANT = "P11"
CONDITION = "0"
NB_CYCLES_TO_PLOT = 3 

NAMES_EMG = ["ST", "RF", "BF", "VL", "GM", "SOL", "PL", "TA", "VM", "Gmax", "GL"]
NAMES_ANGLES = ["ANK-X", "ANK-Y", "HIP-X", "HIP-Y", "HIP-Z", "KNEE-X", "KNEE-Y", "KNEE-Z"]

# Correct path: results/Synergies/TYPE/file.pkl
FILE_PATH = ROOT_DIR / "results" / "Synergies" / TYPE_DATA / f"synergy_{TYPE_DATA.lower()}_{PARTICIPANT}_{CONDITION}.pkl"

def plot_multi_cycles():
    """Plots synergies side-by-side for the first few cycles of a trial."""
    
    if not FILE_PATH.exists():
        print(f"❌ Synergy file not found: {FILE_PATH}")
        return

    try:
        with open(FILE_PATH, 'rb') as f:
            results = pickle.load(f)
    except Exception as e:
        print(f"❌ Error loading pickle: {e}")
        return

    # Determine number of cycles to plot
    n_plot = min(NB_CYCLES_TO_PLOT, len(results))
    
    # Identify the maximum number of synergies extracted across these cycles
    syn_counts = [res['Nb_syns'] for res in results[:n_plot]]
    max_syn = max(syn_counts)
    
    # Handle Labels (use metadata if available, otherwise fallback to manual lists)
    # We check the first cycle's spatial matrix (M) columns
    try:
        muscle_labels = results[0]['M'].columns.tolist()
        if all(isinstance(c, (int, np.integer)) for c in muscle_labels):
            muscle_labels = NAMES_EMG if TYPE_DATA == "EMG" else NAMES_ANGLES
    except:
        muscle_labels = NAMES_EMG if TYPE_DATA == "EMG" else NAMES_ANGLES

    # Create figure: Rows = Synergies, Columns = (Activation, Weights) x n_plot
    fig, axes = plt.subplots(max_syn, n_plot * 2, figsize=(5 * n_plot, 2.5 * max_syn), squeeze=False)

    for c_idx in range(n_plot):
        cycle_data = results[c_idx]
        P_df = cycle_data['P']  # Temporal profiles
        M_df = cycle_data['M']  # Spatial weights
        vaf = cycle_data.get('VAF_global', 0)
        n_syn_current = cycle_data['Nb_syns']
        x_percent = np.linspace(0, 100, len(P_df))

        for s_idx in range(max_syn):
            col_p = c_idx * 2
            col_m = c_idx * 2 + 1
            ax_p = axes[s_idx, col_p]
            ax_m = axes[s_idx, col_m]

            # If synergy exists for this cycle
            if s_idx < n_syn_current:
                # --- LEFT: Activation (P) ---
                ax_p.plot(x_percent, P_df.iloc[:, s_idx], color='#d32f2f', lw=2)
                ax_p.fill_between(x_percent, P_df.iloc[:, s_idx], color='#d32f2f', alpha=0.1)
                ax_p.set_xlim(0, 100)
                ax_p.set_ylabel(f"Syn {s_idx+1}")
                
                # --- RIGHT: Weights (M) ---
                weights = M_df.iloc[s_idx, :]
                ax_m.bar(range(len(weights)), weights, color='#1976d2', alpha=0.7)
                ax_m.set_ylim(0, 1.1)
                
                # Add muscle labels only on the bottom row
                if s_idx == n_syn_current - 1 or s_idx == max_syn - 1:
                    ax_m.set_xticks(range(len(weights)))
                    ax_m.set_xticklabels(muscle_labels[:len(weights)], rotation=90, fontsize=8)
                else:
                    ax_m.set_xticks([])

                # Add VAF title only on the top row
                if s_idx == 0:
                    ax_p.set_title(f"CYCLE {c_idx+1}\nVAF: {vaf:.1%}", fontsize=11, fontweight='bold')
            else:
                # Empty subplots if this cycle has fewer synergies than max_syn
                ax_p.axis('off')
                ax_m.axis('off')

            # Aesthetic cleanup
            if s_idx < n_syn_current:
                for ax in [ax_p, ax_m]:
                    ax.spines[['top', 'right']].set_visible(False)
                    ax.grid(axis='y', linestyle=':', alpha=0.3)

    plt.suptitle(f"{TYPE_DATA} SYNERGY STABILITY - {PARTICIPANT} (Condition {CONDITION})", 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

if __name__ == "__main__":
    plot_multi_cycles()