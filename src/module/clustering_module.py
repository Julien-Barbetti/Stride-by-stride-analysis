# -*- coding: utf-8 -*-
"""
Module: Synergy Clustering Orchestrator
---------------------------------------
Main interface to group individual synergies (EMG or Kinematic) into 
representative clusters across participants and slope conditions.

This module coordinates:
1. Data loading via specific paths.
2. Subject filtering (inclusion/exclusion).
3. Two-step hierarchical clustering.
4. Cluster refinement (overlap optimization).

Architecture: src/module/clustering_module.py
"""

from typing import List, Literal, Tuple, Optional, Dict
import pandas as pd

# --- Internal Imports ---
from src.core.clustering_engine import ( 
    limit_cycles_per_participant_condition,
    prepare_synergies_for_clustering, 
    two_step_clustering,
    optimize_cluster_overlap
)
from src.core.utils_import import load_synergy_list
from config import get_file_paths

def clustering_module(signal_type: Literal["angle", "emg"] = "emg",
                      participant_filter: Optional[str] = None,
                      fusion: Literal["M", "P"] = "M",
                      max_overlap: float = 0.0,
                      doublon: bool = False,
                      plot: bool = False,
                      participants_to_exclude: Optional[set] = None) -> pd.DataFrame:
    """
    Orchestrates the complete clustering pipeline.

    Args:
        signal_type: Type of data to process ("emg" or "angle").
        participant_filter: ID of a specific participant to process (optional).
        fusion: "M" for spatial weight clustering, "P" for temporal profile clustering.
        max_overlap: Tolerance threshold for synergies from the same cycle in a cluster.
        doublon: If True, skips the optimization step and allows cycle duplicates.
        plot: If True, displays PCA visualizations of the clusters.
        participants_to_exclude: Set of participant IDs to ignore.

    Returns:
        pd.DataFrame: A structured DataFrame containing synergies and their cluster labels.
    """
    
    # --- 1. PATH CONFIGURATION ---
    paths = get_file_paths()
    if signal_type == "emg":
        data_path = paths["synergy_emg"]
    elif signal_type == "angle":
        data_path = paths["synergy_angle"]
    else:
        raise ValueError(f"Unknown signal type: {signal_type}")

    # --- 2. DATA ACQUISITION ---
    # Load raw synergy data from pickle files
    synergy_data = load_synergy_list(
        directory=data_path, 
        signal_type=signal_type,
        participant_filter=participant_filter
    ) 
    
    # --- 3. PRE-PROCESSING & FILTERING ---
    if participants_to_exclude is None:
        participants_to_exclude = set()
    
    # Remove subjects marked for exclusion (e.g., due to poor data quality)
    synergy_data = synergy_data[~synergy_data["participant"].isin(participants_to_exclude)]
    
    # Format data for the clustering algorithm (extract X matrix)
    clustering_inputs = prepare_synergies_for_clustering(synergy_data, fusion=fusion)
    
    X = clustering_inputs["X"]
    conditions = clustering_inputs["condition"]
    participants = clustering_inputs["participant"]
    cycles = clustering_inputs["cycle"]
    
    # --- 4. STEP 1: HIERARCHICAL CLUSTERING ---
    # Determine if PCA plot is needed for the initial step
    show_step1_plot = True if (plot and (doublon or max_overlap == 0)) else False

    # Perform the two-step clustering (Intra-subject then Global)
    result = two_step_clustering(
        X=X, 
        participant_ids=participants,
        condition_ids=conditions,
        cycle_ids=cycles,
        plot_pca=show_step1_plot,
        max_overlap=max_overlap
    )
  
    # --- 5. STEP 2: CLUSTER REFINEMENT (OVERLAP OPTIMIZATION) ---
    # If duplicates within a cycle are forbidden, relocate them to the next best cluster
    if not doublon and max_overlap > 0:
        show_step2_plot = plot
        cluster_labels, _ = optimize_cluster_overlap(
            synergy_data, 
            result["labels"], 
            plot_pca=show_step2_plot
        )
        result["labels"] = cluster_labels
    
    # --- 6. OUTPUT GENERATION ---
    # Merge cluster labels back with metadata for downstream analysis
    output_df = pd.DataFrame({
        "spatial_module": list(synergy_data["spatial_synergy"]),
        "temporal_module": list(synergy_data["temporal_synergy"]),
        "signal_original": list(synergy_data["signal_original"]),
        "signal_reconstructed": list(synergy_data["signal_reconstructed"]),
        "participant": synergy_data["participant"].values,
        "condition": synergy_data["condition"].values,
        "cycle": synergy_data["cycle"].values,
        "cluster_label": result["labels"]
    })
    
    return output_df