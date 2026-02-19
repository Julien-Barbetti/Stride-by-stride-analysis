# -*- coding: utf-8 -*-
"""
Synergy Clustering Toolkit
--------------------------
Module for constrained hierarchical clustering of muscle and kinematic synergies.
Ensures biological constraints are respected (no cycle overlap within clusters).

Created on Wed Jul 2 10:11:06 2025
@author: Utilisateur
"""

import os
import re
import time
import heapq
import pickle
import gc
import psutil
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from typing import List, Literal, Tuple, Optional, Dict
from collections import defaultdict, Counter
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error


# =============================== #
#      Data Loading Utilities     #
# =============================== #

def limit_cycles_per_participant_condition(df: pd.DataFrame, max_cycles: int = 10) -> pd.DataFrame:
    """
    Limits the number of cycles per participant and condition to ensure balanced analysis.
    """
    filtered = []
    for (_, _), group in df.groupby(['participant', 'condition']):
        selected = group[group['cycle'].isin(sorted(group['cycle'].unique())[:max_cycles])]
        filtered.append(selected)
    return pd.concat(filtered, ignore_index=True)


def prepare_synergies_for_clustering(df: pd.DataFrame, participant: Optional[str] = None,
                                     fusion: Literal['M', 'P'] = 'M') -> Dict[str, np.ndarray]:
    """
    Prepares data for the clustering process by extracting features and metadata.
    """
    if participant:
        df = df[df['participant'] == participant]

    if fusion == 'M':
        features = df['spatial_synergy'].apply(np.array).tolist()
    elif fusion == 'P':
        features = df['temporal_synergy'].apply(np.array).tolist()
    else:
        raise ValueError("fusion must be 'M' or 'P'")

    return {
        'X': np.array(features),
        'cycle': df['cycle'].values,
        'condition': df['condition'].values,
        'participant': df['participant'].values
    }


# =============================== #
#      Clustering Functions       #
# =============================== #

def constrained_agglomerative_clustering_two_step(
    X: np.ndarray,
    participant_ids: np.ndarray,
    condition_ids: np.ndarray,
    cycle_ids: np.ndarray,
    max_overlap: float = 0.0,
    verbose: bool = False,
    plot_pca: bool = False,
    external_triplet_lists=None
):
    """
    Hierarchical clustering with a constraint preventing exact overlap within clusters.
    Every occurrence of a triplet (Participant, Condition, Cycle) is preserved.
    """
    start_time = time.time()
    X = np.asarray(X, dtype=np.float32)
    n = len(X)

    # Distance matrix calculation
    D = squareform(pdist(X, metric='euclidean').astype(np.float32))
    
    # Initial clusters: one point per cluster
    clusters = [{i} for i in range(n)]
    
    # Triplets for each cluster: tracking occurrences
    triplet_lists = external_triplet_lists or [
        [(participant_ids[i], condition_ids[i], cycle_ids[i])] for i in range(n)
    ]

    # Priority queue for merging
    heap = [(D[i, j], i, j) for i in range(n) for j in range(i+1, n)]
    heapq.heapify(heap)
    active = list(range(n))
    merge_history = []

    while heap:
        dist, i, j = heapq.heappop(heap)
        if active[i] is None or active[j] is None:
            continue

        ci, cj = active[i], active[j]
        ti, tj = triplet_lists[ci], triplet_lists[cj]
        
        # Overlap check using unique triplets
        ti_set, tj_set = set(ti), set(tj)
        overlap_triplets = len(ti_set & tj_set)
        overlap_ci = overlap_triplets / len(ti_set)
        overlap_cj = overlap_triplets / len(tj_set)
        
        # Hard constraint enforcement
        if overlap_ci > max_overlap or overlap_cj > max_overlap:
            continue

        # Valid merge operation
        new_cluster = clusters[ci] | clusters[cj]
        new_triplets = ti + tj

        clusters.append(new_cluster)
        triplet_lists.append(new_triplets)

        new_idx = len(clusters) - 1
        active[i] = active[j] = None
        active.append(new_idx)

        # Update distances using UPGMA logic
        for k, ck in enumerate(clusters[:-1]):
            if active[k] is not None:
                d = np.mean(D[np.ix_(list(new_cluster), list(ck))])
                heapq.heappush(heap, (d, new_idx, k))

        merge_history.append((clusters[ci], clusters[cj], dist))

    # Final label assignment
    final_clusters = [c for idx, c in enumerate(clusters) if idx < len(active) and active[idx] is not None]
    cluster_labels = np.zeros(n, dtype=int)
    for label, cluster in enumerate(final_clusters):
        for idx in cluster:
            cluster_labels[idx] = label

    return {
        'clusters': final_clusters,
        'labels': cluster_labels,
        'merge_history': merge_history,
        'distance_matrix': D,
        'execution_time': time.time() - start_time,
        'participant_ids': participant_ids,
        'condition_ids': condition_ids,
        'cycle_ids': cycle_ids,
        'X': X,
        'triplet_lists': triplet_lists
    }


def two_step_clustering(
    X: np.ndarray,
    participant_ids: np.ndarray,
    condition_ids: np.ndarray,
    cycle_ids: np.ndarray,
    max_overlap: float = 0.0,
    verbose: bool = False,
    plot_pca: bool = False
):
    """
    Two-step clustering pipeline:
    1. Intra-participant clustering to group local synergies.
    2. Global clustering based on participant centroids.
    """
    start_time = time.time()
    X = np.asarray(X, dtype=np.float32)
    n_samples = len(X)

    reduced_X = []
    triplet_lists = []
    cluster_origin_map = []
    new_pid, new_cid = [], []

    # Step 1: Intra-participant clustering
    participant_to_indices = defaultdict(list)
    for i, pid in enumerate(participant_ids):
        participant_to_indices[pid].append(i)

    for pid, indices in participant_to_indices.items():
        X_sub = X[indices]
        cid_sub = np.array(condition_ids)[indices]
        cyc_sub = np.array(cycle_ids)[indices]

        # Signature matched exactly to original
        clustering_result = constrained_agglomerative_clustering_two_step(
            X_sub, [pid]*len(indices), cid_sub, cyc_sub,
            max_overlap=max_overlap, verbose=verbose, plot_pca=False
        )

        clusters_sub = clustering_result['clusters']
        triplet_lists_sub = clustering_result['triplet_lists']

        for cluster, triplets in zip(clusters_sub, triplet_lists_sub):
            global_indices = [indices[i] for i in cluster]
            reduced_X.append(np.mean(X[global_indices], axis=0))
            new_pid.append(pid)
            new_cid.append(cid_sub[list(cluster)[0]])
            cluster_origin_map.append(global_indices)
            triplet_lists.append(triplets)

    reduced_X = np.array(reduced_X)

    # Step 2: Global clustering on centroids
    clustering_result = constrained_agglomerative_clustering_two_step(
        reduced_X, new_pid, new_cid, list(range(len(reduced_X))),
        max_overlap=max_overlap, verbose=verbose, plot_pca=False,
        external_triplet_lists=triplet_lists
    )

    final_clusters = clustering_result['clusters']
    cluster_labels = np.zeros(n_samples, dtype=int)
    for global_label, cluster in enumerate(final_clusters):
        for centro_idx in cluster:
            for original_idx in cluster_origin_map[centro_idx]:
                cluster_labels[original_idx] = global_label

    return {
        'clusters': [set(c) for c in final_clusters],
        'labels': cluster_labels,
        'merge_history': clustering_result['merge_history'],
        'distance_matrix': clustering_result['distance_matrix'],
        'execution_time': time.time() - start_time,
        'participant_ids': participant_ids,
        'condition_ids': condition_ids,
        'cycle_ids': cycle_ids,
        'X': X,
        'triplet_lists': triplet_lists
    }

# =============================== #
#      Validation & Analysis      #
# =============================== #

def plot_pca_participant_condition_subplots(X, participant_ids, condition_ids, cluster_labels):
    """
    PCA Visualization per Participant and Condition.
    """
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    participant_condition_dict = defaultdict(lambda: defaultdict(list))
    for i, (pid, cid) in enumerate(zip(participant_ids, condition_ids)):
        participant_condition_dict[pid][cid].append(i)

    unique_clusters = np.unique(cluster_labels)
    cmap = plt.get_cmap('tab20', len(unique_clusters))
    norm = mcolors.Normalize(vmin=min(unique_clusters), vmax=max(unique_clusters))
    scalar_map = cm.ScalarMappable(norm=norm, cmap=cmap)
    cluster_color_map = {label: scalar_map.to_rgba(label) for label in unique_clusters}

    for participant, condition_dict in participant_condition_dict.items():
        conditions = list(condition_dict.keys())
        n_conditions = len(conditions)
        ncols = min(4, n_conditions)
        nrows = (n_conditions + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False)
        fig.suptitle(f"PCA Analysis: Participant {participant}", fontsize=16)

        for idx, condition in enumerate(conditions):
            row, col = divmod(idx, ncols)
            ax = axes[row][col]
            indices = condition_dict[condition]
            plotted_clusters = set()

            for i in indices:
                label = cluster_labels[i]
                color = cluster_color_map[label]
                if label not in plotted_clusters:
                    ax.scatter(X_pca[i, 0], X_pca[i, 1], color=color, alpha=0.7, label=f"Cluster {label}")
                    plotted_clusters.add(label)
                else:
                    ax.scatter(X_pca[i, 0], X_pca[i, 1], color=color, alpha=0.7)

            ax.set_title(f"Condition {condition}")
            ax.grid(True)
            ax.legend(loc='best', fontsize='small')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()


def detect_intra_cluster_duplicates(result, verbose: bool = False):
    """
    Checks for constraint violations (duplicate cycles within a cluster).
    """
    df = pd.DataFrame({
        "cluster_label": result['labels'],
        "participant": result['participant_ids'],
        "condition": result['condition_ids'],
        "cycle": result['cycle_ids']
    })

    duplicates = (
        df.groupby(["cluster_label", "participant", "condition", "cycle"])
          .size()
          .reset_index(name="count")
          .query("count > 1")
    )

    if duplicates.empty:
        if verbose: print("✅ Constraints respected: No duplicates found.")
    else:
        if verbose: print("⚠️ Constraint violation detected:")
        if verbose: print(duplicates)

    return duplicates


def optimize_cluster_overlap(synergy_df, cluster_label, metric='icc', min_similarity=0.6, verbose=False, plot_pca=False):
    """
    Iteratively relocates synergies to resolve intra-cluster duplicates.
    """
    clusters = {}
    participants = np.array(synergy_df["participant"])
    conditions = np.array(synergy_df["condition"])
    cycles = np.array(synergy_df["cycle"])
    X = np.stack(synergy_df["spatial_synergy"].values)
    
    for idx, cl in enumerate(cluster_label):
        clusters.setdefault(cl, set()).add(idx)

    def similarity(a, b):
        if metric == 'icc':
            mean_a, mean_b = X[a].mean(), X[b].mean()
            ss_total = np.sum((X[a]-mean_a)**2) + np.sum((X[b]-mean_b)**2)
            ss_error = np.sum((X[a]-X[b])**2)
            return 1 - (ss_error / ss_total) if ss_total != 0 else 0
        return -np.mean((X[a]-X[b])**2)

    df = synergy_df.copy()
    df['cluster_label'] = cluster_label
    duplicates = df.groupby(["cluster_label", "participant", "condition", "cycle"]).size().reset_index(name="count").query("count > 1")
    
    for _, row in duplicates.iterrows():
        c_idx = row["cluster_label"]
        triplet = (row["participant"], row["condition"], row["cycle"])
        idxs = [i for i in clusters[c_idx] if (participants[i], conditions[i], cycles[i]) == triplet]
        to_move = idxs[1:]

        for idx in to_move:
            clusters[c_idx].remove(idx)
            best_c, best_sim = None, -np.inf
            for alt_c, members in clusters.items():
                if triplet in {(participants[m], conditions[m], cycles[m]) for m in members}:
                    continue
                sims = [similarity(idx, m) for m in members]
                avg_sim = np.mean(sims) if sims else -np.inf
                if avg_sim > best_sim:
                    best_c, best_sim = alt_c, avg_sim
            
            if best_c is not None and best_sim >= min_similarity:
                clusters[best_c].add(idx)
            else:
                new_c = max(clusters.keys()) + 1
                clusters[new_c] = {idx}

    final_labels = np.zeros(len(X), dtype=int)
    for cl, members in clusters.items():
        for idx in members:
            final_labels[idx] = cl

    if plot_pca:
        plot_pca_participant_condition_subplots(X, participants, conditions, final_labels)

    return final_labels, clusters