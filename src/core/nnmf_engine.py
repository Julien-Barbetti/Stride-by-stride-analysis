# -*- coding: utf-8 -*-
"""
NNMF Engine: Muscle Synergy Extraction and Quality Assessment
------------------------------------------------------------
Core mathematical logic for Non-negative Matrix Factorization (NMF) applied
to biomechanical data (EMG, Kinematics). 

This module provides:
1. Reconstruction quality metrics (Global and Per-Channel VAF).
2. Automated selection of synergy counts based on physiological thresholds.
3. Stable decomposition using multi-start iterative optimization.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict
from sklearn.decomposition import NMF
from tqdm import tqdm

# =============================================================================
# 1. QUALITY METRICS (VAF)
# =============================================================================

def compute_vaf(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """
    Calculates the global Variance Accounted For (VAF).
    VAF = 1 - (SSE / SST), assessing the overall reconstruction quality.
    """
    return 1 - np.sum((original - reconstructed) ** 2) / np.sum(original ** 2)

def compute_vaf_per_channel(original: np.ndarray, reconstructed: np.ndarray) -> np.ndarray:
    """
    Calculates the VAF for each channel (muscle/angle) individually.
    Ensures that no specific signal is overlooked by the global metric.
    """
    numerator = np.sum((original - reconstructed) ** 2, axis=0)
    denominator = np.sum(original ** 2, axis=0)
    return 1 - numerator / denominator

# =============================================================================
# 2. OPTIMAL SYNERGY ESTIMATION
# =============================================================================

def find_optimal_synergy_count(data: pd.DataFrame,
                               threshold: float = 0.85,
                               channel_vaf_threshold: float = 0.7,
                               max_synergies: int = 11,
                               n_iterations: int = 5,
                               init: str = 'nndsvd',
                               solver: str = 'cd',
                               max_iter: int = 3000,
                               tol: float = 1e-4,
                               random_state: int = 1,
                               verbose: bool = False) -> int:
    """
    Iteratively tests different synergy counts (n) to find the minimum number
    required to satisfy both global and local VAF criteria.
    """
    for n in range(2, max_synergies + 1):
        best_vaf_min = -np.inf

        for i in range(n_iterations):
            model = NMF(n_components=n, init=init, solver=solver,
                        max_iter=max_iter, tol=tol, random_state=random_state + i)
            W = model.fit_transform(data)
            
            if model.n_iter_ == max_iter:
                print(f"⚠️(find_optimal_synergy_count) --> Non-convergence for n={n}, iteration {i+1}")
                
            H = model.components_
            Vr = np.dot(W, H)

            vaf_global = compute_vaf(data.values, Vr)
            vaf_per_channel = compute_vaf_per_channel(data.values, Vr)
            min_vaf = vaf_per_channel.min()

            if min_vaf > best_vaf_min:
                best_vaf_min = min_vaf
                best_result = (vaf_global, vaf_per_channel)

        vaf_global, vaf_per_channel = best_result

        if verbose:
            print(f"[n={n}] Global VAF: {vaf_global:.2%}, Min Channel: {vaf_per_channel.min():.2%}")

        # Criteria: Overall reconstruction > threshold AND all muscles > channel_threshold
        if vaf_global >= threshold and np.all(vaf_per_channel >= channel_vaf_threshold):
            return n

    return max_synergies

# =============================================================================
# 3. CORE NMF DECOMPOSITION
# =============================================================================

def nmf_decomposition(data: pd.DataFrame,
                      vaf_target: float = 0.85,
                      n_synergies: Optional[int] = None,
                      init: str = 'nndsvd',
                      max_iter: int = 3000,
                      solver: str = 'cd',
                      tol: float = 1e-4,
                      random_state: int = 1,
                      channel_vaf_threshold: float = 0.70,
                      n_iterations: int = 50) -> Dict[str, pd.DataFrame]:
    """
    Performs the final NMF decomposition (V ≈ W x H).
    Runs multiple times to avoid local minima and returns the best reconstruction.
    """
    if (data < 0).any().any():
        raise ValueError("Data contains negative values (NMF requires values ≥ 0).")

    # Determine optimal number of synergies if not provided
    if n_synergies is None:
        n_synergies = find_optimal_synergy_count(
            data, threshold=vaf_target,
            channel_vaf_threshold=channel_vaf_threshold,
            init=init, solver=solver,
            max_iter=max_iter, tol=tol,
            random_state=random_state
        )

    best_result = None
    best_vaf_global = -np.inf

    # Multi-start NMF to ensure stability
    for i in range(n_iterations):
        model = NMF(n_components=n_synergies, init=init, solver=solver,
                    max_iter=max_iter, tol=tol, random_state=random_state + i)
        W = model.fit_transform(data)
        H = model.components_
        Vr = np.dot(W, H)

        vaf_global = compute_vaf(data.values, Vr)
        vaf_channels = compute_vaf_per_channel(data.values, Vr)

        if vaf_global > best_vaf_global:
            best_vaf_global = vaf_global
            best_result = {
                'P': pd.DataFrame(W), # Temporal Profiles (W)
                'M': pd.DataFrame(H, columns=data.columns), # Spatial Modules (H)
                'V': data, # Original Data
                'Vr': pd.DataFrame(Vr, columns=data.columns), # Reconstructed Data
                'Nb_syns': n_synergies,
                'VAF_global': vaf_global,
                'VAF_channels': vaf_channels
            }

        if model.n_iter_ == max_iter:
            print(f"⚠️(nmf_decomposition) --> Possible non-convergence (n_synergies={n_synergies}, iteration {i+1}).")

    return best_result

# =============================================================================
# 4. BATCH PROCESSING (STRIDE-BY-STRIDE)
# =============================================================================

def nmf_per_cycle(cycle_data_list: List[pd.DataFrame],
                  vaf_target: float = 0.85,
                  channel_vaf_threshold : float = 0.70,
                  n_synergies: int = 4,
                  init: str = 'nndsvd',
                  solver: str = 'cd',
                  max_iter: int = 3000,
                  random_state: int = 1,
                  tol: float = 1e-4,
                  verbose: bool = True) -> List[Dict]:
    """
    Utility function to process multiple gait cycles in a loop.
    Returns a list of dictionaries containing decomposition results for each cycle.
    """
    results = []
    
    pbar = tqdm(enumerate(cycle_data_list), 
                total=len(cycle_data_list), 
                desc="Computing NMF for cycles", 
                disable=not verbose and len(cycle_data_list) < 2)

    for i, data in pbar:
        try:
            if verbose and not pbar.disable:
                pbar.set_postfix({"Cycle": i})

            res = nmf_decomposition(data, 
                                    vaf_target=vaf_target,
                                    channel_vaf_threshold=channel_vaf_threshold,
                                    n_synergies=n_synergies, 
                                    init=init,
                                    max_iter=max_iter, 
                                    solver=solver,
                                    tol=tol, 
                                    random_state=random_state)

            res["cycle_index"] = i
            results.append(res)

        except ValueError as e:
            if verbose:
                print(f"\n⚠️ Error in cycle {i+1} : {e}")

    return results