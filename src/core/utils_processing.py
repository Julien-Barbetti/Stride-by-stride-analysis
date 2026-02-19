# -*- coding: utf-8 -*-
"""
Core Signal Processing Utilities.
---------------------------------
This module provides low-level functions for biomechanical signal processing:
1. Signal Filtering (Butterworth filters, Enveloping).
2. Temporal Segmentation (Cycle extraction based on gait events).
3. Normalization (Time resampling and Amplitude scaling).
4. Interpolation (Frequency matching for multi-sensor systems).

Architecture: src/core/utils_processing.py
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Union, Optional, Literal
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d
from sklearn.preprocessing import MinMaxScaler

# Type Alias
TimeSeriesData = Union[pd.DataFrame, pd.Series, np.ndarray, List]

# ==========================================
# 1. SIGNAL CLEANING & ENVELOPE EXTRACTION
# ==========================================

def signal_preprocessing(
    data: pd.DataFrame, 
    fs: float,
    demean: bool = True, 
    rectif: Optional[Literal["fullwave", "halfwave"]] = "fullwave", 
    f_low: float = 50, 
    f_high: float = 450, 
    band_order: int = 4, 
    lowpass_freq: float = 20, 
    lowpass_order: int = 4, 
    subtract_min: bool = True, 
    normalize: bool = True,
    norm_type: Literal["max", "minmax"] = "max",
    force_positive: bool = False
) -> pd.DataFrame:
    """
    Complete preprocessing pipeline for EMG or Kinematics.
    
    Processing Steps (Standard EMG Envelope):
    Raw -> Demean -> Bandpass (50-450Hz) -> Rectification -> Lowpass (20Hz) -> Normalization.
    """
    processed = data.copy()
    
    # Handle missing values (critical for filtering)
    if processed.isnull().any().any():
        processed = processed.interpolate(method='linear').fillna(0)
        
    # Step A: Centering (Demean)
    if demean:
        processed = processed - processed.mean()
        
    # Step B: Bandpass filtering (removes movement artifacts and high-frequency noise)
    if f_low > 0 and f_high > 0:
        nyq = 0.5 * fs
        wn = [f_low / nyq, f_high / nyq]
        if 0 < wn[0] < 1 and 0 < wn[1] < 1:
            b, a = butter(band_order, wn, btype='bandpass')
            processed = processed.apply(lambda col: filtfilt(b, a, col), axis=0)
            
    # Step C: Rectification (Absolute values)
    if rectif == "fullwave":
        processed = processed.abs()
    elif rectif == "halfwave":
        processed = processed.clip(lower=0)
        
    # Step D: Lowpass filtering (Extraction of the Linear Envelope)
    if lowpass_freq > 0:
        wn = lowpass_freq / (0.5 * fs)
        if 0 < wn < 1:
            b, a = butter(lowpass_order, wn, btype='low')
            processed = processed.apply(lambda col: filtfilt(b, a, col), axis=0)
    
    # 

    # Step E: Post-processing (ensure strictly positive data for NMF)
    if force_positive:
        # Replaces <= 0 values with the smallest positive value found in the data
        min_pos = processed[processed > 0].min().min() if (processed > 0).any().any() else 1e-6
        processed = processed.clip(lower=min_pos)

    if subtract_min:
        processed = processed - processed.min()
    
    # Step F: Amplitude Normalization
    if normalize:
        if norm_type == "max":
            processed = processed / processed.max().replace(0, 1)
        elif norm_type == "minmax":
            scaler = MinMaxScaler()
            processed.iloc[:, :] = scaler.fit_transform(processed.values)
    
    return processed

# ==========================================
# 2. SEGMENTATION & TIME NORMALIZATION
# ==========================================

def extract_signal_subset(
    data: pd.DataFrame, 
    cycles: pd.DataFrame, 
    max_cycles: int, 
    start_cycle: int = 1,
    time_column: Optional[str] = "Time"
) -> Dict[str, pd.DataFrame]:
    """
    Cuts a continuous signal into a specific window of gait cycles.
    """
    if not {'IC', 'TC'}.issubset(cycles.columns):
        raise ValueError("Cycles DataFrame must contain 'IC' (Contact) and 'TC' (Toe-off).")

    end_idx = min(start_cycle + max_cycles - 1, len(cycles))
    selected_cycles = cycles.iloc[start_cycle - 1:end_idx]

    if selected_cycles.empty:
        raise ValueError("Cycle selection resulted in an empty set.")

    t_start = selected_cycles.iloc[0]['IC']
    t_end = selected_cycles.iloc[-1]['TC']

    if time_column in data.columns:
        mask = (data[time_column] >= t_start) & (data[time_column] <= t_end)
        signal = data.loc[mask]
    else:
        signal = data.loc[t_start:t_end]

    return {
        'cycles': selected_cycles.reset_index(drop=True),
        'data': signal.reset_index(drop=True)
    }

def normalize_cycles(
    signal_dict: Dict[str, Any], 
    divisions: List[int], 
    time_column: Optional[str] = None,
    trim: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Time-normalizes each cycle to a fixed number of points.
    FIXED: Uses logical masking for both time and values to ensure equal lengths.
    """
    cycles, data = signal_dict["cycles"], signal_dict["data"]
    
    time = data[time_column].values if (time_column and time_column in data.columns) else data.index.values
    values = data.drop(columns=[time_column]) if (time_column and time_column in data.columns) else data.copy()

    start_i, end_i = (1, len(cycles)-1) if trim else (0, len(cycles))
    all_interp_data = []

    for i in range(start_i, end_i):
        t_ic, t_tc = cycles.loc[i, 'IC'], cycles.loc[i, 'TC']
        t_next_ic = cycles.loc[i + 1, 'IC'] if i + 1 < len(cycles) else time[-1]

        # On boucle sur Stance (Appui) puis Swing (Oscillation)
        for phase_times, n_points in zip([(t_ic, t_tc), (t_tc, t_next_ic)], divisions):
            # --- CORRECTION ICI : On utilise le MÊME masque pour t et v ---
            mask = (time >= phase_times[0]) & (time <= phase_times[1])
            t_phase = time[mask]
            v_phase = values.values[mask] # .values pour numpy, plus rapide

            if len(t_phase) < 2: 
                continue 
            
            # Sécurité supplémentaire : On force la correspondance si un arrondi traîne
            if len(t_phase) != len(v_phase):
                min_len = min(len(t_phase), len(v_phase))
                t_phase = t_phase[:min_len]
                v_phase = v_phase[:min_len]

            interp_time = np.linspace(t_phase[0], t_phase[-1], n_points)
            
            # Interpolation linéaire
            f_interp = interp1d(t_phase, v_phase, axis=0, kind='linear', fill_value="extrapolate")
            
            # On stocke en DataFrame pour garder les noms des colonnes (muscles/angles)
            all_interp_data.append(pd.DataFrame(f_interp(interp_time), columns=values.columns))

    if not all_interp_data:
        return {"cycles": pd.DataFrame(), "data": pd.DataFrame()}

    return {
        "cycles": cycles.iloc[start_i:end_i].reset_index(drop=True),
        "data": pd.concat(all_interp_data, ignore_index=True)
    }

def split_into_cycles(normed_signal_output: Dict[str, pd.DataFrame]) -> List[pd.DataFrame]:
    """Converts a long normalized DataFrame into a list of cycle-by-cycle DataFrames."""
    cycles, data = normed_signal_output['cycles'], normed_signal_output['data']
    if cycles.empty or data.empty: return []
    return np.array_split(data, len(cycles))

# ==========================================
# 3. INTERPOLATION & UTILS
# ==========================================

def interpolate_emg(signal: np.ndarray, fs_target: float, fs_original: float) -> np.ndarray:
    """Resamples EMG signal to match a target frequency (e.g., from 2148Hz to 2000Hz)."""
    n_samples_orig = signal.shape[0]
    duration = n_samples_orig / fs_original
    n_samples_target = int(round(duration * fs_target))

    x_old = np.linspace(0, duration, n_samples_orig)
    x_new = np.linspace(0, duration, n_samples_target)

    f = interp1d(x_old, signal, axis=0, kind='linear', fill_value="extrapolate")
    return f(x_new)

def filter_signal(data: TimeSeriesData, fs: float, filter_type='lowpass', cutoff=None, order=4):
    """Generic Butterworth zero-phase filter (filtfilt)."""
    nyq = 0.5 * fs
    if filter_type == 'bandpass':
        norm_cutoff = [f / nyq for f in cutoff]
    else:
        norm_cutoff = cutoff / nyq
    
    b, a = butter(order, norm_cutoff, btype=filter_type)
    
    if isinstance(data, (pd.DataFrame, pd.Series)):
        return data.apply(lambda x: filtfilt(b, a, x))
    return filtfilt(b, a, data, axis=0)