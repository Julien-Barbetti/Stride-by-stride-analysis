# -*- coding: utf-8 -*-
"""
Module: Gait Cycle Analysis
Functionality: IC/TC detection, error handling, and visualization for running/cycling.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from typing import Literal, List, Tuple, Optional, Dict
import tkinter as tk
from tkinter import messagebox

from config import get_file_paths
from src.core.utils_processing import filter_signal
from src.core.find_peaks import trouve_pics

# === UTILS ===

def compute_derivatives(signal, dt):
    """Calculates velocity, acceleration, and jerk from a position signal."""
    velocity = np.gradient(signal, dt)
    acceleration = np.gradient(velocity, dt)
    jerk = np.gradient(acceleration, dt)
    return velocity, acceleration, jerk

def compute_prominence(jerk, threshold: float = 0.2):
    """Calculates the prominence for peak detection based on jerk amplitude."""
    return threshold * (np.max(jerk) + np.abs(np.min(jerk)))

def estimate_step_frequency(signal: np.array, fs: float, fmin: float = 0.5, fmax: float = 4.0):
    """Estimates the dominant step frequency using Fast Fourier Transform (FFT)."""
    n = len(signal)
    freqs = np.fft.rfftfreq(n, d=1/fs)
    fft_magnitude = np.abs(np.fft.rfft(signal - np.mean(signal)))

    # Filter for the relevant human gait frequency range
    valid = (freqs >= fmin) & (freqs <= fmax)
    freqs = freqs[valid]
    fft_magnitude = fft_magnitude[valid]

    if len(freqs) == 0:
        raise ValueError("No frequency components found in the specified range.")

    dominant_freq = freqs[np.argmax(fft_magnitude)]
    return dominant_freq

def threshold_verification(threshold: float, peak_count: int, duration: float, step_freq: float = 1.3, margin: float = 0.02, increment: float = 0.05):
    """Adjusts the detection threshold if the detected peak frequency deviates from the estimated step frequency."""
    verified = False
    detected_freq = peak_count / duration
    
    if detected_freq > step_freq + margin:
        threshold = threshold + increment
        print(f"\033[91mStep frequency too high. Increasing threshold to: {threshold:.2f}\033[0m")
    elif detected_freq < step_freq - margin:
        threshold = threshold - increment
        print(f"\033[91mStep frequency too low. Decreasing threshold to: {threshold:.2f}\033[0m")
    else:
        verified = True
    return verified, threshold

def is_interval_valid(interval: int, expected_interval: float, tolerance: float = 0.3) -> bool:
    """Checks if the temporal interval between two peaks is within biological tolerance."""
    return abs(interval - expected_interval) / expected_interval <= tolerance

def has_bad_intervals(peaks: List[int], expected_interval: float, tolerance: float = 0.3) -> bool:
    """Scans peak list for timing inconsistencies."""
    for i in range(1, len(peaks)):
        interval = peaks[i] - peaks[i - 1]
        if not is_interval_valid(interval, expected_interval, tolerance):
            return True
    return False

def refine_peaks(jerk_signal, peaks, fs, step_freq, tolerance=0.3, exclusion_zone=10):
    """Refines detected peaks by removing duplicates or finding missing intermediate peaks based on expected timing."""
    from scipy.signal import find_peaks

    expected_interval = int(fs / step_freq)
    peak_indices = sorted(set(peaks['index'].tolist()))
    if not peak_indices:
        return peaks.copy()

    refined_indices = [peak_indices[0]]
    already_inserted = set(refined_indices)

    for i in range(1, len(peak_indices)):
        prev_idx = refined_indices[-1]
        curr_idx = peak_indices[i]
        interval = curr_idx - prev_idx

        if interval < (1 - tolerance) * expected_interval:
            # Too close: keep the peak with higher jerk amplitude
            if jerk_signal[curr_idx] > jerk_signal[prev_idx]:
                refined_indices[-1] = curr_idx
                already_inserted.add(curr_idx)
        elif interval > (1 + tolerance) * expected_interval:
            # Too far: search for a missing peak in the gap
            target = prev_idx + expected_interval
            search_width = int(0.2 * interval)
            start = max(prev_idx + exclusion_zone, target - search_width)
            end = min(curr_idx - exclusion_zone, target + search_width)

            if end > start:
                candidate_zone = jerk_signal[start:end]
                local_peaks, _ = find_peaks(candidate_zone)
                for offset in sorted(local_peaks, key=lambda i: -candidate_zone[i]):
                    new_idx = start + offset
                    if new_idx in already_inserted: continue
                    if is_interval_valid(new_idx - prev_idx, expected_interval, tolerance):
                        next_interval = curr_idx - new_idx
                        if next_interval >= (1 - tolerance) * expected_interval:
                            refined_indices.append(new_idx)
                            already_inserted.add(new_idx)
                            break
            if curr_idx not in already_inserted:
                refined_indices.append(curr_idx)
                already_inserted.add(curr_idx)
        else:
            refined_indices.append(curr_idx)
            already_inserted.add(curr_idx)

    refined_indices = sorted(already_inserted)
    return pd.DataFrame({'index': refined_indices, 'value': jerk_signal[refined_indices]})

# === IC / TC DETECTION ===

def detect_running_stance(toe_signal: np.array, fs: float, foot: Literal["left", "right"], tolerance: int = 10, threshold: float = 0.3) -> pd.DataFrame:
    dt = 1 / fs
    duration_total = len(toe_signal) / fs
    # On calcule les dérivées sur le signal COMPLET une seule fois
    _, _, jerk_full = compute_derivatives(toe_signal, dt)
    
    try:
        step_freq = estimate_step_frequency(toe_signal, fs)
        print(f"✅ Estimated step frequency: {step_freq:.2f} Hz")
    except ValueError:
        step_freq = 1.3
        print("⚠️ Estimation failed, using default frequency (1.3 Hz).")

    min_distance = int(fs * (1 / step_freq) * 0.7)
    max_distance = int(fs * (1 / step_freq) * 1.6)

    root = tk.Tk()
    root.withdraw()

    search_start = 0  # Index à partir duquel on cherche les pics

    while True:
        # On travaille sur la portion de signal restante sans modifier l'original
        current_jerk = jerk_full[search_start:]
        current_duration = len(current_jerk) / fs
        
        verified = False
        current_threshold = threshold
        
        while not verified:
            prom = compute_prominence(current_jerk, current_threshold)
            raw_peaks = trouve_pics(current_jerk, tolerance=tolerance,
                                    right_prominence=[prom, None],
                                    left_prominence=[prom, None],
                                    distance=[min_distance, max_distance])
    
            nb_peaks = len(raw_peaks['index'])
            verified, current_threshold = threshold_verification(current_threshold, nb_peaks, current_duration, step_freq=step_freq, margin=0.1)
    
            if not verified: continue
            
            refined_peaks = refine_peaks(jerk_signal=current_jerk, peaks=raw_peaks, fs=fs, step_freq=step_freq)
            # RECALAGE : On ajoute search_start pour retrouver l'index ABSOLU
            IC_index = sorted([idx + search_start for idx in refined_peaks["index"].tolist()])
    
            expected_interval = fs / step_freq
            if has_bad_intervals(IC_index, expected_interval, tolerance=0.3):
                # Si besoin, on pourrait reaffiner ici, mais attention à garder les index absolus
                pass
            
            verified = True 

# --- Vérification visuelle avec index ABSOLUS et Position superposée ---
        first_peak = IC_index[0]
        fig = plt.figure(figsize=(10, 5))
        
        # Normalisation Z-score pour la superposition (moyenne 0, écart-type 1)
        jerk_norm = (jerk_full - np.mean(jerk_full)) / np.std(jerk_full)
        pos_norm = (toe_signal - np.mean(toe_signal)) / np.std(toe_signal)
        
        # Tracé des signaux
        plt.plot(pos_norm, "--", color="gray", label="Position (Z-norm)", alpha=0.5)
        plt.plot(jerk_norm, label="Jerk (Z-norm)", color="#1f77b4", linewidth=1.5)
        
        # Marqueur du pic détecté
        plt.plot(first_peak, jerk_norm[first_peak], 'ro', markersize=10, label="First Detected IC")
        
        # Ligne verticale pour bien voir la synchronisation
        plt.axvline(first_peak, color='red', linestyle=':', alpha=0.6)

        plt.title(f"Verify First Initial Contact - {foot.capitalize()} Foot\n(Index: {first_peak})", fontsize=12, fontweight='bold')
        plt.xlabel("Samples")
        plt.ylabel("Normalized Amplitude")
        
        # Zoom sur la zone d'intérêt (1 seconde autour du pic)
        delta = int(fs * 1.0)
        plt.xlim(max(0, first_peak - delta // 2), min(len(jerk_full), first_peak + delta // 2))
        
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.show(block=False)
        plt.pause(0.1)

        response = messagebox.askyesno("Confirmation", 
                                      f"Is the red dot the correct first IC?\n\n"
                                      f"It should match the start of the impact shock on the jerk signal.")
        plt.close(fig)

        if response:
            break # On a trouvé le bon départ
        else:
            print(f"⚠️ Skipping peak at {first_peak}. Recalculating from next sample...")
            search_start = first_peak + 1 # On décale le point de départ de la recherche

    # --- TC Detection (Terminal Contact) ---
    TC_index = []
    for i in range(len(IC_index) - 1):
        start, end = IC_index[i], IC_index[i + 1]
        center = (start + end) // 2
        window = int(0.2 * (end - start))
        s_start, s_end = max(start, center - window), min(end, center + window)
        if s_end > s_start:
            # On cherche dans jerk_full avec les index absolus
            local_max = np.argmax(jerk_full[s_start:s_end]) + s_start
            TC_index.append(local_max)

    return pd.DataFrame({
        "cycle": np.arange(1, len(TC_index) + 1),
        "foot": foot,
        "IC_time": np.array(IC_index[:-1]) / fs, # Le temps est maintenant correct !
        "TC_time": np.array(TC_index) / fs
    })

# === ERROR HANDLING ===

def check_errors(df_both: pd.DataFrame, sport: Literal["running", "cycling"]) -> list:
    """Validates the gait events for logic errors (missing data, timing order, or limb alternation)."""
    errors = []
    for i, row in df_both.iterrows():
        ic_time, tc_time = row.get("IC_time"), row.get("TC_time")

        if sport == "running":
            if pd.isna(ic_time) or pd.isna(tc_time):
                errors.append((row, "missing_data"))
                continue
            if ic_time > tc_time:
                errors.append((row, "ic_after_tc"))

        if sport in ["running", "cycling"] and i > 0:
            if df_both.iloc[i]["foot"] == df_both.iloc[i - 1]["foot"]:
                errors.append((row, "alternation_error"))
    return errors

def plot_error(row, jerk_R, jerk_L, df_both, fs, window_s=1.5, error_type="", threshold=None):
    """Visualizes specific detection errors for manual review."""
    t = np.arange(len(jerk_R)) / fs
    idx = int(row["IC_time"] * fs)
    w = int(window_s * fs)
    start, end = max(0, idx - w), min(len(t), idx + w)
    
    plt.figure(figsize=(10, 4))
    plt.plot(t[start:end], jerk_R[start:end], color="black", label="Right Jerk", linewidth=1.2)
    plt.plot(t[start:end], jerk_L[start:end], color="dimgray", label="Left Jerk", linewidth=1.2)

    if threshold is not None:
        color = "black" if row["foot"] == "right" else "dimgray"
        plt.axhline(threshold, color=color, linestyle='--', linewidth=1)

    plt.axvline(row["IC_time"], linestyle="--", color="red", alpha=0.5, label="Error Source")
    plt.title(f"Error: {error_type} — Cycle {row['cycle']} — {row['foot']}")
    plt.xlabel("Time (s)")
    plt.ylabel("Jerk (m/s³)")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.show()

def detect_stance_cut_errors(df_both: pd.DataFrame, dot_filt: pd.DataFrame, sport: Literal["running", "cycling"], fs: float, right_foot : str, left_foot : str, window_s: float = 1.5):
    """Pipeline to verify events and plot detected errors."""
    dt = 1 / fs
    jerk_R = compute_derivatives(dot_filt[right_foot], dt)[-1]
    jerk_L = compute_derivatives(dot_filt[left_foot], dt)[-1]
    errors = check_errors(df_both, sport=sport)

    if not errors:
        print("✅ No errors detected in gait events.")
    else:
        for row, error_type in errors:
            print(f"❌ [Cycle {row['cycle']} - {row['foot']}] Error detected: {error_type}")
            plot_error(row, jerk_R, jerk_L, df_both, fs, window_s, error_type)

# === MAIN PIPELINE ===

def run_pipeline(subject_id: str, sport: Literal["running", "cycling"], fs: int = 300, filter_cutoff: int = 20):
    """Executes the full event detection pipeline for a specific subject."""
    paths = get_file_paths(subject_id)
    dot_df = pd.read_csv(paths["dot_df"])
    dot_filt = filter_signal(dot_df, fs=fs, filter_type="lowpass", cutoff=filter_cutoff)
    
    # Specific signal columns (Vertical trajectory of toe/foot markers)
    df_R = detect_running_stance(dot_filt["R_BP Z"], fs=fs, foot="right")
    df_L = detect_running_stance(dot_filt["L_BP Z"], fs=fs, foot="left")

    df_both = pd.concat([df_R, df_L], ignore_index=True)
    df_both.sort_values(by="IC_time", inplace=True)
    df_both.reset_index(drop=True, inplace=True)

    detect_stance_cut_errors(df_both, dot_filt, sport, fs, right_foot="R_BP Z", left_foot="L_BP Z")
    return df_both