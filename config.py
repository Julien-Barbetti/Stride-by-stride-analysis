from pathlib import Path

def load_parameters():
    """Fixed scientific parameters for example data."""
    return {
        "stride_start": 1,
        "stride_count": 5 + 2, # +2 to have 5 because we cut the first and last cycle
        "fs_cam": 300,
        "fs_emg": 2000,
        "vaf_target": 0.90
    }

def get_file_paths(try_id=None):
    """
    Manages file paths relative to the script location.
    """
    # Locate project root (where config.py is situated)
    BASE_DIR = Path(__file__).resolve().parent
    
    # Example data folder
    DATA_ROOT = BASE_DIR / "data_example"
    
    # Output folder to keep input data clean
    OUTPUT_ROOT = BASE_DIR / "results"

    paths = {
        # --- Inputs (Clean CSVs) ---
        "emg_df": DATA_ROOT / "EMG" / f"{try_id}_emg.csv",
        "dot_df": DATA_ROOT / "MoCap" / f"{try_id}_dot_raw.csv",
        "angle_df": DATA_ROOT / "Angle" / f"{try_id}_angle.csv",
        "speed_df": DATA_ROOT / "Vitesse" / "vitesse_participants.csv",
        "static_df": DATA_ROOT / "MoCap" / f"{try_id}_static_raw.csv",

        # --- Intermediate Outputs (Events, Cycles) ---
        "running_event": OUTPUT_ROOT / "Events",
        "running_style": OUTPUT_ROOT / "Running_style",
        "emg_cycle_data": OUTPUT_ROOT / "Cycles" / "EMG",
        "angle_cycle_data": OUTPUT_ROOT / "Cycles" / "Angle",

        # --- Final Outputs (Synergies, Clustering) ---
        "synergy_emg": OUTPUT_ROOT / "Synergies" / "EMG",
        "synergy_angle": OUTPUT_ROOT / "Synergies" / "Angle",
        "clustered_emg": OUTPUT_ROOT / "Clustering" / "EMG",
        "clustered_angle": OUTPUT_ROOT / "Clustering" / "Angle"
    }

    # Automatically create output directories if they don't exist
    for key, path in paths.items():
        if "df" not in key:  # If it is a directory (not a file)
            Path(path).mkdir(parents=True, exist_ok=True)

    return paths

# --- DISPLAY FLAGS ---
VERBOSE = True    # If False, hides minor details
SHOW_PLOTS = False # If False, disables plt.show()
WARNING = False    # If False, silences non-critical warnings