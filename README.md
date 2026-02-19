# STRIDE-BY-STRIDE GAIT ANALYSIS TOOLKIT

**Author:** Julien Barbetti  
**Context:** Biomechanics Research  
**Version:** 1.1.0  
**Date:** 2025-2026  

---

## 1. PROJECT OVERVIEW

This repository provides a modular pipeline for the **stride-by-stride analysis of human locomotion**. It handles everything from raw signal processing to muscle/kinematic synergy extraction and hierarchical clustering.

### Key scientific features:
- EMG & Kinematic signal processing  
- Automatic Gait Event detection (HS, TO)  
- Stride Type classification (Van Oevoren 2024)  
- Synergy extraction via NMF & Constrained Clustering  

---

## 2. DIRECTORY STRUCTURE

```text
E:.
│   config.py                 # Global paths and parameter settings
├── data_example              # Raw CSV data (EMG, Angle, MoCap, Speed)
├── results                   # Processed outputs (.pkl and .csv)
│   ├── Clustering            # Final clustered synergies
│   ├── Cycles                # Segmented data per gait cycle
│   ├── Events                # Detected stance/swing events
│   ├── Running_style         # Stride Type results (Van Oevoren 2024)
│   └── Synergies             # Extracted NMF modules (M and P)
├── src                       # Source code
│   ├── core                  # Internal engines (Math, NMF, Clustering)
│   ├── module                # High-level task orchestrators
│   ├── script                # Executable analysis scripts
│   └── visualisation         # Plotting tools for analysis
```

---

## 3. EXECUTION PIPELINE (STEP-BY-STEP)

To perform a complete analysis, run the scripts in this specific order:

### Step 1: Event Detection
```bash
python src/script/Stance_event.py
```
Identifies gait cycles and timing events (Heel Strike, Toe Off).

### Step 2: Stride Type Analysis
```bash
python src/script/Stride_type.py
```
Classifies the running style (e.g., Duty Factor × Stride Length).

### Step 3: Signal Processing
```bash
python src/script/Process_cycle.py
```
Filters and segments raw data into individual strides.

### Step 4: Synergy Extraction
```bash
python src/script/Synergy.py
```
Performs NMF to decompose signals into Spatial/Temporal modules.

### Step 5: Clustering
```bash
python src/script/Clustering.py
```
Groups synergies into global clusters across the dataset.

---

## 4. VISUALIZATION TOOLS

Scripts located in `src/visualisation/`:

- `visu_running_style.py` → Plot running style (e.g., Duty Factor × Stride Length)  
- `visu_synergy.py` → View extracted synergies (muscle or joint)  
- `visu_cluster.py` → View mean synergy of selected cluster  
- `visu_gait_event.py` → Check cycle detection accuracy  

---

## 5. CORE METHODOLOGY

- **Stride Classification:** Based on kinematic markers to distinguish running style  
- **Signal Processing:** Butterworth filtering, RMS, and time-normalization  
- **Decomposition:** NMF (Non-negative Matrix Factorization)  
- **Clustering:** Hierarchical Agglomerative Clustering with a hard cycle-overlap constraint  

## 6. FUNCTIONAL WORKFLOW

```mermaid
flowchart TD
classDef data_dir fill:#f9f9f9,stroke:#333
classDef results_dir fill:#fff4dd,stroke:#d4a017
classDef script_dir fill:#e1f5fe,stroke:#01579b
classDef visu_dir fill:#f3e5f5,stroke:#7b1fa2

subgraph DATA [data_example/]
A1[EMG]:::data_dir
A2[Angle]:::data_dir
A3[MoCap]:::data_dir
A4[Vitesse]:::data_dir
end

A3 --> B[script/Stance_event.py]:::script_dir
B --> B_out[results/Events]:::results_dir

A4 --> C[script/Stride_type.py]:::script_dir
B_out --> C
C --> C_out[results/Running_style]:::results_dir

A1 & A2 --> D[script/Process_cycle.py]:::script_dir
B_out --> D
D --> D_out[results/Cycles]:::results_dir

D_out --> E[script/Synergy.py]:::script_dir
E --> E_out[results/Synergies]:::results_dir

E_out --> F[script/Clustering.py]:::script_dir
F --> F_out[results/Clustering]:::results_dir

F_out & E_out & D_out & B_out & C_out --> G[src/visualisation/]:::visu_dir
```

**END OF README**
