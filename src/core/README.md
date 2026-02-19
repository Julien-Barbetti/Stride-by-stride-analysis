# CORE ENGINE DOCUMENTATION

This directory contains the fundamental algorithmic building blocks of the toolkit. These functions are designed to be purely computational and are called by the high-level orchestrators in the `src/module/` folder. **Feel free to use these building blocks for new project.**

## 🧠 Main Algorithms

### 1. Synergy Extraction (nnmf_engine.py)
This script utilizes **Non-negative Matrix Factorization (NMF)** to decompose electromyographic or kinematic signals into two primary components:
* **W (Spatial component):** Effectors contribution (which muscles work together).
* **H (Temporal component):** Temporal activation (when they are active).

Batch process of individual cycle and the Variance Accounted For (VAF) are included.

The solved equation is:
$$V \approx W \times H$$

### 2. Synergy Clustering (clustering_engine.py)
This script use **Hierarchical Agglomerative Clustering** to group similar synergies across different participants or conditions.
* **Distance Metrics:** Pearson correlation or Euclidean distance.
* **Domain Constraint:** The engine implements a "Hard Constraint" logic, preventing two synergies from the same cycle from being assigned to the same cluster.



### 3. Signal Processing (utils_processing.py)
Provides essential functions for filtering and normalization:
* Low-pass/High-pass filtering (Butterworth).
* Rectification of EMG signals.
* Time-normalization (0-100% of the gait cycle) using spline interpolation.

---

## 🛠️ Utilities (Utils)

| File | Role |
| :--- | :--- |
| `utils_math.py` | Vector calculus functions, integral calculations, RMSE. |
| `utils_import.py` | Robust management of CSV file loading and DataFrame conversion. |
| `find_peaks.py` | Peak detection algorithm for gait event identification. |

**END OF README**