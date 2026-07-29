# Comprehensive Execution Guide & Code Flow Documentation

This document provides a step-by-step execution guide for the **Adaptive Multi-Feature Linear Fuzzy Information Granulation (LFIG)** time series classification project. It details **which file to run first**, **what happens during execution**, **how to run the Jupyter/Colab notebook**, and the **exact end-to-end data and code flow**.

---

## 1. Quick Execution Summary Table

| Execution Mode | File to Run | Recommended Command / Platform | Purpose & Expected Output |
| :--- | :--- | :--- | :--- |
| **Setup** | `requirements.txt` | `pip install -r requirements.txt` | Installs core dependencies (`aeon`, `ruptures`, `fastdtw`, `scikit-learn`, `matplotlib`, `seaborn`, `scipy`, `tqdm`). |
| **Option A (Notebook)** | `LFIG_Adaptive_Pipeline_Colab.ipynb` | Google Colab or Local Jupyter Notebook | **Self-contained pipeline.** Runs 4 diagnostic proofs (CPD bounds, 3D vs 10D table, LOFO ablation, Baselines) + 5-fold nested CV. |
| **Option B (Python Script)** | `run_colab_standalone.py` | `python run_colab_standalone.py` | **Standalone Python runner.** Evaluates UCR datasets locally and prints proof tables + nested CV metrics directly to console. |
| **Option C (Module Benchmarking)** | `src/evaluation/benchmark.py` | `python -c "from src.evaluation.benchmark import run_full_benchmark; run_full_benchmark()"` | **Full multi-dataset benchmark.** Evaluates all 23 UCR catalog datasets, runs baselines, and generates `plots/cd_diagram.png`. |
| **Option D (Feature Redundancy)** | `src/features/redundancy.py` | `python -c "from src.features.redundancy import run_redundancy_analysis; run_redundancy_analysis('GunPoint')"` | **Feature analysis.** Generates Pearson correlation heatmaps, PCA variance curves, and VIF tables. |

---

## 2. Option A: Running the Google Colab / Jupyter Notebook

### File to Run: `LFIG_Adaptive_Pipeline_Colab.ipynb`

This notebook is **100% self-contained** and designed to run in Google Colab (CPU or T4 GPU) or any local Jupyter Lab/Notebook environment.

### Step-by-Step Notebook Procedure:

1. **Open the Notebook:**
   - Upload `LFIG_Adaptive_Pipeline_Colab.ipynb` to [Google Colab](https://colab.research.google.com/) or open it in VS Code / Jupyter Lab.

2. **Run Step 1: Install Dependencies & Check Environment:**
   - Automatically checks and installs the required packages (`aeon`, `ruptures`, `fastdtw`, `scikit-learn`, `joblib`, `tqdm`).
   - **What Happens:** Satisfies all requirements needed to load datasets and run granular calculations.

3. **Run Steps 2 to 5: Import Packages, Core Pipeline Modules, Hybrid Similarity, and Strategy Selector:**
   - Contains all modular functions: segmentation algorithms (`fixed_segmentation`, `cpd_segmentation`), 10D granule feature extractor, pairwise distance functions, fusion weight learning, KNN & Precomputed Kernel SVM classifiers, and nested CV helpers.
   - **What Happens:** Loads the core pipeline logic, dynamic over-subscription guard, and caching engines into the runtime memory.

4. **Run Step 6: Full Benchmark & Diagnostic Proofs Demonstration:**
   - Automatically downloads datasets dynamically via `aeon.datasets.load_classification` and loops over all **23 UCR datasets** (starting with `GunPoint`, `Coffee`, `ArrowHead`, etc.).
   - **What Happens:** For each dataset, it executes:
     - **[Proof 1]** Granule Boundary Breakdown table (demonstrating CPD vs Fixed segmentation).
     - **[Proof 2]** 3D Standard vs. 10D Proposed LFIG performance comparison table.
     - **[Proof 3]** Leave-One-Feature-Out (LOFO) ablation impact table.
     - **[Proof 4]** Comparative Baselines Benchmark table (comparing Our 10D LFIG vs DTW-1NN, ROCKET, MiniROCKET, and HIVE-COTE 2.0).
     - **5-Fold Nested Cross-Validation:** Runs outer 5 folds (with an inner 3-fold CV grid search parameter selector) with real-time progress bars, logging fold accuracy.
     - **Incremental Saving & Auto-Downloads**: Saves results after each dataset to `master_benchmark_results.csv` and triggers automatic browser download prompts at the end of the run.
     - **Master Comparative Benchmark Table:** Prints a final master summary table comparing 3D standard LFIG, 10D proposed LFIG, and Nested CV accuracy across all datasets.

---

## 2.1 Benchmark Dataset Sizes and Data Dimensions

The full UCR Time Series Archive subset evaluated in this project consists of **23 datasets** across six domains. Their detailed sizes (training/test split) and sequence lengths are tabulated below:

| # | Dataset | Domain | Train Samples | Test Samples | Series Length | Classes | Total Samples |
|---|---|---|---|---|---|---|---|
| 1 | GunPoint | Motion | 50 | 150 | 150 | 2 | 200 |
| 2 | Coffee | Spectro | 28 | 28 | 286 | 2 | 56 |
| 3 | ArrowHead | Image | 36 | 175 | 251 | 3 | 211 |
| 4 | ECG200 | ECG | 100 | 100 | 96 | 2 | 200 |
| 5 | Chinatown | Sensor | 20 | 345 | 24 | 2 | 365 |
| 6 | ItalyPowerDemand | Sensor | 67 | 1029 | 24 | 2 | 1096 |
| 7 | SonyAIBORobotSurface1 | Sensor | 20 | 601 | 70 | 2 | 621 |
| 8 | TwoLeadECG | ECG | 23 | 1139 | 82 | 2 | 1162 |
| 9 | ECGFiveDays | ECG | 23 | 861 | 136 | 2 | 884 |
| 10 | MoteStrain | Sensor | 20 | 1252 | 84 | 2 | 1272 |
| 11 | Beef | Spectro | 30 | 30 | 470 | 5 | 60 |
| 12 | OliveOil | Spectro | 30 | 30 | 570 | 4 | 60 |
| 13 | Meat | Spectro | 60 | 60 | 448 | 3 | 120 |
| 14 | BeetleFly | Image | 20 | 20 | 512 | 2 | 40 |
| 15 | BirdChicken | Image | 20 | 20 | 512 | 2 | 40 |
| 16 | FaceFour | Image | 24 | 88 | 350 | 4 | 112 |
| 17 | SyntheticControl | Simulated | 300 | 300 | 60 | 6 | 600 |
| 18 | CBF | Simulated | 30 | 900 | 128 | 3 | 930 |
| 19 | TwoPatterns | Simulated | 1000 | 4000 | 128 | 4 | 5000 |
| 20 | Wafer | Sensor | 1000 | 6164 | 152 | 2 | 7164 |
| 21 | FordA | Sensor | 3601 | 1320 | 500 | 2 | 4921 |
| 22 | Yoga | Image | 300 | 3000 | 426 | 2 | 3300 |
| 23 | SwedishLeaf | Image | 500 | 625 | 128 | 15 | 1125 |

---

## 3. Option B: Running Local Python Commands

### Environment Setup First:
Open your terminal in the project directory (`/Users/adarshfulzele/Desktop/RP/Best A`) and run:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Command 1: Run Single-Dataset Nested Cross-Validation
To test the nested cross-validation and automatic segmentation selector on a single dataset:
```bash
python -c "from src.evaluation.tuning import run_nested_cv; run_nested_cv('GunPoint')"
```
- **What Happens:** Loads `GunPoint` from UCR data, computes training autocorrelation variance to pick CPD vs Fixed windowing, runs 3-fold inner CV to pick best hyperparameter grid ($z, k, w$, classifier), and evaluates across 5 outer folds. Prints fold accuracies and mean $\pm$ std.

### Command 2: Run Feature Redundancy & Multicollinearity Analysis
To inspect correlation, PCA variance, and Variance Inflation Factors (VIF) for the 10 granule features:
```bash
python -c "from src.features.redundancy import run_redundancy_analysis; run_redundancy_analysis('GunPoint')"
```
- **What Happens:** Extracts 10D granule feature vectors across all samples of `GunPoint`. Generates:
  - `plots/feature_correlation_matrix.png` (Pearson correlation heatmap).
  - `plots/feature_pca_variance.png` (PCA cumulative variance curve).
  - Terminal output of Variance Inflation Factors (VIF) flagging any collinear features ($VIF > 10$).

### Command 3: Run Full Benchmark Suite & Demšar CD Diagram
To evaluate all datasets in the UCR catalog, reproduce aeon baselines, and generate statistical Critical Difference diagrams:
```bash
python -c "from src.evaluation.benchmark import run_full_benchmark; run_full_benchmark()"
```
- **What Happens:**
  1. Iterates over 23 UCR datasets across 6 domains (Motion, Spectro, Image, ECG, Sensor, Simulated).
  2. Runs reproducible baselines (`ROCKET`, `MiniROCKET`, `DTW-1NN`) under identical splits.
  3. Conducts Friedman chi-square significance testing across all models.
  4. Generates Demšar-style Critical Difference diagram saved to `plots/cd_diagram.png`.
  5. Updates detailed performance records in `plots/evaluation_results.md`.

---

## 4. End-to-End System Procedure & Code Execution Flow

The diagram and narrative below explain how data and code flow through the system during execution:

```
+-----------------------------------------------------------------------------------+
| 1. DATASET LOADING & PREPROCESSING (src/datasets/loader.py / aeon.datasets)      |
|    - Load raw univariate time series X shape: (N, L), class labels y: (N,)            |
|    - Format into explicit C-contiguous np.float64 NumPy arrays                       |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| 2. AUTO-SEGMENTATION STRATEGY ROUTING (src/evaluation/tuning.py)                 |
|    - Compute lag-1 autocorrelation variance across training signals: sigma^2_r       |
|    - Decision Rule:                                                               |
|        * If sigma^2_r > 0.05  => Route to Bottom-Up Change Point Detection (CPD)  |
|        * If sigma^2_r <= 0.05 => Route to Fixed-Window Partitioning               |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| 3. 10D MULTI-FEATURE GRANULE EXTRACTION (src/features/extractor.py)              |
|    - For each segment [start, end], fit linear trend y = a*tau + b                |
|    - Compute fuzzy envelope bounds [Lower, Upper] using z * sigma_residuals       |
|    - Extract 10D vector per granule:                                             |
|      [Lower, Upper, Slope, Shannon Entropy, Variance, Volatility,                 |
|       Curvature, Intercept, Energy, Skewness]                                     |
|    - Signal X of length L is compressed into S granules: shape (S, 10)            |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| 4. PAIRWISE 3-WAY DISTANCE MATRICES COMPUTATION (src/similarity/hybrid.py)       |
|    - Hausdorff Distance Matrix (D_H): Set overlap between boundary envelopes      |
|    - Slope FastDTW Matrix (D_DTW): Phase alignment of trend slopes                |
|    - 10D Feature Cosine FastDTW Matrix (D_Cos): Directional feature alignment     |
|    - Symmetric handling for train-train (D[j, i] = dh); asymmetric for test-train|
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| 5. HYBRID DISTANCE FUSION & WEIGHT LEARNING (src/similarity/hybrid.py)           |
|    - Min-Max Normalize D_H, D_DTW, D_Cos independently to [0, 1]                 |
|    - Weight Learning: Fit Pairwise Logistic Regression or Inner CV Grid Search    |
|      to find weights [w_H, w_DTW, w_Cos] summing to 1.0                           |
|    - Compute Fused Distance Matrix: D_Fused = w_H*D_H + w_DTW*D_DTW + w_Cos*D_Cos|
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| 6. CLASSIFICATION & EVALUATION (src/classifiers/models.py)                        |
|    - KNN Classifier: Custom distance-space k-Nearest Neighbors                    |
|    - Precomputed Kernel SVM: Transform D_Fused to RBF Kernel:                      |
|      K(i, j) = exp(-gamma * D_Fused[i, j]^2) where gamma via median heuristic    |
|    - Fit precomputed SVC(kernel='precomputed') on training matrix                 |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| 7. STATISTICAL REPORTING & PROOFS GENERATION (src/evaluation/benchmark.py)       |
|    - Print Proof 1 (Granule lengths), Proof 2 (3D vs 10D), Proof 3 (LOFO table)   |
|    - Print Proof 4 (Baselines comparison vs ROCKET, MiniROCKET, DTW-1NN, HC2)     |
|    - Report Outer Fold Mean +/- Std metrics                                       |
|    - Generate Demšar Critical Difference Diagram (plots/cd_diagram.png)           |
+-----------------------------------------------------------------------------------+
```

---

## 5. File Structure Quick Reference

- **`LFIG_Adaptive_Pipeline_Colab.ipynb`**: Interactive, self-contained Google Colab notebook. **(Run this for an all-in-one interactive demonstration)**.
- **`run_colab_standalone.py`**: Standalone command-line Python script mirroring the Colab notebook logic.
- **`EXECUTION_GUIDE.md`**: This comprehensive execution guide and workflow manual.
- **`README.md`**: Primary repository overview, setup instructions, and documentation sitemap.
- **`research_defense_guide.md`**: 22-question viva defense guide (Question $\rightarrow$ Answer $\rightarrow$ Gap $\rightarrow$ Fix).
- **`paper_draft.md`**: Scientific paper manuscript with related work differentiation table and proof metrics.
- **`src/datasets/`**: Dataset loading (`loader.py`), statistics (`stats.py`), and catalog of 23 UCR datasets (`ucr_catalog.py`).
- **`src/segmentation/`**: Adaptive segmentation routines (`adaptive.py`) including CPD, fixed windowing, variance, and entropy splitting.
- **`src/granulation/`**: Linear Fuzzy Information Granulation (`lfig.py`).
- **`src/features/`**: 10D feature extractor (`extractor.py`), Random Forest importance (`importance.py`), and correlation/PCA/VIF redundancy (`redundancy.py`).
- **`src/similarity/`**: Hybrid distance computation and fusion weight learning (`hybrid.py`).
- **`src/classifiers/`**: Custom distance KNN and precomputed RBF Kernel SVM models (`models.py`).
- **`src/evaluation/`**: Hyperparameter tuning & nested CV (`tuning.py`), aeon baselines (`baselines.py`), Demšar CD diagrams (`critical_difference.py`), and full benchmark harness (`benchmark.py`).
