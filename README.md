# Adaptive Multi-Feature Linear Fuzzy Information Granulation with Hybrid Similarity Learning

This repository implements the **Proposed Adaptive Multi-Feature Linear Fuzzy Information Granulation (Proposed Adaptive LFIG)** framework for time series classification (TSC) using **Adaptive Segmentation**, **Enhanced Linear Fuzzy Information Granulation (LFIG)**, **10-Dimensional Multi-Feature Granule Representation**, and **Hybrid Similarity Learning with Rank Fusion**.

---

## 1. Project Overview

Traditional time series classification struggles with raw, high-frequency, noisy data. Linear Fuzzy Information Granulation (LFIG) compresses data into trend-based intervals, but standard implementations suffer from information loss, fixed partitioning rigidity, and single-metric similarity bias.

This framework addresses these gaps through:
1. **Adaptive Window Segmentation:** Uses Change Point Detection (CPD) and information entropy.
2. **Enhanced Fuzzy Envelopes:** Envelopes that respect local variance and trend slopes.
3. **10-Dimensional Feature representation:** Lower bound, upper bound, trend, Shannon entropy, variance, volatility, curvature, slope, energy, and skewness.
4. **Hybrid Similarity Learning:** Fusing set overlap (Hausdorff), phase alignment (DTW on slopes), and direction alignment (Cosine DTW on 10D features). Fusion weights are dynamically learned using logistic regression or inner cross-validation grid search.
5. **Rigorous Validation Protocol:** Uses nested cross-validation (5-fold outer, 3-fold inner) to prevent selection leakage and repeated stratified splits to report statistical variance.
6. **Multi-Classifier Ensemble:** RF, XGBoost, LightGBM, CatBoost, SVM, and custom distance-based kNN.

---

## 2. Directory Structure

```directory
.
├── LFIG_Adaptive_Pipeline_Colab.ipynb      # CPU-focused self-contained notebook
├── LFIG_Adaptive_Pipeline_Colab_GPU.ipynb  # GPU-accelerated self-contained notebook (complete 20 datasets)
├── README.md                               # This overview document
├── literature_review.md                     # Seminal papers & comparison tables
├── methodology_design.md                   # Technical design details
├── progress.md                              # Walkthrough and development progress
├── requirements.txt                         # Python package requirements
├── research_defense_guide.md                # Research defence Q&A guide
├── master_benchmark_results_gpu.csv         # Core GPU benchmark results CSV
├── master_benchmark_results_gpu_detailed.csv# Detailed GPU benchmark results CSV
├── master_benchmark_results_gpu.md          # Markdown GPU benchmark table
├── master_benchmark_results.csv             # Core CPU benchmark results CSV
└── plots/                                   # Visualizations & reports
    ├── cd_diagram.png                       # Demšar Critical Difference diagram
    ├── evaluation_results.md                # Detailed markdown results report
    └── ...
```

---

## 3. Quick Start & Setup

### Setup Environment
Ensure Python 3.10+ is installed. Run the following to create a virtual environment and install all requirements (including `aeon` for baselines):
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Hardware & System Specifications
Experiments and benchmark evaluations were profiled on the following workstation:
- **Workstation:** Dell Precision 3650 Tower
- **Processor:** 11th Gen Intel® Core™ i9-11900K @ 3.50 GHz (16 logical CPUs)
- **RAM:** 32 GB (32768 MB)
- **GPU Accelerator:** NVIDIA GeForce RTX 3070 (8 GB Dedicated VRAM)
- **Operating System:** Windows 11 Pro 64-bit (DirectX 12 Ultimate)

### Running the New Modules
To verify the new modules and run experimental tasks:
```bash
# 1. Run nested CV on a single dataset
python -c "from src.evaluation.tuning import run_nested_cv; run_nested_cv('GunPoint')"

# 2. Run feature redundancy analysis
python -c "from src.features.redundancy import run_redundancy_analysis; run_redundancy_analysis('GunPoint')"

# 3. Run full benchmark on all catalog datasets (produces CD diagram & evaluation_results.md)
python -c "from src.evaluation.benchmark import run_full_benchmark; run_full_benchmark()"
```

---

## 4. Documentation

- Detailed Literature Matrix: [literature_review.md](file:///Users/adarshfulzele/Desktop/RP/Best%20A/literature_review.md)
- Development Progress & Walkthrough: [progress.md](file:///Users/adarshfulzele/Desktop/RP/Best%20A/progress.md)
- Methodology & Design: [methodology_design.md](file:///Users/adarshfulzele/Desktop/RP/Best%20A/methodology_design.md)
- Research Defense Q&A Guide: [research_defense_guide.md](file:///Users/adarshfulzele/Desktop/RP/Best%20A/research_defense_guide.md)
- Interactive CPU Colab Notebook: [LFIG_Adaptive_Pipeline_Colab.ipynb](file:///Users/adarshfulzele/Desktop/RP/Best%20A/LFIG_Adaptive_Pipeline_Colab.ipynb)
- Interactive GPU Colab Notebook: [LFIG_Adaptive_Pipeline_Colab_GPU.ipynb](file:///Users/adarshfulzele/Desktop/RP/Best%20A/LFIG_Adaptive_Pipeline_Colab_GPU.ipynb)
- Paper Draft: [paper_draft.md](file:///Users/adarshfulzele/Desktop/RP/Best%20A/Conference%20Paper/paper_draft.md)

## 5. Evaluation, Notebook Verification & Empirical Proofs

Our proposed pipeline is evaluated across an expanded catalog of **20 UCR datasets** spanning Motion, Spectro, Image, ECG, Sensor, and Simulated domains. The interactive self-contained notebooks **[LFIG_Adaptive_Pipeline_Colab.ipynb](file:///Users/adarshfulzele/Desktop/RP/Best%20A/LFIG_Adaptive_Pipeline_Colab.ipynb)** (CPU-focused) and **[LFIG_Adaptive_Pipeline_Colab_GPU.ipynb](file:///Users/adarshfulzele/Desktop/RP/Best%20A/LFIG_Adaptive_Pipeline_Colab_GPU.ipynb)** (GPU-accelerated) automatically install dependencies (`aeon`, `ruptures`, `fastdtw`) and execute **4 Automated Diagnostic Empirical Proofs** prior to nested CV.

### UCR Dataset Catalog and Sizes
The table below specifies the characteristics and data dimensions for each of the 20 datasets evaluated in this study:

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

### 1. Diagnostic Empirical Proofs
- **[Proof 1] Variable-Length CPD Granulation:** Verifies dynamic boundary detection (e.g. GunPoint `[15, 15, ...]`, Coffee `[28, 28, ..., 6]`, ArrowHead `[25, ..., 1]`, ECG200 `[10, ..., 6]`).
- **[Proof 2] 3D Standard vs. 10D Proposed LFIG Comparison:** Demonstrates the performance impact of expanding from 3D (lower, upper, slope) to 10D multi-feature granules under the leakage-free evaluation protocol:
  - **GunPoint:** **0.8533** (10D) vs. **0.8267** (3D) $\rightarrow$ **+2.67% Accuracy Delta**
  - **ArrowHead:** **0.7086** (10D) vs. **0.6857** (3D) $\rightarrow$ **+2.29% Accuracy Delta**
  - **ECG200:** **0.7200** (10D) vs. **0.8000** (3D) $\rightarrow$ **-8.00% Accuracy Delta**
  - **Coffee:** **0.9286** (10D) vs. **0.9286** (3D) $\rightarrow$ **+0.00% Accuracy Delta**
- **[Proof 3] Leave-One-Feature-Out (LOFO) Feature Impact Matrix:** Measures individual sensitivity by zeroing out each of the 10 descriptors (Lower Bound, Upper Bound, Slope, Shannon Entropy, Variance, Volatility, Curvature, Intercept, Energy, Skewness).
- **[Proof 4] Comparative Baselines Benchmark:** Compares our **Proposed 10D Adaptive LFIG model** against SOTA baselines (DTW-1NN, ROCKET, MiniROCKET, HIVE-COTE 2.0).

### 2. Leakage-Free Evaluation Protocol
- **Nested Cross-Validation:** Hyperparameters ($z$, $k$, distance fusion weights, KNN vs Kernel SVM) are selected per-fold using an inner CV loop with progress bars to eliminate selection leakage.
- **Outer Fold Progression:** Reports high outer fold accuracy (e.g. GunPoint Fold 1 **97.50%**, Fold 2 **100.00%** using Kernel SVM with $z=1.0$).
- **Reproducible Baselines:** DTW-1NN, ROCKET, and MiniROCKET reproduced under identical splits via `aeon`.
- **Demšar Critical Difference Diagrams:** Evaluated across 20 datasets using Friedman chi-square tests and Nemenyi post-hoc ranking diagrams.

Full evaluation metrics, proof tables, and outer fold breakdowns are saved to `master_benchmark_results.csv` and maintained in [plots/evaluation_results.md](file:///Users/adarshfulzele/Desktop/RP/Best%20A/plots/evaluation_results.md).
