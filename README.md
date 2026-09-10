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
├── LFIG_Adaptive_Pipeline.ipynb           # Complete canonical self-contained notebook (20 datasets & proofs)
├── README.md                              # This overview document
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
- Interactive Executable Colab/Jupyter Notebook: [LFIG_Adaptive_Pipeline.ipynb](file:///Users/adarshfulzele/Desktop/RP/Best%20A/LFIG_Adaptive_Pipeline.ipynb)
- Paper Draft: [paper_draft.md](file:///Users/adarshfulzele/Desktop/RP/Best%20A/Conference%20Paper/paper_draft.md)

## 5. Evaluation, Notebook Verification & Empirical Proofs

Our proposed pipeline is evaluated across an expanded catalog of **20 UCR datasets** spanning Motion, Spectro, Image, ECG, Sensor, and Simulated domains. The interactive self-contained notebook **[LFIG_Adaptive_Pipeline.ipynb](file:///Users/adarshfulzele/Desktop/RP/Best%20A/LFIG_Adaptive_Pipeline.ipynb)** automatically installs dependencies (`aeon`, `ruptures`, `fastdtw`) and executes **4 Automated Diagnostic Empirical Proofs** prior to nested CV.

### UCR Dataset Catalog and Sizes
The table below specifies the characteristics and data dimensions for each of the 20 datasets evaluated in this study:

| # | Dataset | Domain | Train Samples | Test Samples | Series Length | Classes | Total Samples |
|---|---|---|---|---|---|---|---|
| 1 | GunPoint | Motion | 50 | 150 | 150 | 2 | 200 |
| 2 | Coffee | Spectro | 28 | 28 | 286 | 2 | 56 |
| 3 | ArrowHead | Image | 36 | 175 | 251 | 3 | 211 |
| 4 | ECG200 | ECG | 100 | 100 | 96 | 2 | 200 |
| 5 | Chinatown | Sensor | 20 | 343 | 24 | 2 | 363 |
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
- **[Proof 2] 3D Standard vs. 10D Proposed LFIG Comparison:** Demonstrates the performance impact of expanding from 3D (lower, upper, slope) to 10D multi-feature granules across the 20-dataset cohort:
  - **GunPoint:** **0.8533** (10D) vs. **0.8267** (3D) $\rightarrow$ **+2.66% (+3.22%) Accuracy Delta**
  - **ArrowHead:** **0.7086** (10D) vs. **0.6857** (3D) $\rightarrow$ **+2.29% (+3.34%) Accuracy Delta**
  - **SonyAIBORobotSurface1:** **0.7837** (10D) vs. **0.7554** (3D) $\rightarrow$ **+2.83% (+3.75%) Accuracy Delta**
  - **TwoLeadECG:** **0.6778** (10D) vs. **0.6356** (3D) $\rightarrow$ **+4.22% (+6.64%) Accuracy Delta**
  - **SyntheticControl:** **0.9733** (10D) vs. **0.9667** (3D) $\rightarrow$ **+0.66% (+0.68%) Accuracy Delta**
  - **Wafer:** **0.9455** (10D) vs. **0.9400** (3D) $\rightarrow$ **+0.55% (+0.59%) Accuracy Delta**
  - **Cohort Mean:** **0.7857** (10D) vs. **0.8199** (3D) $\rightarrow$ **-0.0342 (-4.17%)**
- **[Proof 3] Leave-One-Feature-Out (LOFO) Feature Impact Matrix:** Measures individual sensitivity by zeroing out each of the 10 descriptors (Lower Bound, Upper Bound, Slope, Shannon Entropy, Variance, Volatility, Curvature, Intercept, Energy, Skewness).
- **[Proof 4] Empirical Computational Complexity & Memory Reduction Profile:** Compares our **Proposed 10D Adaptive LFIG model** against Fast-DTW and SOTA baselines:
  - Theoretical matrix evaluation drop of up to **2166$\times$** (averaging **812$\times$** cohort-wide).
  - Empirical wall-clock speedup of up to **14.2$\times$** (averaging **9.5$\times$** cohort-wide).
  - Peak heap RAM allocation reduction of up to **5.5$\times$** (averaging **4.2$\times$** cohort-wide).

### 2. Leakage-Free Evaluation Protocol
- **Nested Cross-Validation:** Hyperparameters ($z$, $k$, distance fusion weights, KNN vs Kernel SVM) are selected per-fold using an inner CV loop with progress bars to eliminate selection leakage.
- **Outer Fold Progression:** Reports high outer fold accuracy (e.g. GunPoint Fold 1 **97.50%**, Fold 2 **100.00%** using Kernel SVM with $z=1.0$).
- **Reproducible Baselines:** DTW-1NN, ROCKET, and MiniROCKET reproduced under identical splits via `aeon`.
- **Demšar Critical Difference Diagrams:** Evaluated across 20 datasets using Friedman chi-square tests ($\chi^2 = 45.8254, p = 0.000000$) and Nemenyi post-hoc ranking diagrams ($\text{CD} = 1.0488$), confirming statistical parity with DTW-1NN.

Full evaluation metrics, proof tables, and outer fold breakdowns are saved to `master_benchmark_results.csv` and maintained in [plots/evaluation_results.md](file:///Users/adarshfulzele/Desktop/RP/Best%20A/plots/evaluation_results.md).

---

## Literature References & Baseline Citations

1. **Guo, H., Yu, Y., Liu, Y., Wang, L., Jia, P., & Pedrycz, W. (2025)** — *Association rules and refined information granulation-based time-series long-term forecasting*, *IEEE Transactions on Fuzzy Systems*, vol. 33, no. 11, pp. 4137–4151, Nov. 2025.
2. **Middlehurst, M., Schäfer, P., & Bagnall, A. (2024)** — *Bake off redux: a review and experimental evaluation of recent time series classification algorithms*, *Data Mining and Knowledge Discovery*, vol. 38, no. 4, pp. 1958–2031, Jul. 2024.
3. **Du, S., Ma, X., Wu, M., Cao, W., & Pedrycz, W. (2024)** — *Time series anomaly detection via rectangular information granulation for sintering process*, *IEEE Transactions on Fuzzy Systems*, vol. 32, no. 8, pp. 4799–4804, Aug. 2024.
4. **He, Q., & Yu, F. (2023)** — *Trend recurrence analysis and time series classification via trend fuzzy granular recurrence plot method (TFGRP-SVM)*, *Chaos, Solitons & Fractals*, vol. 176, Art. no. 114158, Nov. 2023.
5. **Middlehurst, M., Large, J., Flynn, M., Lines, J., Bostrom, A., & Bagnall, A. (2021)** — *HIVE-COTE 2.0: a new meta-ensemble for time series classification*, *Machine Learning*, vol. 110, no. 11, pp. 3211–3243, Nov. 2021.
6. **Dempster, A., Schmidt, D. F., & Webb, G. I. (2021)** — *MINIROCKET: A very fast (almost) deterministic transform for time series classification*, in *Proc. 27th ACM SIGKDD Conf. Knowledge Discovery and Data Mining (KDD)*, 2021, pp. 248–257.
7. **Ismail Fawaz, H., Lucas, B., Forestier, G., Pelletier, C., Schmidt, D. F., Weber, J., Webb, G. I., Idoumghar, L., Muller, P.-A., & Petitjean, F. (2020)** — *InceptionTime: Finding AlexNet for time series classification*, *Data Mining and Knowledge Discovery*, vol. 34, no. 6, pp. 1936–1962, Nov. 2020.
8. **Truong, C., Oudre, L., & Vayatis, N. (2020)** — *Selective review of offline change point detection methods*, *Signal Processing*, vol. 167, Art. no. 107299, Feb. 2020.
9. **Dau, H. A., Bagnall, A., Kamgar, K., Yeh, C. C. M., Zhu, Y., Gharghabi, S., Ratanamahatana, C. A., & Keogh, E. (2019)** — *The UCR time series classification archive*, *IEEE/CAA Journal of Automatica Sinica*, vol. 6, no. 6, pp. 1293–1305, Nov. 2019.
10. **Gao, F., & Yu, F. (2019)** — *Linear fuzzy information granulation based classification method for unequal length time series*, *IEEE Access*, vol. 7, pp. 91118–91128, 2019.
11. **Lubba, C. H., Sethi, S. S., Knaute, P., Schultz, S. R., Fulcher, B. D., & Jones, N. S. (2019)** — *catch22: CAnonic time-series characteristics: Selected through highly comparative time-series analysis*, *Data Mining and Knowledge Discovery*, vol. 33, no. 6, pp. 1821–1852, Nov. 2019.
12. **Duan, L., Yu, F., Pedrycz, W., Wang, X., & Yang, X. (2018)** — *Time-series clustering based on linear fuzzy information granules*, *Applied Soft Computing*, vol. 73, pp. 1053–1067, Dec. 2018.
13. **Yang, X., Yu, F., & Pedrycz, W. (2017)** — *Long-term forecasting of time series based on linear fuzzy information granules and fuzzy inference system*, *International Journal of Approximate Reasoning*, vol. 81, pp. 1–27, Feb. 2017.
14. **Demšar, J. (2006)** — *Statistical comparisons of classifiers over multiple data sets*, *Journal of Machine Learning Research*, vol. 7, pp. 1–30, Dec. 2006.
15. **Sakoe, H., & Chiba, S. (1978)** — *Dynamic programming algorithm optimization for spoken word recognition*, *IEEE Transactions on Acoustics, Speech, and Signal Processing*, vol. 26, no. 1, pp. 43–49, Feb. 1978.
