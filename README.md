# Adaptive Multi-Feature Linear Fuzzy Information Granulation with Hybrid Similarity Learning

This repository implements an advanced, statistically rigorous framework for time series classification (TSC) using **Adaptive Segmentation**, **Enhanced Linear Fuzzy Information Granulation (LFIG)**, **Multi-Feature Granule Representation**, and **Hybrid Similarity Learning with Rank Fusion**.

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
├── literature_review.md      # Seminal papers & comparison tables
├── phases.md                 # Project development roadmap (Phases 1-12)
├── milestones.md             # Project milestones status checklist
├── progress.md               # Code walkthrough and development progress
├── requirements.txt          # Python package requirements
├── plots/                    # Output plots and visualizations
│   ├── cd_diagram.png        # Critical difference diagram (Friedman + Nemenyi)
│   ├── feature_correlation_matrix.png
│   ├── feature_pca_variance.png
│   ├── evaluation_results.md # Performance results from nested CV
│   └── ...
└── src/                      # Source Code
    ├── datasets/
    │   ├── loader.py         # UCR/UEA datasets loader
    │   ├── stats.py          # Data analysis & statistics
    │   └── ucr_catalog.py    # Catalog of 23 UCR datasets across 6 domains
    ├── segmentation/
    │   ├── adaptive.py       # Segmentation algorithms (Fixed, Var, Ent, CPD)
    │   └── visualize_segmentation.py
    ├── granulation/
    │   ├── lfig.py           # Linear Fuzzy Information Granulation
    │   └── visualize_lfig.py
    ├── features/
    │   ├── extractor.py      # 10D feature sequence extractor
    │   ├── importance.py     # Random Forest feature Gini importance
    │   └── redundancy.py     # Feature correlation, PCA, and VIF analysis
    ├── similarity/
    │   └── hybrid.py         # Hausdorff, DTW, Cosine, and fusion weights learning
    ├── classifiers/
    │   └── models.py         # Precomputed distance kNN/SVM & boosting models
    └── evaluation/
        ├── tuning.py         # Nested CV & auto-segmentation selector
        ├── baselines.py      # Reproducible aeon baselines
        ├── critical_difference.py # Demšar CD diagrams
        └── benchmark.py      # Benchmark runner and ablation harness
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
- Development Roadmap: [phases.md](file:///Users/adarshfulzele/Desktop/RP/Best%20A/phases.md)
- Milestones Checklist: [milestones.md](file:///Users/adarshfulzele/Desktop/RP/Best%20A/milestones.md)
- Development Progress & Walkthrough: [progress.md](file:///Users/adarshfulzele/Desktop/RP/Best%20A/progress.md)
- Methodology & Design: [methodology_design.md](file:///Users/adarshfulzele/Desktop/RP/Best%20A/methodology_design.md)
- Paper Draft: [paper_draft.md](file:///Users/adarshfulzele/Desktop/RP/Best%20A/paper_draft.md)

---

## 5. Evaluation & Baselines

Our proposed pipeline is evaluated across an expanded catalog of **23 UCR datasets** spanning Motion, Spectro, Image, ECG, Sensor, and Simulated domains. 

The evaluation framework incorporates:
- **Nested Cross-Validation:** Hyperparameters are selected per-fold using an inner CV to eliminate selection leakage.
- **Repeated Evaluation:** Accuracy, precision, recall, and Macro F1 scores are reported as `mean ± std` over 10 repeated stratified splits.
- **Reproducible Baselines:** DTW-1NN, ROCKET, and MiniROCKET are run under the exact same splits (via `aeon` integration) to ensure direct comparability.
- **Statistical Significance:** A Demšar-style Critical Difference (CD) diagram is generated using a Friedman test followed by Nemenyi post-hoc tests.

Full evaluation and benchmarking results are maintained in [plots/evaluation_results.md](file:///Users/adarshfulzele/Desktop/RP/Best%20A/plots/evaluation_results.md).
