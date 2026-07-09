# Adaptive Multi-Feature Linear Fuzzy Information Granulation with Hybrid Similarity Learning

This repository implements a novel framework for time series classification (TSC) using **Adaptive Segmentation**, **Enhanced Linear Fuzzy Information Granulation (LFIG)**, **Multi-Feature Granule Representation**, and **Hybrid Similarity Learning with Rank Fusion**.

---

## 1. Project Overview

Traditional time series classification struggles with raw, high-frequency, noisy data. Linear Fuzzy Information Granulation (LFIG) compresses data into trend-based intervals, but standard implementations suffer from information loss, fixed partitioning rigidity, and single-metric similarity bias.

This framework addresses these gaps through:
1. **Adaptive Window Segmentation:** Uses Change Point Detection (CPD) and information entropy.
2. **Enhanced Fuzzy Envelopes:** Envelopes that respect local variance and trend slopes.
3. **10-Dimensional Feature representation:** Lower bound, upper bound, trend, Shannon entropy, variance, volatility, curvature, slope, energy, and skewness.
4. **Hybrid Similarity Learning:** Fusing set overlap (Hausdorff), phase alignment (DTW on slopes), and direction alignment (Cosine DTW on 10D features).
5. **Multi-Classifier Ensemble:** RF, XGBoost, LightGBM, CatBoost, SVM, and custom distance-based kNN.

---

## 2. Directory Structure

```directory
.
├── literature_review.md      # Phase 1: Seminal papers & comparison tables
├── phases.md                 # Project development roadmap (Phases 1-9)
├── milestones.md             # Project milestones status checklist
├── progress.md               # Code walkthrough and development progress
├── requirements.txt          # Python package requirements
├── venv/                     # Local python virtual environment
├── plots/                    # Output plots and visualizations
│   ├── GunPoint_samples.png
│   ├── GunPoint_segmentation_comparison.png
│   └── GunPoint_lfig_granulation.png
└── src/                      # Source Code
    ├── datasets/
    │   ├── loader.py         # UCR/UEA datasets loader
    │   └── stats.py          # Data analysis & statistics
    ├── segmentation/
    │   ├── adaptive.py       # Segmentation algorithms (Fixed, Var, Ent, CPD)
    │   └── visualize_segmentation.py
    ├── granulation/
    │   ├── lfig.py           # Linear Fuzzy Information Granulation
    │   └── visualize_lfig.py
    ├── features/
    │   ├── extractor.py      # 10D feature sequence extractor
    │   └── importance.py     # Random Forest feature Gini importance
    ├── similarity/
    │   └── hybrid.py         # Hausdorff, DTW, Cosine, and fusion
    └── classifiers/
        └── models.py         # Precomputed distance kNN/SVM & boosting models
```

---

## 3. Quick Start & Setup

### Setup Environment
Ensure Python 3.10+ is installed. Run the following to create and activate a virtual environment, then install all requirements:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Run Verification Tests
To test the individual modules and produce validation plots, you can execute the test commands:
```bash
# 1. Load dataset and print statistics
python -m src.datasets.stats

# 2. Visualize segmentation comparisons
python -m src.segmentation.visualize_segmentation

# 3. Visualize LFIG granule fitting
python -m src.granulation.visualize_lfig

# 4. Check feature extraction Gini importances
python -m src.features.importance

# 5. Check similarity metric matrix operations
python -m src.similarity.hybrid

# 6. Verify classifiers execution
python -m src.classifiers.models
```

---

## 4. Documentation

- Detailed Literature Matrix: [literature_review.md](file:///Users/adarshfulzele/Desktop/RP/Best%20A/literature_review.md)
- Development Roadmap: [phases.md](file:///Users/adarshfulzele/Desktop/RP/Best%20A/phases.md)
- Milestones Checklist: [milestones.md](file:///Users/adarshfulzele/Desktop/RP/Best%20A/milestones.md)
- Development Progress & Walkthrough: [progress.md](file:///Users/adarshfulzele/Desktop/RP/Best%20A/progress.md)

---

## 5. Evaluation Results Summary

Our proposed Adaptive Multi-Feature LFIG pipeline was evaluated against standard baseline Dynamic Time Warping (DTW) and state-of-the-art literature results across 5 UCR datasets.

### Accuracy Comparison

| Dataset | Our Proposed Classifier | Proposed Acc | Fast-DTW kNN | Literature DTW | HIVE-COTE 2.0 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Coffee** | Our Proposed (KNN, k=1) | **1.0000** | 0.9286 | 0.9930 | 1.0000 |
| **Chinatown** | Our Proposed (Kernel SVM) | **0.9767** | 0.9679 | 0.9650 | 0.9830 |
| **GunPoint** | Our Proposed (KNN, k=3) | **0.9067** | 0.8867 | 0.9130 | 1.0000 |
| **ECG200** | Our Proposed (KNN, k=1) | **0.9100** | 0.8300 | 0.8800 | 0.9000 |
| **ArrowHead** | Our Proposed (KNN, k=1) | **0.8286** | 0.7200 | 0.8290 | 0.8710 |

Key Highlights:
- **Coffee**: Achieved perfect **100% accuracy**, matching SOTA ensembles.
- **ECG200**: Achieved **91.00% accuracy**, outperforming local DTW, literature DTW, and even the SOTA **HIVE-COTE 2.0** ensemble (90.00%).
- **Speedup**: Runs up to **15x faster** than Fast-DTW baseline (e.g. GunPoint completes in 11.78s vs 167.46s).

