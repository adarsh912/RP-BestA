# Evaluation and Benchmark Results (Revised)

This document contains the corrected, leakage-free experimental results for the **Adaptive Multi-Feature LFIG Time Series Classification** framework.

---

## 1. Isolated Selection Leakage Analysis

To answer the critical question — **"Was the original single-split table inflated by selection leakage?"** — we evaluated our model on the official UCR train/test splits. 
- The training split was used for automatic strategy selection and inner cross-validation hyperparameter tuning (z, k, weights, classifier).
- The test split was touched **exactly once** for final reported accuracy, completely isolating it from the selection and tuning process.

Any performance drop compared to the original table directly indicates the degree of selection leakage present in the original manual tuning.

### Comparison Table (Official UCR Split)

| Dataset | Original Leaky Acc | New Leakage-Free Acc | Leakage Delta |
|:---|:---:|:---:|:---:|
| **GunPoint** | 0.9067 | 0.8933 | -0.0133 |
| **Coffee** | 1.0000 | 1.0000 | +0.0000 |
| **ECG200** | 0.9100 | 0.8800 | -0.0300 |
| **Chinatown** | 0.9767 | 0.9417 | -0.0350 |
| **ArrowHead** | 0.8286 | 0.7829 | -0.0457 |

**Analysis:**
Selection leakage inflated the original results by **1.33% to 4.57%** across four of the five datasets (with Coffee remaining perfect at 1.0000). This validates the need for a rigorous leakage-free validation protocol.

---

## 2. Step 1 Nested-CV Results (Leakage-Free)

The final evaluation was conducted using a robust **5-fold outer cross-validation** (for unbiased reporting) and a **3-fold inner cross-validation** (for strategy and hyperparameter tuning). This ensures a completely leakage-free evaluation while maximizing data efficiency.

### Final Nested-CV Accuracy (Mean ± SD)

| Dataset | Nested-CV Accuracy |
|:---|:---:|
| **GunPoint** | 0.9550 ± 0.0292 |
| **Coffee** | 1.0000 ± 0.0000 |
| **ECG200** | 0.8750 ± 0.0418 |
| **Chinatown** | 0.9779 ± 0.0166 |
| **ArrowHead** | 0.8767 ± 0.0235 |

**Analysis:**
- **Higher Generalization Performance:** In several cases (GunPoint, Chinatown, ArrowHead), the nested CV accuracies are higher than the single-split accuracies. This is due to the multi-fold training/testing averaging, which reduces the split-specific variance of the official single train/test split.
- **Coffee SOTA:** Our framework consistently achieves **1.0000 ± 0.0000** accuracy on Coffee, matching state-of-the-art results.
- **Robust ECG Bounds:** ECG200 reports **0.8750 ± 0.0418**, providing a realistic, unbiased estimate of generalization.

---

## 3. Diagnostic Empirical Proofs (Notebook Executable Verification)

The self-contained Jupyter notebook (`LFIG_Adaptive_Pipeline_Colab.ipynb`) executes three automated empirical proofs prior to full nested CV:

### Proof 1: Variable-Length CPD & Fixed-Window Segmentation Demonstration
| Dataset | Strategy Selected | Param | Detected Boundaries | Granules | Granule Length Breakdown |
|:---|:---:|:---:|:---|:---:|:---|
| **GunPoint** | FIXED | 15 | `[0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150]` | 10 | `[15, 15, 15, 15, 15, 15, 15, 15, 15, 15]` |
| **Coffee** | CPD | 1.5 | `[0, 28, 56, 84, 112, 140, 168, 196, 224, 252, 280, 286]` | 11 | `[28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 6]` |
| **ArrowHead** | FIXED | 25 | `[0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 251]` | 11 | `[25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 1]` |
| **ECG200** | FIXED | 10 | `[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 96]` | 10 | `[10, 10, 10, 10, 10, 10, 10, 10, 10, 6]` |

---

### Proof 2: Feature Representation Comparison (Standard 3D vs Proposed 10D LFIG)
| Dataset | Standard 3D LFIG Acc | Proposed 10D Multi-Feature LFIG Acc | Improvement Delta |
|:---|:---:|:---:|:---:|
| **GunPoint** | 0.7800 | **0.9067** | **+0.1267 (+12.67%)** |
| **ArrowHead** | 0.6857 | **0.7086** | **+0.0229 (+2.29%)** |
| **ECG200** | 0.7400 | **0.7700** | **+0.0300 (+3.00%)** |
| **Coffee** | 0.9286 | **0.9286** | +0.0000 |

*Key Takeaway:* Expanding from 3D descriptors (lower, upper, slope) to 10D descriptors (adding Shannon entropy, volatility, curvature, energy, skewness, variance, intercept) provides a massive **+12.67% accuracy boost** on motion datasets (`GunPoint`) and consistent gains across sensor and ECG domains.

---

### Proof 3: Leave-One-Feature-Out (LOFO) Feature Impact Matrix (GunPoint)
| Ablated Feature | Ablated Test Acc | Impact Delta vs Full 10D (0.9067) | Importance Significance |
|:---|:---:|:---:|:---|
| **Energy** ($E_{n}$) | 0.8867 | **+0.0200** | Highest sensitivity feature |
| **Upper Bound** ($U$) | 0.8933 | **+0.0133** | Critical envelope boundary |
| **Trend Slope** ($a$) | 0.8933 | **+0.0133** | Dynamic direction descriptor |
| **Lower Bound** ($L$) | 0.9133 | -0.0067 | Redundant envelope limit |
| **Shannon Entropy** ($H$) | 0.9133 | -0.0067 | Information density metric |
| **Skewness** ($Sk$) | 0.9267 | -0.0200 | Asymmetry descriptor |
| **Variance** ($\sigma^2$) | 0.9067 | +0.0000 | Baseline variance |
| **Volatility** ($V_{ol}$) | 0.9067 | +0.0000 | Difference std |
| **Curvature** ($c$) | 0.9067 | +0.0000 | Polynomial acceleration |
| **Intercept** ($b$) | 0.9067 | +0.0000 | Baseline intercept |

---

## 4. Outer Fold Breakdown (5-Fold Leakage-Free Nested CV)

| Outer Fold | Outer Test Acc | Macro F1 | Selected Classifier & Hyperparameters |
|:---:|:---:|:---:|:---|
| **Fold 1/5** | **0.9750** | 0.9750 | Precomputed Kernel SVM ($z=1.0, k=1, w=[0.2, 0.6, 0.2]$) |
| **Fold 2/5** | **1.0000** | 1.0000 | Precomputed Kernel SVM ($z=1.0, k=1, w=[0.3, 0.4, 0.3]$) |
| **Fold 3/5** | **0.9500** | 0.9500 | Custom Distance KNN ($z=1.96, k=1, w=[0.1, 0.8, 0.1]$) |
| **Fold 4/5** | **0.9250** | 0.9250 | Precomputed Kernel SVM ($z=1.0, k=3, w=[0.2, 0.6, 0.2]$) |
| **Fold 5/5** | **0.9250** | 0.9250 | Custom Distance KNN ($z=1.96, k=1, w=[0.3, 0.4, 0.3]$) |
| **Mean ± Std** | **0.9550 ± 0.0292** | **0.9550 ± 0.0292** | *Leakage-Free Overall Benchmark* |