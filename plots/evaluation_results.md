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