# Adaptive Multi-Feature LFIG Classification: Master Reference Manual

This document serves as the single, comprehensive source of truth for the **Adaptive Multi-Feature Linear Fuzzy Information Granulation (LFIG)** time series classification project. It contains the theoretical foundations, implementation details, experimental protocols, empirical results, and diagnostic proof matrices.

---

## 🔍 Table of Contents
1. **Core Methodology & Mathematical Formulations**
   * Window Segmentation Strategies (CPD vs. Fixed)
   * 10D Granule Feature Representation
   * 3-Way Hybrid Similarity Learning & Fusion
   * precomputed Distance Space Mapping (SVM & KNN)
2. **Software Architecture & Codebase Guide**
   * Workspace Module Map
   * colab Execution Pipelines
3. **Experimental Setup & Protocol Parity**
   * Evaluated Cohort Characteristics (20 Datasets)
   * Leakage-Free Nested Cross-Validation (5-Fold Outer / 3-Fold Inner)
4. **Empirical Results & Statistical Significance**
   * Master GPU Benchmark results Table
   * Demšar Critical Difference (CD) analysis
   * Wilcoxon Signed-Rank Significance Report (10D vs. 3D)
   * Leave-One-Feature-Out (LOFO) Aggregated Feature Impact (Energy Significance)
   * Domain-Specific Weight Preferences
5. **Runtime Complexity & Honesty Report**
   * Theoretical Scaling Calculations ($N_{\text{total}}$ pairwise DTWs)
   * CPU-Bound Bottlenecks and Future CUDA Optimizations
6. **Publication & Presentation Visualizations**

---

## 1. Core Methodology & Mathematical Formulations

The framework abstracts raw time series sequences $X = \{x_1, x_2, \dots, x_N\}$ into human-interpretable interval-trend structures (granules) and learns similarity weights to classify them.

### A. Window Segmentation Strategies
Before granulation, the series must be partitioned into segments. We implement a **Complexity-Normalized Same-Budget Strategy Selector** to dynamically route datasets to the optimal windowing strategy:

1. **Bottom-Up Change Point Detection (CPD)**: Used for volatile, phase-shifted signals to capture transient regime shifts. CPD locates boundaries by minimizing the global least-squares error of local linear fits subject to a segment count penalty:
   $$\min \sum_{j=1}^{S} \text{Cost}(X[t_{j-1} : t_j]) + \beta \cdot S$$
   Where $\text{Cost}(X[t_a : t_b]) = \sum_{t=t_a}^{t_b} \left( x_t - T(t) \right)^2$ is the sum of squared residuals of an OLS regression fit. The penalty $\beta \in [1.5, 4.0]$ controls segment budget.
2. **Fixed-Window Partitioning**: Used for stationary, phase-aligned signals where matching patterns occur at fixed index offsets:
   $$t_j = j \cdot \left\lfloor \frac{N}{S} \right\rfloor, \quad j = 1, \dots, S$$

> [!NOTE]
> **Strategy Selector Logic**: The router evaluates training split series using a lag-1 autocorrelation variance heuristic:
> $$\sigma^2_{\text{lag-1}} = \text{Var}\left(\{ \rho(X_i, 1) \}_{i=1}^{M_{\text{tr}}}\right)$$
> If $\sigma^2_{\text{lag-1}} > 0.05$ (indicating high sequence shape volatility across samples), the dataset is routed to **CPD Variable Windowing**; otherwise, it is routed to **Fixed Windowing**.

---

### B. Granule Construction & 10D Representation
Within each segment $j$ defined by bounds $[t_{j-1}, t_j]$, we fit a least-squares linear trend $T_j(t) = a_j \cdot t + b_j$. We calculate the standard deviation of residuals $\sigma_j$:
$$\sigma_j = \sqrt{\frac{1}{t_j - t_{j-1}} \sum_{t=t_{j-1}}^{t_j} \left( x_t - T_j(t) \right)^2}$$
Fuzzy lower ($L_j$) and upper ($U_j$) envelopes are constructed using a spread factor $z \in \{1.0, 1.96\}$:
$$L_j(t) = T_j(t) - z \cdot \sigma_j, \quad U_j(t) = T_j(t) + z \cdot \sigma_j$$

To prevent information loss, each granule $g_j$ is represented as a **10-dimensional structural-statistical feature vector** $\mathbf{f}_j$:
1. **Lower Bound ($L_j$)**: Mean lower boundary of the fuzzy spread.
2. **Upper Bound ($U_j$)**: Mean upper boundary of the fuzzy spread.
3. **Slope ($a_j$)**: OLS regression slope representing local trend.
4. **Shannon Entropy**: Quantifies internal segment complexity.
5. **Variance ($\sigma_j^2$)**: Variance of raw values around the segment mean.
6. **Volatility ($V_{ol}$)**: Mean absolute first-difference of segment values.
7. **Curvature ($c_j$)**: Second-order polynomial term representing acceleration.
8. **Intercept ($b_j$)**: Absolute OLS level height.
9. **Energy**: Root Mean Square (RMS) of segment amplitude.
10. **Skewness**: Measure of asymmetry of residuals around the trend.

---

### C. 3-Way Hybrid Distance Fusion
Given two granulated sequences $P = \{p_1, \dots, p_{S_P}\}$ and $Q = \{q_1, \dots, q_{S_Q}\}$, similarity is calculated across three orthogonal dimensions:
1. **Overlap Distance ($D_H$)**: Set overlap calculated via interval Hausdorff distance:
   $$d_H(p_j, q_k) = \max \left( |L_P - L_Q|, |U_P - U_Q| \right)$$
2. **Phase Alignment Distance ($D_{DTW}$)**: 1D Dynamic Time Warping over granule trend slopes ($a_j$) to align local trend rates.
3. **Directional Distance ($D_{Cos}$)**: 10D Dynamic Time Warping over the full feature granule space utilizing a Cosine local metric:
   $$\text{dist}_{\text{Cos}}(\mathbf{f}_P, \mathbf{f}_Q) = 1 - \frac{\mathbf{f}_P \cdot \mathbf{f}_Q}{\|\mathbf{f}_P\| \|\mathbf{f}_Q\|}$$

To prevent test data leakage, pairwise matrices are min-max normalized using training split bounds:
$$\overline{D} = \frac{D - \min(D_{\text{Train}})}{\max(D_{\text{Train}}) - \min(D_{\text{Train}})}$$
The normalized matrices are linearly fused ($w_H + w_{DTW} + w_{Cos} = 1.0$):
$$D_{\text{Fused}} = w_H \cdot \overline{D}_H + w_{DTW} \cdot \overline{D}_{DTW} + w_{Cos} \cdot \overline{D}_{Cos}$$

---

### D. precomputed Distance Space Mapping
The fused distance matrix $D_{\text{Fused}} = \text{Fused}$ is mapped for classification:
* **Custom Distance KNN**: Pairwise neighbor voting computed directly on $D_{\text{Fused}}$.
* **Precomputed Kernel SVM**: Converted to a radial-basis kernel matrix:
  $$K(P, Q) = \exp\left( -\gamma \cdot d_{\text{Fused}}(P, Q)^2 \right), \quad \gamma = \frac{1}{2 \cdot \text{median}(D_{\text{Fused, Train}})^2}$$

---

## 2. Software Architecture & Codebase Guide

The project structure is organized as follows:
* **[`benchmark_pipeline.py`](file:///Users/adarshfulzele/Desktop/RP/Best%20A/benchmark_pipeline.py)**: The canonical script containing the core pipeline functions, custom classifiers, result compilation, and CD diagram generation.
* **[`LFIG_Adaptive_Pipeline_Colab_GPU.ipynb`](file:///Users/adarshfulzele/Desktop/RP/Best%20A/LFIG_Adaptive_Pipeline_Colab_GPU.ipynb)**: The Jupyter notebook containing the actual execution cells for the GPU-based Colab run.
* **[`Conference Paper/paper.tex`](file:///Users/adarshfulzele/Desktop/RP/Best%20A/Conference%20Paper/paper.tex)**: The LaTeX conference paper draft, fully updated with the final GPU run results.
* **`plots/`**: Directory containing generated publication figures (`fig1_lfig_granulation.png`, `fig2_segmentation_comparison.png`, `fig3_lofo_importance.png`, `fig4_fusion_weights.png`, `architecture_diagram.jpg`, `cd_diagram.png`).

---

## 3. Experimental Setup & Protocol Parity

* **Benchmark Cohort**: 20 datasets from the UCR Time Series Archive (spanning ECG, Spectro, Image, Sensor, Simulated, and Motion domains). Runtimes for heavy datasets (`FordA`, `Yoga`, `SwedishLeaf`) were excluded to prevent session timeout during the nested cross-validation sweeps.
* **Tuning Protocol**: 5-fold outer cross-validation for reporting, and 3-fold inner cross-validation for hyperparameter tuning. All parameters ($z$, $k$, weights $w$, and classifier type) are tuned per-fold using training splits only, preventing selection leakage.
* **Baselines**: DTW-1NN, ROCKET, and MiniROCKET reproduced under identical conditions using the `aeon` library.

---

## 4. Empirical Results & Statistical Significance

### A. Master GPU Benchmark Results Table

Below is the complete results compiled from the Colab GPU run (comparing single-split test accuracy `10D_Acc`, Nested CV accuracy, and reproduced baseline classifiers):

| Dataset | Proposed (10D Acc) | Proposed (Nested CV) | DTW-1NN | ROCKET | MiniROCKET | Best Baseline | Gap (vs 10D) | Fold Strategies Selected |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :---: | :--- |
| **GunPoint** | 0.8533 | 0.9300 ±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±± 0.0000 | 1.0000 | 1.0000 | 1.0000 | DTW-1NN (1.0000) | +0.0714 | fixed, fixed, fixed, fixed, fixed |
| **ArrowHead** | 0.7086 | 0.8910 ±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±± 0.0671 | 0.7700 | 0.9200 | 0.9100 | ROCKET (0.9200) | +0.2000 | cpd, cpd, cpd, cpd, cpd |
| **Chinatown** | 0.9184 | 0.9807 ±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±± 0.0159 | 0.9504 | 0.9699 | 0.9640 | ROCKET (0.9699) | +0.0836 | cpd, cpd, cpd, cpd, cpd |
| **SonyAIBORobot...** | 0.7837 | 0.9742 ±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±± 0.0034 | 0.9043 | 0.9991 | 0.9982 | ROCKET (0.9991) | +0.3213 | fixed, cpd, cpd, fixed, fixed |
| **ECGFiveDays** | 0.7793 | 0.9819 ±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±± 0.0138 | 0.8347 | 0.9153 | 0.9257 | MiniROCKET (0.9257) | +0.0655 | fixed, fixed, fixed, fixed, fixed |
| **Beef** | 0.4000 | 0.3667 ±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±± 0.0745 | 0.8333 | 0.9333 | 0.9333 | ROCKET/Mini (0.9333) | +0.1000 | cpd, cpd, cpd, cpd, cpd |
| **Meat** | 0.6833 | 0.8917 ±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±± 0.0935 | 0.7000 | 0.9000 | 0.9000 | ROCKET/Mini (0.9000) | +0.3000 | fixed, fixed, fixed, fixed, fixed |
| **BirdChicken** | 0.7000 | 0.8000 ±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±± 0.0363 | 0.8295 | 0.9773 | 0.9886 | MiniROCKET (0.9886) | +0.2045 | fixed, fixed, fixed, fixed, fixed |
| **SyntheticControl** | 0.9733 | 0.9933 ±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±± 0.0111 | 0.9967 | 1.0000 | 0.9989 | ROCKET (1.0000) | +0.1000 | cpd, fixed, cpd, cpd, cpd |
| **TwoPatterns** | 0.7775 | 0.7110 ±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±± 0.0038 | 0.9799 | 0.9985 | 0.9994 | MiniROCKET (0.9994) | +0.0539 | cpd, cpd, cpd, cpd, cpd |

---

### B. Demšar Critical Difference Analysis
Friedman test statistic = **45.8254** ($p = 0.000000$, highly significant).
* **Average Ranks (Lower is Better)**:
  * ROCKET: **1.5000**
  * MiniROCKET: **1.6750**
  * DTW-1NN (aeon): **3.0500**
  * Proposed LFIG (Single-Split): **3.7750**
* **Critical Difference (CD) Threshold**: **1.0488** (Proposed single-split LFIG and DTW-1NN lie within the same CD interval, indicating no statistically significant difference).

---

### C. Wilcoxon Significance Report
Comparing Proposed 10D single-split accuracies vs. 3D Standard LFIG accuracies:
* **Tied Datasets (Dropped)**: 3 (`Coffee`, `BirdChicken`, `FaceFour`)
* **Effective Sample Size ($N_{\text{eff}}$)**: 17 datasets
* **Wilcoxon Statistic**: $W_+ = 37.0$, $W_- = 116.0$
* **Two-sided p-value**: **0.061504**
* **Rank-Biserial Effect Size ($r_{\text{rb}}$)**: **-0.5163**

---

### D. Leave-One-Feature-Out (LOFO) Aggregated Feature Impact

The table below compiles feature importance across the UCR cohort. A negative delta indicates that removing the feature degraded accuracy:

| Feature | Mean Delta ($\Delta$) | 95% CI | Win / Loss / Tie | Status |
| :--- | :---: | :---: | :---: | :--- |
| **Energy** | -0.0279 | [-0.0533, -0.0024] | 13 w / 6 l / 2 t | **Useful (Significant)** |
| **Upper Bound** | -0.0091 | [-0.0455, 0.0273] | 9 w / 11 l / 1 t | No Detectable Effect |
| **Variance** | -0.0027 | [-0.0066, 0.0012] | 4 w / 1 l / 16 t | No Detectable Effect |
| **Curvature** | -0.0001 | [-0.0003, 0.0001] | 1 w / 1 l / 19 t | No Detectable Effect |
| **Volatility** | 0.0001 | [-0.0012, 0.0013] | 5 w / 3 l / 13 t | No Detectable Effect |
| **Intercept** | 0.0028 | [-0.0040, 0.0097] | 2 w / 12 l / 7 t | No Detectable Effect |
| **Skewness** | 0.0039 | [-0.0027, 0.0105] | 7 w / 8 l / 6 t | No Detectable Effect |
| **Shannon Entropy** | 0.0054 | [-0.0141, 0.0248] | 9 w / 9 l / 3 t | No Detectable Effect |
| **Lower Bound** | 0.0110 | [-0.0163, 0.0382] | 5 w / 12 l / 4 t | No Detectable Effect |
| **Trend Slope** | 0.0243 | [-0.0052, 0.0537] | 8 w / 12 l / 1 t | No Detectable Effect |

> [!IMPORTANT]
> **Key Interpretability Discovery**: **`Energy`** is the only globally statistically significant feature descriptor. Its 95% CI bounds lie entirely in the negative direction, confirming that removing `Energy` consistently degrades sequence generalizability.

---

### E. Domain-Specific Weight Preferences
Learned distance matrix fusion averages grouped by dataset category:

| Domain | n | Avg $w_{\text{Hausdorff}}$ | Avg $w_{\text{DTW}}$ | Avg $w_{\text{Cosine\_DTW}}$ | Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Motion** | 2 | 0.1000 | 0.1000 | **0.8000** | Preliminary (n < 5) |
| **Spectro** | 4 | 0.1000 | **0.6250** | 0.2750 | Preliminary (n < 5) |
| **Image** | 5 | **0.5000** | 0.3200 | 0.1800 | Robust (n >= 5) |
| **ECG** | 3 | 0.1767 | 0.4133 | 0.4100 | Preliminary (n < 5) |
| **Sensor** | 6 | 0.2167 | 0.3667 | **0.4167** | Robust (n >= 5) |
| **Simulated** | 3 | 0.1333 | **0.5000** | 0.3667 | Preliminary (n < 5) |

---

## 5. Runtime Complexity & Honesty Report

The primary computational overhead is $O(M^2)$ pairwise Dynamic Time Warping (DTW) computations on CPU-bound Python interpreter threads:

$$N_{\text{total}} = \frac{M_{\text{tr}}(M_{\text{tr}} - 1)}{2} + (M_{\text{te}} \times M_{\text{tr}})$$

| Dataset | Train ($M_{\text{tr}}$) | Test ($M_{\text{te}}$) | Total DTW Computations ($N_{\text{total}}$) | Measured Runtime (s) |
| :--- | :---: | :---: | :---: | :---: |
| **Wafer** | 1,000 | 6,164 | **6.66 Million** | **15,986.3s** (~4.44 Hours) |
| **TwoPatterns** | 1,000 | 4,000 | **4.50 Million** | **43,516.4s** (~12.08 Hours) |
| **SonyAIBORobot...**| 20 | 601 | **0.012 Million** | **712.4s** (~11.8 Minutes) |
| **Coffee** | 28 | 28 | **0.001 Million** | **109.2s** (~1.8 Minutes) |

### Future Work Optimizations
1. **CUDA-Compiled DTW**: Implementing custom parallel DTW kernels in JAX/PyTorch can parallelize the outer loops on the GPU, yielding an estimated **50x to 100x speedup**, bringing runtimes down from hours to minutes.
