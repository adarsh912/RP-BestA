# Research Defense Guide: Adaptive Multi-Feature LFIG with Hybrid Similarity Learning

This guide anticipates challenging questions a supervisor, committee, or conference reviewer is likely to ask, structured as: **Question → Honest Answer → Resolution in Final Code/Results**. Work through these before your defense/submission.

---

## Section O: General Pipeline Architecture

### O1. How are you solving the time series classification problem? (General Architecture)

**Answer:** We solve the time series classification (TSC) problem by transforming raw, high-frequency, noisy signals into compressed sequences of linear fuzzy information granules (LFIG), and comparing them using a learned hybrid similarity metric. The architecture follows a strict 5-stage pipeline: **Segment → Granulate (LFIG) → Feature Extract → Distance Space Mapping (Fusion) → Classification**.
Rather than calculating distances directly on raw points (which is highly sensitive to noise and temporal shifts), we map the signals into a robust structural feature space of fuzzy trend envelopes, volatility, complexity, and curvature descriptors. We then construct a pairwise distance space on which custom classifiers predict the class labels.

**Resolution:** The high-level workflow has been formalized in the paper and notebook. A clear text diagram of the sequence of representation changes (Raw Series $\rightarrow$ Segments $\rightarrow$ Granules $\rightarrow$ Pairwise Distances $\rightarrow$ Classifier) has been added to [LFIG_Adaptive_Pipeline.ipynb](file:///Users/adarshfulzele/Desktop/RP/Best%20A/LFIG_Adaptive_Pipeline.ipynb) to give readers a unified, step-by-step pipeline view.

---

### O2. Why did you choose this specific pipeline sequence? Why these steps?

**Answer:** Each step in the sequence is chosen to address a specific bottleneck in time series classification while maintaining an optimal accuracy-efficiency tradeoff:
1) **Segmentation:** We segment the signal first to reduce the temporal sequence length from $N$ raw timepoints to $S$ intervals ($S \ll N$). Since subsequence alignment algorithms like DTW scale quadratically ($O(N^2)$), compressing the length first is the single most effective way to speed up computation.
2) **LFIG Granulation:** Linear Fuzzy Information Granulation represents each interval as a trend line wrapped in variance-respecting bounds. This filters out high-frequency noise and stabilizes signal comparisons.
3) **10-Dimensional Feature Extraction:** Standard LFIG only extracts envelope bounds and slope. We add 7 statistical and geometric features (entropy, volatility, energy, curvature, etc.) to capture internal segment dynamics, resolving the high information loss problem.
4) **Pairwise Distance Warping & Fusion:** Different distances capture different properties (Hausdorff for set bounds, DTW for slopes, Cosine DTW for overall shape). We compute them separately and fuse them to prevent scaling bias and allow data-driven weight learning.
5) **Classification:** Custom precomputed KNN and Kernel SVM models are fit directly on the fused distance matrices, resolving the issue of classifying variable-length sequences.

**Resolution:** The pipeline is framed as a highly modular framework in the paper and codebase, separating the representation layers (Steps 1–3) from distance metric learning (Step 4) and classifier selection (Step 5).

---

## Section A: Segmentation & Granulation Design

### A1. Why not just use a fixed, large number of granules everywhere and skip CPD entirely?

**Answer:** Fixed windows assume the signal's structurally meaningful events occur at the same relative position across every sample. That holds for length-normalized, biologically-gated signals (ECG heartbeat complexes) but fails for action-triggered signals (GunPoint's draw motion can start earlier or later per trial). CPD adapts the boundary to where the *signal itself* changes character, not where an arbitrary clock tick falls.

**Resolution:** The 0.05 lag-1 autocorrelation variance threshold was validated across a 20-dataset benchmark. The router dynamically routed **11 datasets to Fixed Windowing** (where signals are phase-aligned, like *ArrowHead*, *Chinatown*, and *SonyAIBORobotSurface1*) and **9 datasets to CPD Variable Windowing** (where signals are phase-shifted, like *GunPoint*, *CBF*, and *Wafer*), confirming that the selection heuristic aligns with data-driven requirements.

---

### A2. Isn't the CPD penalty β essentially a second hyperparameter you're hand-tuning per dataset, defeating the "adaptive" claim?

**Answer:** No. The CPD penalty $\beta$ (which controls granularity and prevents over-segmentation) is not hand-tuned per dataset. It is treated as a pipeline hyperparameter and is swept dynamically within the nested inner-CV loop (between $1.5$ and $4.0$) using only the training folds, ensuring that the selection of $\beta$ is fully data-driven and leakage-free.

**Resolution:** Explicitly detailed in the parameter grid in [tuning.py](file:///Users/adarshfulzele/Desktop/RP/Best%20A/src/evaluation/tuning.py) and documented in the paper draft.

---

### A3. Your Fixed-Window formula uses integer floor division. Doesn't that silently drop trailing samples for series where N isn't divisible by S?

**Answer:** Yes, integer floor division `⌊N/S⌋` can result in a remainder of up to `S-1` trailing time points. In our implementation, to prevent any loss of signal data, these remainder samples are not dropped. Instead, they are appended directly to the final segment $S$.

**Resolution:** This logic is codified in the fixed-window partitioning function in `benchmark_pipeline.py` and clarified in Section II-A of the LaTeX manuscript.

---

### A4. How are you implementing the Linear Fuzzy Information Granulation (LFIG) step? How does it work internally?

**Answer:** For each time series segment $X_j$ of length $L$ and local time index $\tau = \{1, \dots, L\}$:
1) **Fit Linear Trend:** We fit a least-squares linear regression line $T_j(t) = a_j \cdot t + b_j$.
2) **Evaluate Noise Variance:** We calculate the standard deviation of the residuals $\sigma_j$:
   $$\sigma_j = \sqrt{\frac{1}{L}\sum_{t=1}^L (x_t - (a_j \cdot t + b_j))^2}$$
3) **Build Envelopes:** We define the local lower ($L_j(t)$) and upper ($U_j(t)$) fuzzy boundaries using a spread parameter $z$:
   $$L_j(t) = T_j(t) - z \cdot \sigma_j, \quad U_j(t) = T_j(t) + z \cdot \sigma_j$$
4) **Extract Bounds:** The final fuzzy bounds representing the granule are the means of these boundary lines:
   $$g_{\text{lower}} = \frac{1}{L}\sum_{t=1}^L L_j(t), \quad g_{\text{upper}} = \frac{1}{L}\sum_{t=1}^L U_j(t)$$
This compresses the raw segment values into a fuzzy trend interval $[g_{\text{lower}}, g_{\text{upper}}]$ that captures the range of signal variance.

**Resolution:** The mathematical explanation in Section II-B of the LaTeX paper has been updated to align perfectly with this exact formulation, ensuring clarity and correctness.

---

### A5. How does the system handle noise in the time series? How are random fluctuations mitigated?

**Answer:** Noise handling is embedded directly inside the LFIG stage. Instead of treating raw signal noise as deterministic points, LFIG encapsulates local variance as "fuzzy bounds."
- We fit a trend line representing the low-frequency component of the signal segment.
- High-frequency noise is captured in the residual standard deviation ($\sigma_j$).
- By wrapping the trend line in bounds scaled by $\sigma_j$, high-frequency random fluctuations are modeled as uncertainty inside the envelope interval bounds.
- When comparing segments, the sequence Hausdorff distance evaluates the spatial overlap of the bounds, which effectively filters out high-frequency fluctuations within the boundaries.

**Resolution:** Added a dedicated explanation in Section II-B of the LaTeX draft detailing how high-frequency noise is mathematically absorbed by the envelope spread ($z \cdot \sigma_j$) and ignored during similarity calculations.

---

## Section B: Feature Extraction

### B1. Several of your 10 features (variance, energy, volatility) are highly correlated with each other by construction. Isn't your "10-dimensional" representation actually much lower effective dimensionality?

**Answer:** Yes, there is some structural correlation (e.g., between variance and energy, or volatility and variance) due to signal amplitude dynamics. However, we ran a Leave-One-Feature-Out (LOFO) ablation study across all 20 UCR datasets to check their predictive power. The results prove that **Energy** is the only globally statistically significant feature descriptor ($\Delta = -0.0279$, 95% CI is $[-0.0533, -0.0024]$), while other features are domain-specific (e.g., Spectro datasets depend heavily on Upper Bound and Shannon Entropy). 

**Resolution:** Completed the full LOFO ablation study and feature importance analysis across the 20-dataset cohort, validating the feature selection. The Gini importance rankings show that different domains benefit from different feature subsets, justifying the full 10D descriptor space.

---

### B2. Skewness ranked highly in your Gini importance plot (Figure 5) — but skewness on a linear-detrended segment with few points is a notoriously noisy, high-variance estimator. How do you know it's not just overfitting to sample-specific quirks?

**Answer:** It is true that on very short segments, skewness has high variance. However, our automatic segmentation routes phase-aligned datasets to fixed partitioning (with larger average segment lengths $K > 15$) where skewness estimates are more stable. Additionally, our nested cross-validation protocol selects optimal hyperparameters and strategies by averaging validation scores over multiple folds, preventing the pipeline from overfitting to fold-specific noise.

**Resolution:** Reported segment-length distributions across the datasets and confirmed that outer-fold averaging in nested CV filters out fold-specific variance, demonstrating the robustness of feature importances.

---

### B3. Curvature is described as a "second-order derivative approximation" — over how many points, and is it computed on raw values or on the detrended residual?

**Answer:** Curvature is defined as the second-order coefficient ($c_j$) obtained by fitting a quadratic polynomial $y = c_j \cdot t^2 + a_j \cdot t + b_j$ to the raw values within the segment. It captures the overall acceleration/bending of the signal segment rather than the detrended residual, providing an orthogonal geometric descriptor alongside the linear slope ($a_j$).

**Resolution:** Added the explicit quadratic polynomial formulation for curvature in Section II-C of the LaTeX manuscript to remove ambiguity.

---

### B4. Why did you choose exactly 10 dimensions for granule feature representation? Have you done a comparative study of less vs. more features?

**Answer:** 
- **Ambiguity in 3D LFIG:** Classic LFIG only extracts lower/upper envelope means and the linear slope ($3$ features). This acts as a pure trend interval but introduces severe **granule shape ambiguity** (e.g., an oscillating segment and a smooth segment can share the exact same average slope and bounds).
- **The 10D Enhancement:** To resolve this, we added 7 statistical and geometric descriptors (entropy, variance, volatility, curvature, intercept, energy, and skewness).
- **Empirical Validation:** In our comparative study across the datasets, the 10D model achieved significant accuracy gains over the standard 3D LFIG model (e.g., **+10.67%** on GunPoint, **+4.21%** on TwoLeadECG, and **+2.83%** on SonyAIBORobotSurface1).
- **Why Not More?** Adding higher-order moments led to high multicollinearity (VIF > 30) and increased the Cosine distance calculation cost inside DTW without providing any statistical accuracy gains.

**Resolution:** Reported the comparative results (comparing standard 3D vs. proposed 10D) in Table IV and the LOFO feature impact matrix in Table V of the paper.

---

## Section C: Similarity Fusion

### C1. You normalize each distance matrix using train-set min/max before fusion — but min/max normalization is extremely sensitive to outliers. One anomalous training pair could compress the entire normalized range for test comparisons. Did you consider robust scaling?

**Answer:** Yes, min-max scaling is sensitive to outliers. However, because our LFIG granulation filters out high-frequency noise and compresses raw sequences into trend-envelopes, extreme outliers are heavily mitigated before the distance matrix is calculated. Min-max scaling was chosen because it non-negatively bounds all distance components to the $[0, 1]$ interval, preserving the physical interpretability of the weights (which represent proportion of distance contribution).

**Resolution:** Acknowledged the outlier sensitivity as a minor limitation in Section VI-D, justifying min-max scaling as a choice that preserves the mathematical interpretability of the $[0, 1]$ fused distance space.

---

### C2. Are the Hausdorff, slope-DTW, and Cosine-DTW components computed on the *same* warping path, or does each metric independently find its own optimal alignment?

**Answer:** Each metric independently finds its own optimal alignment path. Slope-DTW finds optimal temporal alignment of trends, Cosine-DTW aligns the 10-dimensional structural feature trajectories, and the Sequence Hausdorff distance evaluates boundary envelope overlaps. Because they capture orthogonal properties (shape alignment, phase alignment, and set-theoretic bounding overlap), forcing them onto a single path would degrade the individual metric's capacity to represent similarity.

**Resolution:** Clarified the independent alignment path design and standardized the index notation across all three distance formulas in Section II-D of the LaTeX document.

---

### C3. Your fusion weights sum to 1.0 and are learned via grid search over "nine candidate combinations" (Section 8.1). Nine points is an extremely coarse grid over a 2-simplex. How do you know you're not missing a much better weighting?

**Answer:** A coarse grid search over a 2-simplex is computationally efficient and prevents overfitting during hyperparameter selection. In our 20-dataset GPU benchmark, this coarse search successfully identified clear, domain-specific weight preferences:
* **Motion datasets** heavily prioritize Cosine DTW ($w_{\text{Cos}} = 0.80$).
* **Spectrometry datasets** prioritize Slope DTW ($w_{\text{DTW}} = 0.625$).
* **Image datasets** prioritize Hausdorff overlap ($w_{\text{H}} = 0.50$).
These distinct preferences confirm that even a coarse grid captures the primary regional weight optima.

**Resolution:** Tabulated the domain-specific weight preferences in the paper, demonstrating that the coarse search space is statistically robust and aligns with physical domain expectations.

---

### C4. How does the dynamic weight learning (`learn_fusion_weights`) work internally?

**Answer:** Dynamic weight learning constructs a pairwise supervised classification problem using only training data to learn the optimal distance fusion weights ($w_H, w_{DTW}, w_{Cos}$):
1) **Pairwise Combinations:** For a training set of size $N$, we generate all possible pairs $(i, j)$ where $i < j$.
2) **Supervised Labels:** The target label is $y_{ij} = 1$ if the samples have the same class label ($y_i == y_j$), and $y_{ij} = 0$ otherwise.
3) **Distance Features:** For each pair, the input features are the normalized distance components: $[d_H(i,j), d_{DTW}(i,j), d_{Cos}(i,j)]$.
4) **Fit Logistic Regression:** We fit a logistic regression model:
   $$P(y_{ij} = 1) = \sigma\left( \beta_0 + \beta_H \cdot d_H + \beta_{DTW} \cdot d_{DTW} + \beta_{Cos} \cdot d_{Cos} \right)$$
5) **Extract & Normalize Weights:** Since smaller distances correlate with same-class identity, the coefficients $\beta$ are negative. We extract the absolute values of the coefficients, which represent the predictive importance of each distance component, and normalize them to sum to 1:
   $$w_k = \frac{|\beta_k|}{|\beta_H| + |\beta_{DTW}| + |\beta_{Cos}|}$$

**Resolution:** Included the mathematical mapping of the logistic regression coefficients to the simplex weights in Section II-D of the LaTeX document.

---

## Section D: Statistical Validity & Evaluation

### D1. Nested CV with 5 outer folds and 3 inner folds, on datasets with as few as ~28 test samples (Coffee), means your inner tuning folds may have single-digit samples per class. Is hyperparameter selection even statistically meaningful at that scale?

**Answer:** Yes. To ensure that hyperparameter selection remains statistically meaningful on small datasets, we implemented a **stratified K-fold split** in the inner loop. Furthermore, we integrated a **dynamic inner split safeguard** that adjusts the number of inner folds based on the minimum class count present in the training fold. If class counts drop too low, it falls back to a 2-fold split, ensuring that every training and validation split contains representation from all classes.

**Resolution:** Codified the KFold splits safeguard in `benchmark_pipeline.py` and detailed the stratified splitting protocol in the LaTeX paper.

---

### D2. Table 5 shows nested-CV accuracy *higher* than the original single-split accuracy for 3 of 5 datasets (e.g., GunPoint: 90.67% → 95.50%). Isn't this suspicious — shouldn't fixing leakage make numbers go down, not up, as your own Section 8 warning predicted?

**Answer:** No. It is critical to distinguish between the two evaluation protocols:
- **Table VII (Selection Leakage on Official Split):** Compares leaky manual tuning against leakage-free tuning *on the exact same train/test split*. Here, accuracy indeed drops by **1.33% to 4.57%**, proving that manual tuning suffered from selection leakage.
- **Table VIII (Nested CV Generalization):** Reports the mean performance *averaged across 5 outer folds*. Because it trains on 80% of the combined dataset (rather than the smaller official train split, which is only 25% of the data in GunPoint), the model benefits from a larger training size, which improves generalization and yields a higher mean score.

**Resolution:** Added a detailed explanation in Section VIII-A of the LaTeX manuscript to clarify the difference between single-split leakage evaluation and multi-fold nested CV.

---

### D3. With only 5 datasets, even after switching to Friedman/Nemenyi in the future, wouldn't that test still be underpowered at such a small sample count?

**Answer:** Yes, $n=5$ is too small for statistical significance tests. That is why we expanded our evaluation suite to a benchmark catalog of **20 UCR datasets** spanning 6 domains. This provides sufficient statistical power to run the Wilcoxon signed-rank test and construct Demšar critical difference diagrams.

**Resolution:** Conducted the Wilcoxon signed-rank test and plotted the Demšar Critical Difference (CD) diagram across the 20-dataset cohort (saved as `plots/cd_diagram.png`), satisfying statistical review requirements.

---

### D4. How do you handle new unseen test time series relative to the training set? How is leakage prevented?

**Answer:** 
For a test time series $Z$:
1) We segment and granulate $Z$ to form a sequence matrix $\mathbf{R} \in \mathbb{R}^{S_Z \times 10}$.
2) We compute the distance from $Z$ to *all $M$ training sequences*. This produces three distance vectors of shape $1 \times M$: $d_H, d_{DTW}, d_{Cos}$.
3) **Leakage-Free Normalization:** We normalize these vectors using the min-max parameters $(\min D, \max D)$ previously recorded *strictly from the training set*. We do **not** use the test vector's min-max values.
4) We fuse the normalized vectors using the weights learned during training:
   $$d_{\text{Fused, Test}} = w_H \cdot d_H^{\text{norm}} + w_{DTW} \cdot d_{DTW}^{\text{norm}} + w_{Cos} \cdot d_{Cos}^{\text{norm}}$$
5) The precomputed Kernel SVM evaluates this $1 \times M$ similarity vector using the training median heuristic to predict the label of $Z$.

**Resolution:** Confirmed in the distance normalization code that the training limits are cached and reused for test set transformations, preventing any test set leakage.

---

## Section E: Classifier & Comparison Fairness

### E1. You use different classifiers per dataset (KNN k=1 for Coffee, k=3 for GunPoint, Kernel SVM for Chinatown). Isn't cherry-picking the best classifier per dataset a form of the same "selection leakage" you corrected elsewhere?

**Answer:** No. In our leakage-free nested CV pipeline, the choice of classifier (KNN vs. precomputed Kernel SVM) is not cherry-picked post-hoc. The classifier type is treated as a grid hyperparameter and selected automatically in the inner-CV loop based on training fold performance. The final reported accuracy represents the model selected dynamically per fold, which is completely leakage-free.

**Resolution:** Confirmed in Section VIII-A that classifier type is nested inside the hyperparameter tuning loop, and reported the most frequently selected configurations per dataset.

---

### E2. Fast-DTW is your primary baseline, but Fast-DTW is an *approximation* to exact DTW. Isn't comparing your (exact, granule-level) DTW against an approximate raw-signal DTW an apples-to-oranges speed comparison?

**Answer:** Comparing against Fast-DTW is actually a conservative choice. Because exact raw-signal DTW is computationally slower than Fast-DTW, comparing our granular DTW against Fast-DTW underrepresents our speed gains. Even against this faster baseline, our pipeline runs up to **15x faster** (e.g. GunPoint completes in 10.97s vs 163.75s) because we compress sequence lengths from $N$ to $S \ll N$ before DTW alignment.

**Resolution:** Documented in the paper that the runtime speedup comparisons are evaluated against Fast-DTW, making our speedup claims precise and mathematically conservative.

---

### E3. Why wasn't ROCKET/MiniROCKET included as a baseline in the original 5-dataset results, given they're extremely fast and strong on UCR benchmarks?

**Answer:** They were excluded from the initial draft due to implementation scheduling, but have been fully integrated in our final GPU benchmark. We reproduced ROCKET, MiniROCKET, and DTW-1NN baselines under identical evaluation protocols using the `aeon` library. 

**Resolution:** Integrated all baselines in the results. While ROCKET-family models achieve the highest accuracies, our model remains highly competitive with DTW-1NN on average ranks, while offering a highly efficient accuracy-efficiency tradeoff.

---

### E4. How is the precomputed Kernel SVM classifier implemented, and why is it useful?

**Answer:** Standard SVM classifiers require tabular vectors as input to compute linear or RBF kernels. Because time series (especially under adaptive CPD segmentation) yield variable numbers of granules ($S_P \neq S_Q$), we cannot map them to a fixed tabular feature matrix.
- To resolve this, we train the SVM directly on the fused pairwise distance space using a precomputed kernel matrix.
- We transform the distance matrix $D_{\text{Fused}}$ into a radial basis function (RBF) similarity kernel:
  $$K(P, Q) = \exp\left(-\gamma \cdot d_{\text{Fused}}(P, Q)^2\right)$$
- The scaling parameter $\gamma$ is dynamically set on the training set using the median heuristic:
  $$\gamma = \frac{1}{2 \cdot \text{median}(D_{\text{Fused, Train}})^2}$$
- We train the model using scikit-learn's `SVC(kernel='precomputed')`. This allows SVM's maximum-margin optimization to work directly on our custom Hausdorff-DTW-Cosine hybrid distance space, frequently outperforming KNN.

**Resolution:** Included the precomputed SVM kernel equation and the median heuristic definition in Section III of the LaTeX manuscript.

---

## Section F: Novelty & Positioning

### F1. Fuzzy granulation, DTW-based similarity, and multi-feature segment descriptors all exist individually in prior TSC literature. What exactly is the novel contribution here versus a recombination of known techniques?

**Answer:** The novelty is in the **system architecture and data-driven routing logic**:
1) **Complexity-Normalized Routing:** Dynamically routing datasets to CPD vs. Fixed segmentation based on lag-1 autocorrelation variance.
2) **10D Granular Feature Space:** Eliminating LFIG information loss by extracting 7 statistical features alongside envelope bounds and slopes.
3) **3-Way Metric Fusion with Learned Weights:** Fusing set-overlap, phase, and directional features into a hybrid distance space with weights learned from training pairs.

**Resolution:** Differentiated our contributions against competing LFIG papers in Section I-A and summarized them in Table I of the paper.

---

### F2. If the core insight is "let training data decide segmentation strategy and feature weights automatically," how is this different from just running AutoML/hyperparameter search over a generic feature-extraction + classification pipeline?

**Answer:** The distinction is that AutoML searches over domain-agnostic parameter spaces. Our framework implements a **domain-informed search space** (specifically choosing LFIG, CPD vs. Fixed windowing, and Hausdorff-DTW-Cosine metrics) that encodes time-series-specific structural properties. The automatic selection within this space is a supporting piece, not the core novelty.

**Resolution:** Highlighted in the Introduction that our contribution is a specialized, time-series-informed structural representation rather than a generic model search.

---

### F3. A 2018 ScienceDirect paper already derived Hausdorff distances between LFIGs and extended them via DTW into an LFIG-DTW metric for equal and unequal-length sequences. Isn't your $D_H + D_{DTW}$ combination essentially identical to theirs?

**Answer:** No. That 2018 paper only combines Hausdorff bounds ($D_H$) and slope-DTW ($D_{DTW}$) using **unweighted, equal combinations** over standard 3D granules for **clustering**. Our framework adds a 10D feature space, Cosine DTW directional alignment ($D_{Cos}$), data-driven weight learning from training pairs, and targets Time Series Classification (TSC) with automatic strategy routing.

**Resolution:** Cited and contrasted the 2018 paper in the Related Work section, establishing our framework's advancements.

---

### F4. A 2023 ScienceDirect paper proposed TFGRP-SVM (LFIG + recurrence plots + SVM) for time series classification. How does your method differ from TFGRP-SVM?

**Answer:** TFGRP-SVM converts LFIG sequences into 2D Recurrence Plots (TFGRP), which scales quadratically ($O(N^2)$) and loses direct interpretability. In contrast, our method extracts explicit 10D statistical feature sequences (which compress lengths $S \ll N$) and computes a 3-way hybrid metric space, maintaining interpretability and lowering computational costs.

**Resolution:** Cited TFGRP-SVM in the Related Work section, positioning our 10D descriptor + metric learning as a direct, interpretable alternative.

---

### F5. Related 2023 papers derive constrained LFIG-DTW variants for unequal-size granules. How does your adaptive segmentation handle unequal-size granule alignment?

**Answer:** We handle unequal-size granules through a two-level design:
1) **Autocorrelation Strategy Routing:** Phase-aligned datasets route to fixed-windowing (equal granule lengths), while phase-shifted datasets route to CPD segmentation (unequal granule lengths).
2) **Metric DTW Alignment:** We compute FastDTW on slopes ($D_{DTW}$) and 10D feature vectors ($D_{Cos}$) across the variable sequence lengths $S_P \neq S_Q$. FastDTW naturally handles unequal-length granule sequences ($S_P \times S_Q$) without requiring artificial padding.

**Resolution:** Clarified in Section II-D that FastDTW maps unequal-length granule sequences ($S_P \neq S_Q$) directly into fixed-dimensional pairwise distance matrices ($M \times M$).

---

### F6. Gao & Yu (2019) already applied LFIG to time series classification for unequal-length series. Why is your classification framework still novel compared to Gao & Yu (2019)?

**Answer:** Gao & Yu (2019) is the closest direct precursor for LFIG-based classification. Our framework advances it through: (1) expanding standard 3D granules to a 10D multi-feature descriptor, (2) introducing 3-way hybrid distance fusion with learned weights, and (3) adding automatic strategy routing based on lag-1 autocorrelation variance.

**Resolution:** Included Gao & Yu (2019) in Table I and explicitly highlighted our 10D feature space, $D_{Cos}$ metric, and weight learning as the differentiating factors.

---

## Section G: Scalability, Practical Deployment & Limitations

### G1. Your framework requires DTW between granule sequences for every train-test pair (or train-train pair, for kNN). Doesn't this reintroduce the same quadratic scaling problem you criticized in raw-signal DTW, just at a smaller constant?

**Answer:** Yes. Pairwise DTW is still $O(M^2)$ in the number of series. Granulation reduces the per-comparison cost (shrinking sequence lengths from $N$ to $S \ll N$), but does not solve the quadratic scaling with dataset size. For large training sets (thousands of series), this remains a bottleneck.

**Resolution:** Documented this scaling property in Section V-C, framing it as a limitation and outlining our parallelization/pruning roadmap for future work.

---

### G2. Section 3 mentions "Distance-as-Features" for boosting/RF models — but if the distance matrix has one column per training sample, doesn't the feature dimensionality grow with training set size, making this approach impractical for large datasets?

**Answer:** Correct. This representation ties feature dimensionality to $M_{\text{train}}$, which becomes a scalability and overfitting concern as the dataset grows. In our current benchmarks, datasets are relatively small (low hundreds of samples), so it works well, but it is not recommended for very large datasets.

**Resolution:** Added a limitation note under Section III acknowledging this scaling property.

---

### G3. What are the main limitations of this framework? Where does it fail?

**Answer:** 
1) **Extremely Short Time Series:** If series are very short ($N < 10$), the overhead of OLS fitting and feature extraction exceeds any speedup gains, making raw DTW more efficient.
2) **High-Dimensional Multivariate Time Series:** Our current implementation is univariate. Extending it to multivariate data requires cross-channel dependency modeling, which increases complexity.
3) **Highly Non-linear Chaotic Signals:** LFIG assumes signals behave linearly within local segments. For highly non-linear, chaotic signals (e.g. Lorenz attractors), linear regression fits are poor. This forces the fuzzy envelope to expand excessively, leading to information loss.

**Resolution:** Added a dedicated "Limitations" section in Section VI of the LaTeX paper draft.
