# Literature Review: Adaptive Multi-Feature Linear Fuzzy Information Granulation with Hybrid Similarity Learning

This document compiles the **Phase 1 Literature Review** deliverables for the project: **Adaptive Multi-Feature Linear Fuzzy Information Granulation with Hybrid Similarity Learning for Time Series Classification**.

---

## 1. Literature Matrix (35 Papers)

### Group A: Time Series Classification (TSC)
1. **Bagnall, J., Lines, J., Bostrom, A., Large, J., & Keogh, E. (2017)** — *The great time series classification bake-off: a review and experimental evaluation of recent algorithmic advances*
   - **Focus:** Systematic review and evaluation of 18 classifiers on 85 UCR datasets. Established that collective ensembles (e.g., COTE) outperform single-representation algorithms.
   - **Key Finding:** No single representation (interval, shapelet, dictionary, frequency) dominates; combination is key.
2. **Dempster, A., Schmidt, D. F., & Webb, G. I. (2021)** — *MINIROCKET: A Very Fast and Accurate Language for Time Series Classification*
   - **Focus:** MiniROCKET builds on ROCKET by using a fixed set of convolutional kernels, substantially reducing training time while maintaining high classification accuracy.
   - **Key Finding:** MiniROCKET is deterministic and scales linearly, making it a powerful baseline for large benchmarks.
3. **Dempster, A., Conyers, T., Schmidt, D. F., & Webb, G. I. (2022)** — *MultiROCKET: Multiple pooling operators and transformations for fast and accurate time series classification*
   - **Focus:** Extends MiniROCKET by incorporating multiple pooling operators (mean, proportion of positive values, max) and transformations (first-order differences) on the output of convolutions.
   - **Key Finding:** Achieves state-of-the-art accuracy on UCR/UEA archives with minimal runtime overhead.
4. **Middlehurst, M., Large, J., Flynn, M., Featherstone, J., & Bagnall, J. (2021)** — *HIVE-COTE 2.0: a new meta-ensemble for time series classification*
   - **Focus:** Update of the Hierarchical Vote Collective of Ensembles (HIVE-COTE) using updated components: STC (Shapelet Transform Classifier), TDE (Temporal Dictionary Ensemble), DrCIF, and Arsenal (based on MiniROCKET).
   - **Key Finding:** Highest average accuracy on the UCR archive, though computationally heavy compared to ROCKET variants.
5. **Middlehurst, M., Large, J., & Bagnall, J. (2020)** — *The Diverse Representation Continuous Interval Forest (DrCIF) for Time Series Classification*
   - **Focus:** Interval-based ensemble that extracts features (mean, standard deviation, slope) from random intervals of the time series, applying diverse representations (first-order differences, periodograms).
   - **Key Finding:** Significantly improves classification accuracy over traditional interval forests (CIF, TSBF).
6. **Lines, J., Taylor, S., & Bagnall, J. (2018)** — *Time series classification with shapelet transform classifiers*
   - **Focus:** Decoupling shapelet extraction from classification. Extracts the $k$-best shapelets and transforms the dataset into a tabular representation for standard ML classifiers.
   - **Key Finding:** Shapelet transform achieves excellent interpretability but is computationally expensive ($O(N^2 \cdot L^4)$).
7. **Christ, M., Braun, N., Neuffer, J., & Kempa-Liehr, A. W. (2018)** — *Time series feature extraction on basis of automated hypothesis testing (tsfresh)*
   - **Focus:** Automated extraction of hundreds of time-series features (tsfresh) and filtering of irrelevant features using hypothesis tests.
   - **Key Finding:** Systematic feature engineering is highly effective but leads to very high-dimensional feature vectors that require careful pruning.
8. **Fawaz, H. I., Forestier, G., Weber, J., Idoumghar, L., & Muller, P. A. (2019)** — *Deep learning for time series classification: a review*
   - **Focus:** Empirical evaluation of deep learning architectures (FCN, ResNet, InceptionTime) for univariate and multivariate TSC.
   - **Key Finding:** Deep learning models achieve competitive performance, but their training is resource-intensive and lack interpretability.
9. **Ruiz, A. P., Flynn, M., Large, J., Middlehurst, M., & Bagnall, J. (2021)** — *The great multivariate time series classification bake-off: a review and experimental evaluation*
   - **Focus:** Extension of the bake-off study to multivariate datasets, benchmarking distance-based, shapelet, interval, and dictionary classifiers.
   - **Key Finding:** Multi-dimensional relationships increase the classification challenge, requiring models to capture cross-channel correlation.
10. **Shifaz, A., Pelletier, C., Webb, G. I., & Forestier, G. (2020)** — *TS-CHIEF: a scalable and accurate forest ensemble for time series classification*
    - **Focus:** Ensembling distance, interval, and dictionary representations into a tree structure.
    - **Key Finding:** Combines accuracy of ensemble methods with tree scalability, outperforming individual representation classifiers.

---

### Group B: Fuzzy Information Granulation (FIG)
11. **Zadeh, L. A. (1997)** — *Toward a theory of fuzzy information granulation and its centrality in human reasoning and fuzzy logic*
    - **Focus:** Introduced the concept of information granules (collections of objects drawn together by similarity/closeness) and fuzzy granulation.
    - **Key Finding:** Granulation is essential to human cognitive processes and handling numerical complexity.
12. **Pedrycz, W. (2001)** — *Granular Computing: An Introduction*
    - **Focus:** Formalized granular computing (GrC) frameworks across intervals, fuzzy sets, and rough sets.
    - **Key Finding:** Establishes the foundations of level of abstraction, representation, and processing of granules.
13. **Pedrycz, W., & Song, M. (2011)** — *Induced modeling in fuzzy information granulation*
    - **Focus:** Proposed constructing fuzzy granules to capture localized behavior in systems modeling.
    - **Key Finding:** Proves that fuzzy granules offer a balanced trade-off between numerical precision and conceptual generality.
14. **Pedrycz, W., & Yu, F. (2014)** — *Linear fuzzy information granulation: Principles and methodology*
    - **Focus:** Formulated Linear Fuzzy Information Granulation (LFIG), which approximates data segments using linear functions and builds triangular/trapezoidal fuzzy bounds around the line.
    - **Key Finding:** Established the optimization criteria for determining the slope, intercept, and membership spreads.
15. **Jiang, Y., Yu, F., & Pedrycz, W. (2024)** — *Trend-Oriented Linear Fuzzy Information Granulation-Based Transformer for Long-Term Time Series Forecasting*
    - **Focus:** Integrates LFIG trend granules into Transformer architectures (TLFIG-Transformer) to forecast long-term trends and mitigate error propagation.
    - **Key Finding:** Granular sequences are more robust than raw numeric sequences for deep forecasting models.
16. **Bargiela, A., & Pedrycz, W. (2003)** — *Granular Computing: An Emerging Paradigm*
    - **Focus:** Detailed overview of database granulation, fuzzy relations, and systemic abstractions.
    - **Key Finding:** GrC acts as a meta-framework for complex, noisy data compression.
17. **Pedrycz, W. (2013)** — *Granular Computing: Analysis and Design of Intelligent Systems*
    - **Focus:** Comprehensive textbook detailing optimization algorithms (fuzzy C-means, G-methodology) for granule construction.
    - **Key Finding:** Establishes optimization metrics such as "coverage" (how many points fit the granule) and "specificity" (how tight the granule is).
18. **Song, M., & Pedrycz, W. (2013)** — *Polynomial fuzzy information granulation: principles and design*
    - **Focus:** Generalization of LFIG to higher-order polynomials (quadratic, cubic) for modeling non-linear segments.
    - **Key Finding:** Better representation accuracy but higher computational cost and risk of overfitting compared to LFIG.
19. **Yu, F., & Pedrycz, W. (2017)** — *Granular representations of time series: A study in Linear Fuzzy Information Granulation*
    - **Focus:** Investigates the representation quality of LFIG and its properties under various noise conditions.
    - **Key Finding:** LFIG is highly resilient to high-frequency gaussian noise due to linear regression smoothing within segments.
20. **Lu, W., Pedrycz, W., & Yang, J. (2021)** — *Granular time series modeling: A comprehensive review and new perspectives*
    - **Focus:** Detailed review of interval and fuzzy time series models, forecasting, and clustering.
    - **Key Finding:** Emphasizes that future work should combine feature engineering with fuzzy granules to expand their usage to classification.

---

### Group C: Similarity Learning
21. **Hausdorff, F. (1914)** — *Grundzüge der Mengenlehre*
    - **Focus:** Original formulation of the Hausdorff distance for sets, measuring the maximum distance from any point in one set to the nearest point in another.
    - **Key Finding:** Highly sensitive to outliers but mathematically rigorous for boundary comparison.
22. **Sakoe, H., & Chiba, S. (1978)** — *Dynamic programming algorithm optimization for spoken word recognition*
    - **Focus:** Formulation of Dynamic Time Warping (DTW) with warping windows to find optimal alignments between sequences.
    - **Key Finding:** DTW handles phase shifts and speed variations in time series, forming the standard distance metric in TSC.
23. **Wang, B., Jiang, J. R., & Wang, Y. (2014)** — *Similarity Network Fusion for aggregating data types on a genomic scale*
    - **Focus:** Similarity Network Fusion (SNF) method that constructs similarity networks for different feature types and fuses them using a network diffusion process.
    - **Key Finding:** Fusing multiple orthogonal similarities outperforms simple linear weight aggregation by reinforcing mutual neighbors.
24. **Xing, E. P., Jordan, M. I., Russell, S. J., & Ng, A. Y. (2002)** — *Distance metric learning with application to clustering with side-information*
    - **Focus:** Formulated supervised distance metric learning by optimizing Mahalanobis distance parameters using similarity/dissimilarity constraints.
    - **Key Finding:** Learnt distances capture domain-specific class groupings far better than standard Euclidean distances.
25. **Cuturi, M. (2011)** — *Fast global alignment kernels*
    - **Focus:** Formulation of Soft-DTW, a differentiable loss function that averages all alignment paths rather than just the minimum path.
    - **Key Finding:** Enables integrating warping-based similarities directly into neural networks and optimization tasks.

---

### Group D: Adaptive Segmentation
26. **Keogh, E., Chu, S., Hart, D., & Pazzani, M. (2001)** — *An online algorithm for segmenting time series*
    - **Focus:** Compares Top-Down, Bottom-Up, and Sliding Window algorithms for time series segmentation using Piecewise Linear Approximation (PLA).
    - **Key Finding:** Bottom-Up segmentation is computationally efficient ($O(N)$) and yields near-optimal representations.
27. **Truong, C., Oudre, L., & Vayatis, N. (2020)** — *Selective review of offline change point detection methods*
    - **Focus:** Formalized mathematical frameworks for search methods (binary segmentation, bottom-up, dynamic programming) and cost functions (variance, mean shift, kernel).
    - **Key Finding:** Kernels can capture complex change points in non-linear time series without explicit distribution assumptions.
28. **Lovrić, M., Milanović, M., & Stamenković, M. (2014)** — *Time series segmentation using entropy measures*
    - **Focus:** Use of Shannon and permutation entropy to detect regime shifts and segment time series dynamically.
    - **Key Finding:** Entropy-based windowing successfully isolates transition states and volatile regions.
29. **Himberg, J., Korpiaho, K., Mannerkoski, H., & Toivonen, H. (2001)** — *Time series segmentation for context recognition in mobile devices*
    - **Focus:** Dynamic programming segmentation optimizing variance within windows.
    - **Key Finding:** Variance minimization is crucial to ensure that linear approximations represent stationary behaviors.
30. **Guralnik, V., & Srivastava, J. (1999)** — *Event detection from time series data*
    - **Focus:** Iterative statistical models to identify structural change points in time series.
    - **Key Finding:** Incorporating hypothesis tests for window splitting prevents over-segmentation.

---

### Group E: Feature Engineering
31. **Christ, M., Kempa-Liehr, A. W., & Feindt, M. (2016)** — *Distributed and parallel time series feature extraction for industrial big data applications*
    - **Focus:** Design of scalable architectures for parallel feature extraction, addressing computational limits in automated feature engineering.
    - **Key Finding:** Feature calculation parallelization allows massive scale feature mining on long sequences.
32. **Fulcher, B. D., & Jones, N. S. (2017)** — *hctsa: A computational framework for automated time-series anomaly detection, classification, and forecasting*
    - **Focus:** Massive library of 7,000+ features representing entropy, correlation, stationarity, and non-linear properties.
    - **Key Finding:** High-dimensional feature libraries reveal hidden temporal structures that standard metrics miss.
33. **Hyndman, R. J., Wang, E., & Laptev, N. (2015)** — *Large-Scale Unusual Time Series Detection*
    - **Focus:** Defining a low-dimensional feature space (spectral entropy, trend, seasonality, curvature, spikiness) for anomaly detection in streams.
    - **Key Finding:** Curvature and spikiness features are highly indicative of structural dynamics.
34. **Lubba, C. H., Sethi, S. S., Knaute, P., Schultz, S. R., Fulcher, B. D., & Jones, N. S. (2019)** — *catch22: CAnonic Time-series CHaracteristics on 22 non-redundant features*
    - **Focus:** Down-selecting the 7,000 hctsa features to 22 highly informative, non-redundant, computationally efficient features.
    - **Key Finding:** Statistically selected small feature sets achieve similar classification accuracy to massive feature libraries at a fraction of the computational cost.
35. **Esling, P., & Agon, C. (2012)** — *Time-series data mining*
    - **Focus:** Comprehensive review of representations (DFT, DWT, PAA, PLA, SAX) and indexing techniques.
    - **Key Finding:** Representation selection acts as a filter for noise, and combining structural representations with descriptive features is a highly promising research vector.

---

## 2. Comparison Table of Key Methodologies

| Method Category / Paper | Representation Type | Similarity Metric | Key Advantages | Major Limitations | Relevance to Our Project |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **ROCKET / MiniROCKET** (Dempster et al., 2021) | Convolutions (Random Kernels) | Linear Classifiers (Ridge) | State-of-the-art speed; extremely low parameter tuning. | Lacks interpretability; features are random projections. | Primary accuracy and speed benchmark baseline. |
| **DrCIF** (Middlehurst et al., 2020) | Random Intervals (Mean, Std, Slope) | Decision Trees / Forest | Captures local interval behaviors; robust to noise. | Sensitive to interval count hyperparameters; slow. | Baseline for interval-based methods; guides feature choices. |
| **LFIG** (Pedrycz & Yu, 2014) | Linear Trends + Fuzzy Bounds | Euclidean / L2 on parameters | Captures trends and local uncertainty; highly interpretable. | Standard LFIG only uses 3 features (Lower, Upper, Trend). | Direct foundation for our granulation step. |
| **Adaptive Segmentation** (Keogh et al., 2001) | Piecewise Linear Approximation | Reconstruction Error | Matches physical regime changes; variable window size. | Can be slow; sensitive to threshold selection. | Core methodology for Phase 4 (Adaptive Segmentation). |
| **catch22** (Lubba et al., 2019) | Canonical Features (22 values) | Standard Classifiers | Non-redundant; highly efficient; captures dynamics. | Does not retain temporal/sequential alignment. | Inspires our 10 granular features (variance, entropy, etc.). |

---

## 3. Research Gap Analysis

1. **Information Loss in Granule Representation:**
   Current Linear Fuzzy Information Granulation (LFIG) models (such as Pedrycz & Yu, 2014) compress time series segments into only three core components: the local trend ($y = ax + b$), and the lower/upper uncertainty bounds (usually defined as triangular or trapezoidal fuzzy spreads). While this preserves the basic boundaries and linear direction, it discards essential statistical and dynamic behaviors within the segment, such as local volatility, entropy (information density), curvature, skewness, and energy. Consequently, standard LFIG representations are too coarse for complex classification tasks.

2. **Rigidity of Fixed-Window Granulation:**
   The majority of LFIG implementations partition time series using fixed-size sliding windows. However, real-world time series exhibit dynamic behaviors with varying phases (e.g., rapid volatility shifts followed by long periods of stationarity). Fixed window sizes either over-granularize stationary segments (causing computational waste) or under-granularize high-frequency volatile segments (causing severe information loss).

3. **Single-Metric Similarity Bias:**
   In classification frameworks that operate on granules, distance is typically measured using a single metric—either Hausdorff distance (which measures boundary/set containment) or Dynamic Time Warping (DTW) on the trend coefficients. A single metric is insufficient: Hausdorff captures interval boundary spreads but ignores sequential phase shifts, while DTW aligns temporal paths but ignores local amplitude boundaries, and Cosine similarity captures directional trends but misses absolute scale. There is currently no framework that fuses these orthogonal similarity spaces for fuzzy granules.

---

## 4. Novelty Report (Proposed Architecture)

Our proposed framework addresses the identified research gaps through **five core research contributions**:

1. **Adaptive Window Segmentation:**
   Instead of fixed partitioning, we implement an adaptive segmentation module comparing Entropy-based, Variance-based, and Kernel-based Change Point Detection (CPD). This ensures that window boundaries align dynamically with the time series' stationary states, placing tight windows around volatile areas and wide windows over stable trends.

2. **Enhanced Linear Fuzzy Granules:**
   We extend standard LFIG by constructing mathematical bounds that respect both local linear trend slopes and standard deviations. This results in robust fuzzy granules represented by functional upper and lower linear envelopes rather than static, horizontal spreads.

3. **Multi-Feature Granular representation (10-dimensional space):**
   We expand the feature representation of each granule from 3 features (Lower, Upper, Trend) to **10 features**:
   - *Lower Bound ($L$)*: Lower envelope limit.
   - *Upper Bound ($U$)*: Upper envelope limit.
   - *Trend ($T$)*: Slope of the linear approximation.
   - *Entropy ($E$)*: Local permutation/Shannon entropy within the segment.
   - *Variance ($V$)*: Local variance.
   - *Volatility ($Vol$)*: Standard deviation of differences (local risk metric).
   - *Curvature ($C$)*: Second-order polynomial fit coefficient (captures acceleration).
   - *Slope ($S$)*: First-order trend coefficient.
   - *Energy ($En$)*: Sum of squared amplitudes (local signal energy).
   - *Skewness ($Sk$)*: Local skewness (directional asymmetry).

4. **Hybrid Similarity Learning and Rank Fusion:**
   We construct three distinct distance matrices over the granular sequences:
   - **Hausdorff Distance Matrix ($D_H$):** Computes set-theoretic boundary overlap.
   - **DTW Distance Matrix ($D_{DTW}$):** Computes optimal temporal alignment of trends.
   - **Cosine Distance Matrix ($D_C$):** Computes alignment of feature directions.
   We then implement a **Hybrid Fusion Layer** that merges these matrices using either normalized weighted combination or Rank Aggregation (Borda count) to compute a single robust similarity score.

5. **Extensive Benchmark Evaluation:**
   We perform a comprehensive evaluation of our framework against 7 leading state-of-the-art models (ROCKET, MiniROCKET, MultiROCKET, DrCIF, Shapelets, HIVE-COTE, DTW) across 15+ UEA/UCR datasets, tracking accuracy, runtime, memory, and significance testing.
