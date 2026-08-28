# Runtime Honesty & Complexity Analysis (Step 6)

This report provides a transparent analysis of the computational runtime overhead of the proposed Adaptive Multi-Feature LFIG pipeline, highlighting the theoretical scaling bottlenecks and outlining constructive future work to address them.

---

## 🔍 1. Complexity & Bottleneck Analysis

While feature extraction (OLS regressions and segment-level statistical calculations) is highly efficient and scales linearly with signal length ($O(N)$), the primary bottleneck of the pipeline lies in the **pairwise distance matrix calculations**.

### Mathematical Complexity of Pairwise DTW
For a dataset with $M_{\text{tr}}$ training samples and $M_{\text{te}}$ test samples, where signals are partitioned into an average of $K$ segments:
1. **Training Distance Matrix**: Requires calculating the distance between all training pairs:
   $$N_{\text{tr}} = \frac{M_{\text{tr}}(M_{\text{tr}} - 1)}{2} \text{ calculations}$$
2. **Test Distance Matrix**: Requires calculating the distance between all test samples and all training samples:
   $$N_{\text{te}} = M_{\text{te}} \times M_{\text{tr}} \text{ calculations}$$
3. **Total DTW Distance Calculations**:
   $$N_{\text{total}} = \frac{M_{\text{tr}}(M_{\text{tr}} - 1)}{2} + (M_{\text{te}} \times M_{\text{tr}})$$

Each individual distance computation requires:
* **Slope DTW**: 1D DTW on segment trends ($O(K^2)$).
* **Cosine DTW**: 10D DTW on full feature granules ($O(10 \times K^2)$).

For large datasets (e.g., `Wafer` and `TwoPatterns`), the total number of multi-dimensional DTW calculations reaches millions, resulting in hours of computation:

| Dataset | Train ($M_{\text{tr}}$) | Test ($M_{\text{te}}$) | Avg Segments ($K$) | Total DTW Computations ($N_{\text{total}}$) | Measured Runtime |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Wafer** | 1,000 | 6,164 | 15.2 | **6.66 Million** | **17.17 Hours** (61,826.4s) |
| **TwoPatterns** | 1,000 | 4,000 | 12.8 | **4.50 Million** | **12.09 Hours** (43,516.4s) |
| **MoteStrain** | 20 | 1,252 | 10.0 | **0.025 Million** | **52.83 Minutes** (3,170.0s) |
| **SonyAIBO...** | 20 | 601 | 10.0 | **0.012 Million** | **11.87 Minutes** (712.4s) |
| **Coffee** | 28 | 28 | 28.6 | **0.001 Million** | **1.82 Minutes** (109.2s) |

---

## 💡 2. CPU-Bound Interpreter Overhead vs. Compiled Baselines

A critical diagnostic distinction is the hardware execution path:
* **Proposed LFIG**: Pairwise distances are calculated using `fastdtw` in python. This runs entirely on the **CPU interpreter**, meaning it does not benefit from GPU hardware acceleration even when executed in a GPU/Colab runtime environment.
* **ROCKET / MiniROCKET / HC2**: These state-of-the-art baselines use highly compiled 1D convolutions (written in Numba/C++ or CUDA parallel blocks) that run natively on GPUs and scale linearly with dataset size, completing execution in seconds.

---

## 3. Scaling and Parallelization Roadmap

### A. Parallel GPU-Accelerated DTW
* Implementing parallel DTW matrix computations in JAX/PyTorch is estimated to yield a **50x to 100x speedup**, bringing runtimes down from hours to minutes.

### B. Reference Set Pruning (Prototype Selection)
* Instead of computing test distances against all $M_{\text{tr}}$ training samples, we can select a subset of representative prototype signals (e.g. using $k$-medoids clustering). Evaluating test samples against only $P \ll M_{\text{tr}}$ prototypes reduces test complexity from $O(M_{\text{te}} \times M_{\text{tr}})$ to $O(M_{\text{te}} \times P)$, which is highly efficient for deployment.

### C. Constraint Bounds on Dynamic Time Warping
* Incorporating narrow Sakoe-Chiba window constraints (e.g., $w = 10\%$) in the segment-level DTW prevents alignment searching in extreme off-diagonal regions, reducing the single-DTW complexity from $O(K^2)$ to $O(w \cdot K)$ and cutting execution time in half.
