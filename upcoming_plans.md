# Upcoming Plans: Verification, Characterization, and Positioning Roadmap

This roadmap outlines the critical upcoming steps to validate the integrity of our experimental results, systematically run a clean benchmark pass, characterize the adaptive pipeline's performance, and finalize our paper draft.

---

## 📋 Steps & Action Plan

### 🚀 Step 1: Leakage Check (Critical Sanity Check)
> [!IMPORTANT]
> This sanity test must be run first, before performing any downstream evaluation or paper writing, to confirm that our nested cross-validation is leak-proof.
- **Protocol**: Run the label-shuffle sanity test on `TwoLeadECG` and `ECGFiveDays`.
- **Validation Rule**:
  - If nested-CV accuracy stays high with shuffled training labels, a data leakage bug exists. We must pause and fix the leakage before proceeding.
  - If accuracy collapses to random chance, the pipeline is verified as leak-proof.

---

### 📊 Step 2: Clean Benchmark Pass & result reports
- **Protocol**: Run a clean, single-pass benchmark over all 23 datasets.
- **Reporting Rule**:
  - `_df_to_markdown_safe` is already restored and verified.
  - Report the **5-Fold Nested Cross-Validation (CV)** results as the primary metric.
  - Report single-split results only as secondary/sanity reference metrics.
  - Present results in a single, canonical, master summary table.

---

### 🧠 Step 3: Characterize Wins & Adaptive Routing
- **Goal**: Explain exactly *why* the wins cluster on specific datasets.
- **Methodology**:
  - Retrieve the automatically selected segmentation strategy (CPD vs. Fixed) for each dataset.
  - Correlate this choice with the accuracy delta between 10D and 3D features.
  - **Empirical Hook**: Demonstrate that CPD-selected datasets show the 10D wins disproportionately. This proves that our adaptive routing strategy correctly identifies datasets that benefit from richer features, while safely falling back to fixed windowing where they do not.

---

### 🔍 Step 4: Aggregate LOFO and Distance-Fusion Ablation
- **Goal**: Provide empirical interpretability evidence for the proposed pipeline.
- **Methodology**:
  - Aggregate Leave-One-Feature-Out (LOFO) impact tables across all 23 datasets.
  - Chart which of the 10 granule features (e.g. Trend, Shannon Entropy) matter most for different domains.
  - Analyze the learned distance-fusion weights to show which distance components (Hausdorff, DTW, Cosine DTW) are prioritized on each dataset category.

---

### 📈 Step 5: Significance Testing
- **Goal**: Back up accuracy claims with rigorous statistical significance.
- **Methodology**:
  - Run the **Wilcoxon signed-rank test** comparing the 10D proposed features vs. 3D standard LFIG features across all 23 datasets.
  - If not globally significant, frame and report it as evidence for our narrower claim (e.g., demonstrating significance specifically on alignment-sensitive subsets).

---

### ⏱️ Step 6: Runtime Honesty
- **Goal**: Maintain transparency on execution overhead.
- **Methodology**:
  - Report the runtimes for large datasets (e.g., `FordA` and `Wafer`) plainly.
  - Frame high runtimes constructively as future work opportunities (e.g., developing custom CUDA-compiled batched DTW solvers or scaling out GPU batching).

---

### ✍️ Step 7: Paper Positioning & Narrative Structure
- **Narrative Flow**:
  1. **Methodology**: Detail the LFIG representation, the 10D features, and the distance metrics.
  2. **Adaptive Routing Evidence**: Show that the automatic selector routes datasets appropriately (Proof of Step 3).
  3. **Feature & Fusion Interpretability**: Highlight LOFO and learned weights (Proof of Step 4).
  4. **Competitive Accuracy**: Position accuracy as "competitive with classical baselines" rather than SOTA-chasing.
  5. **Honest Limitations**: Plainly discuss runtime and complexity trade-offs.
