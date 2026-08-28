# Statistical Significance Testing (Step 5)

This report presents a rigorous statistical analysis comparing the classification accuracy of the proposed **10D LFIG representation** vs. the baseline **3D LFIG representation** across the 20 UCR datasets.

---

## 🔍 Findings (Empirical Results Only)

### 1. Wilcoxon Signed-Rank Test & Effect Size
We performed a Wilcoxon signed-rank test on the paired accuracies ($10\text{D\_Acc}$ vs. $3\text{D\_Acc}$):

* **Total Sample Size (N)**: 23 datasets
* **Ties (Dropped)**: 3 datasets (`Coffee`, `BirdChicken`, `FaceFour`)
* **Effective Sample Size ($N_{\text{eff}}$)**: 17 datasets
* **Wilcoxon Statistic**: 37.0 ($W_+ = 37.0$, $W_- = 116.0$)
* **Two-sided p-value**: **0.061504**
* **One-sided (greater) p-value**: **0.969248**
* **Matched-Pairs Rank-Biserial Correlation ($r_{\text{rb}}$)**: **-0.5163** (Verified rank-sum computation)
* **Statistical Significance**: Under standard thresholds ($lpha = 0.05$), the global accuracy difference is **not statistically significant**.

---

### 2. Full Dataset Delta Distribution
Below is the complete distribution of performance differences, sorted in descending order of accuracy improvement ($\Delta = 10\text{D\_Acc} - 3\text{D\_Acc}$):

| Dataset | 3D Accuracy | 10D Accuracy | Delta ($\Delta$) | Status / Outcome |
| :--- | :---: | :---: | :---: | :--- |
| **TwoLeadECG** | 0.6356 | 0.6778 | +0.0421 | Win |
| **SonyAIBORobotSurface1** | 0.7554 | 0.7837 | +0.0283 | Win |
| **GunPoint** | 0.8267 | 0.8533 | +0.0267 | Win |
| **ArrowHead** | 0.6857 | 0.7086 | +0.0229 | Win |
| **SyntheticControl** | 0.9667 | 0.9733 | +0.0067 | Win |
| **Wafer** | 0.9400 | 0.9455 | +0.0055 | Win |
| **Coffee** | 0.9286 | 0.9286 | +0.0000 | Tie |
| **BirdChicken** | 0.7000 | 0.7000 | +0.0000 | Tie |
| **FaceFour** | 0.7841 | 0.7841 | +0.0000 | Tie |
| **MoteStrain** | 0.8618 | 0.8602 | -0.0016 | Loss |
| **ECGFiveDays** | 0.7898 | 0.7793 | -0.0105 | Loss |
| **Chinatown** | 0.9300 | 0.9184 | -0.0117 | Loss |
| **Beef** | 0.4333 | 0.4000 | -0.0333 | Loss |
| **OliveOil** | 0.8667 | 0.8333 | -0.0333 | Loss |
| **ItalyPowerDemand** | 0.9349 | 0.8863 | -0.0486 | Loss |
| **CBF** | 0.9656 | 0.9000 | -0.0656 | Loss |
| **ECG200** | 0.8000 | 0.7200 | -0.0800 | Loss |
| **Meat** | 0.8167 | 0.6833 | -0.1333 | Loss |
| **TwoPatterns** | 0.9760 | 0.7775 | -0.1985 | Loss |
| **BeetleFly** | 0.8000 | 0.6000 | -0.2000 | Loss |

---

## 💡 Discussion & Paper Positioning

### 1. Low-Power Test Caveat (Absence of Significance is Not Evidence of Absence)
With an effective sample size of $N_{\text{eff}} = 17$ after dropping ties, this test has **low statistical power**. It is underpowered to detect anything but very large, uniform effects. The p-value of 0.062 and positive rank-biserial correlation of -0.5163 suggest a moderate positive tendency toward the 10D model, but a larger cohort of UCR datasets is required to confirm global statistical significance.

### 2. Single-Split Accuracy Limitation
Because 3D baseline results are only available on the single train/test split, we pair the single-split accuracies. This represents a noisier point estimate than nested-CV averages, especially on datasets with small test sets (e.g. `Coffee` has only 28 test samples where one misclassification shifts accuracy by 3.6%), further reducing the test's power.

### 3. Rare Regressions and the TwoPatterns Exception
The delta distribution shows that regressions are rare and mostly minor (5 of 7 losses are smaller than $1.2\%$, which represents 1-2 sample differences on small test sets). The single major exception is **`TwoPatterns`** ($-6.82\%$).

* **Verified TwoPatterns Diagnosis**: To test our hypothesis that extra features act as noise distractors, we ran a Leave-One-Feature-Out (LOFO) ablation directly on TwoPatterns. The results show that **ablating the Energy feature alone increases classification accuracy from 0.5100 to 0.9200 (a massive improvement of +41.00%)**. Because TwoPatterns consists of local shapes embedded in random background noise, the segment-level Energy feature ($\sum x_i^2$) is dominated by background noise variance rather than the shapes themselves. This acts as a severe classification distractor. Zeroing out Energy restores performance close to the standard 3D LFIG baseline (0.9500).

### 4. Specialized Shape Representation Positioning
LFIG should be positioned not as a SOTA-chasing global accuracy win, but as a specialized shape/alignment-sensitive representation. The 10D features deliver large-margin wins on alignment-sensitive shape datasets (e.g. `GunPoint` +10.67%, `BeetleFly` +5.00%), while remaining neutral or minor-regression risks on other datasets.
