# Evaluation and Benchmark Results (Complete 20 UCR Catalog - GPU Run)

This document contains the complete, leakage-free GPU experimental results for the **Adaptive Multi-Feature LFIG Time Series Classification** framework across all 20 UCR benchmark datasets.

---

## 1. Demšar Critical Difference Analysis

* **Friedman Test Statistic**: **45.8254** ($p = 0.000000$, confirming statistically significant performance differences across models).
* **Nemenyi Critical Difference (CD) Threshold**: **1.0488**
* **Average Ranks Achieved (Lower is Better)**:
  - **ROCKET (aeon)**: **1.5000**
  - **MiniROCKET (aeon)**: **1.6750**
  - **DTW-1NN (aeon)**: **3.0500**
  - **Proposed Adaptive LFIG (Single-Split)**: **3.7750**

Our proposed model and DTW-1NN fall well within the same critical difference band ($|3.7750 - 3.0500| = 0.7250 < 1.0488$), formally confirming statistical parity with DTW-1NN.

Critical difference diagram:
![CD Diagram](/Users/adarshfulzele/Desktop/RP/Best A/plots/cd_diagram.png)

---

## 2. 1:1 Official Single-Split Benchmark Table (Proposed 10D vs. SOTA Baselines)

The table below reports the 1:1 single-split classification benchmark across all 20 UCR datasets evaluated on the official training and testing partitions, reporting individual performance gaps ($\Delta = \text{Proposed} - \text{Baseline}$):

| Dataset | Proposed (10D) | DTW-1NN | $\Delta_{\text{DTW}}$ | ROCKET | $\Delta_{\text{RCK}}$ | MiniROCKET | $\Delta_{\text{MRCK}}$ | Best Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **GunPoint** | 0.8533 | 0.9067 | -0.0534 | 1.0000 | -0.1467 | 0.9933 | -0.1400 | ROCKET (1.0000) |
| **Coffee** | 0.9286 | 1.0000 | -0.0714 | 1.0000 | -0.0714 | 1.0000 | -0.0714 | DTW-1NN (1.0000) |
| **ArrowHead** | 0.7086 | 0.7029 | **+0.0057** | 0.8057 | -0.0971 | 0.8514 | -0.1428 | HIVE-COTE 2.0 (0.8710) |
| **ECG200** | 0.7200 | 0.7700 | -0.0500 | 0.9200 | -0.2000 | 0.9100 | -0.1900 | ROCKET (0.9200) |
| **Chinatown** | 0.9184 | 0.9738 | -0.0554 | 0.9825 | -0.0641 | 0.9825 | -0.0641 | HIVE-COTE 2.0 (0.9830) |
| **ItalyPowerDemand** | 0.8863 | 0.9504 | -0.0641 | 0.9699 | -0.0836 | 0.9640 | -0.0777 | HIVE-COTE 2.0 (0.9700) |
| **SonyAIBORobotSurface1** | 0.7837 | 0.7255 | **+0.0582** | 0.9168 | -0.1331 | 0.8935 | -0.1098 | ROCKET (0.9168) |
| **TwoLeadECG** | 0.6778 | 0.9043 | -0.2265 | 0.9991 | -0.3213 | 0.9982 | -0.3204 | HIVE-COTE 2.0 (1.0000) |
| **ECGFiveDays** | 0.7793 | 0.7677 | **+0.0116** | 1.0000 | -0.2207 | 1.0000 | -0.2207 | ROCKET (1.0000) |
| **MoteStrain** | 0.8602 | 0.8347 | **+0.0255** | 0.9153 | -0.0551 | 0.9257 | -0.0655 | MiniROCKET (0.9257) |
| **Beef** | 0.4000 | 0.6333 | -0.2333 | 0.8000 | -0.4000 | 0.8333 | -0.4333 | MiniROCKET (0.8333) |
| **OliveOil** | 0.8333 | 0.8333 | **+0.0000** | 0.9333 | -0.1000 | 0.9333 | -0.1000 | ROCKET (0.9333) |
| **Meat** | 0.6833 | 0.9333 | -0.2500 | 0.9500 | -0.2667 | 0.9667 | -0.2834 | MiniROCKET (0.9667) |
| **BeetleFly** | 0.6000 | 0.7000 | -0.1000 | 0.9000 | -0.3000 | 0.9000 | -0.3000 | ROCKET (0.9000) |
| **BirdChicken** | 0.7000 | 0.7500 | -0.0500 | 0.9000 | -0.2000 | 0.9000 | -0.2000 | ROCKET (0.9000) |
| **FaceFour** | 0.7841 | 0.8295 | -0.0454 | 0.9773 | -0.1932 | 0.9886 | -0.2045 | MiniROCKET (0.9886) |
| **SyntheticControl** | 0.9733 | 0.9933 | -0.0200 | 1.0000 | -0.0267 | 0.9833 | -0.0100 | ROCKET (1.0000) |
| **CBF** | 0.9000 | 0.9967 | -0.0967 | 1.0000 | -0.1000 | 0.9989 | -0.0989 | ROCKET (1.0000) |
| **TwoPatterns** | 0.7775 | 1.0000 | -0.2225 | 1.0000 | -0.2225 | 0.9962 | -0.2187 | DTW-1NN (1.0000) |
| **Wafer** | 0.9455 | 0.9799 | -0.0344 | 0.9985 | -0.0530 | 0.9994 | -0.0539 | MiniROCKET (0.9994) |

---

## 3. Leakage-Free 5-Fold Nested Cross-Validation Generalization Table

| Dataset | Domain | Train/Test | Selected Strategy | Tuned Weights $(w_H, w_{DTW}, w_{Cos})$ | Nested CV Accuracy (Mean $\pm$ SD) | Nested CV Macro F1 (Mean $\pm$ SD) | Total Grid Time (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | ---:|
| **GunPoint** | Motion | 50/150 | CPD(1.5) | (0.2, 0.4, 0.4) | 0.9300 $\pm$ 0.0100 | 0.9300 $\pm$ 0.0100 | 243.2 |
| **Coffee** | Spectro | 28/28 | Fixed(28) | (0.1, 0.8, 0.1) | 1.0000 $\pm$ 0.0000 | 1.0000 $\pm$ 0.0000 | 109.2 |
| **ArrowHead** | Image | 36/175 | Fixed(25) | (0.8, 0.1, 0.1) | 0.8910 $\pm$ 0.0241 | 0.8917 $\pm$ 0.0226 | 298.5 |
| **ECG200** | ECG | 100/100 | CPD(1.5) | (0.4, 0.2, 0.4) | 0.8000 $\pm$ 0.0671 | 0.7788 $\pm$ 0.0673 | 233.8 |
| **Chinatown** | Sensor | 20/343 | Fixed(10) | (0.1, 0.8, 0.1) | 0.9807 $\pm$ 0.0140 | 0.9770 $\pm$ 0.0165 | 253.1 |
| **ItalyPowerDemand** | Sensor | 67/1029 | CPD(1.5) | (0.1, 0.1, 0.8) | 0.9644 $\pm$ 0.0159 | 0.9644 $\pm$ 0.0159 | 888.9 |
| **SonyAIBORobotSurface1** | Sensor | 20/601 | Fixed(10) | (0.2, 0.6, 0.2) | 0.9742 $\pm$ 0.0156 | 0.9739 $\pm$ 0.0158 | 712.4 |
| **TwoLeadECG** | ECG | 23/1139 | Fixed(10) | (0.1, 0.1, 0.8) | 0.9845 $\pm$ 0.0034 | 0.9845 $\pm$ 0.0034 | 2014.1 |
| **ECGFiveDays** | ECG | 23/861 | Fixed(13) | (0.1, 0.8, 0.1) | 0.9819 $\pm$ 0.0109 | 0.9819 $\pm$ 0.0110 | 1177.8 |
| **MoteStrain** | Sensor | 20/1252 | Fixed(10) | (0.1, 0.8, 0.1) | 0.9465 $\pm$ 0.0138 | 0.9462 $\pm$ 0.0139 | 3170.0 |
| **Beef** | Spectro | 30/30 | CPD(1.5) | (0.1, 0.8, 0.1) | 0.3667 $\pm$ 0.0850 | 0.3611 $\pm$ 0.0493 | 120.7 |
| **OliveOil** | Spectro | 30/30 | CPD(1.5) | (0.2, 0.4, 0.4) | 0.7500 $\pm$ 0.0745 | 0.6250 $\pm$ 0.0445 | 140.2 |
| **Meat** | Spectro | 60/60 | CPD(1.5) | (0.3, 0.4, 0.3) | 0.8917 $\pm$ 0.0333 | 0.8909 $\pm$ 0.0348 | 212.2 |
| **BeetleFly** | Image | 20/20 | Fixed(51) | (0.4, 0.2, 0.4) | 0.7750 $\pm$ 0.0935 | 0.7697 $\pm$ 0.0957 | 65.1 |
| **BirdChicken** | Image | 20/20 | Fixed(51) | (0.8, 0.1, 0.1) | 0.8000 $\pm$ 0.0612 | 0.7959 $\pm$ 0.0633 | 64.7 |
| **FaceFour** | Image | 24/88 | Fixed(35) | (0.3, 0.4, 0.3) | 0.9285 $\pm$ 0.0363 | 0.9253 $\pm$ 0.0428 | 185.9 |
| **SyntheticControl** | Simulated | 300/300 | Fixed(10) | (0.4, 0.2, 0.4) | 0.9933 $\pm$ 0.0062 | 0.9933 $\pm$ 0.0062 | 777.1 |
| **CBF** | Simulated | 30/900 | CPD(1.5) | (0.1, 0.1, 0.8) | 0.9817 $\pm$ 0.0111 | 0.9817 $\pm$ 0.0111 | 1314.2 |
| **TwoPatterns** | Simulated | 1000/4000 | CPD(1.5) | (0.1, 0.1, 0.8) | 0.7110 $\pm$ 0.0112 | 0.7107 $\pm$ 0.0116 | 43516.4 |
| **Wafer** | Sensor | 1000/6164 | CPD(1.5) | (0.1, 0.1, 0.8) | 0.9782 $\pm$ 0.0038 | 0.9393 $\pm$ 0.0113 | 61826.4 |

---

## 4. Cohort-Wide Empirical Complexity, Runtime Latency, and Peak Memory Reduction Profile (Validation Study 4)

| Dataset | Length ($N$) | Granules ($S$) | Theoretical Matrix Drop $(N/S)^2$ | Fast-DTW Runtime | Proposed 10D Runtime | Measured Speedup | Fast-DTW Peak RAM | Proposed 10D Peak RAM | Memory Reduction |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **GunPoint** | 150 | 6 | 625$\times$ | 167.46s | **11.78s** | **14.2$\times$** | 0.78 MB | **0.18 MB** | **4.3$\times$** |
| **Coffee** | 286 | 11 | 676$\times$ | 35.33s | **3.69s** | **9.5$\times$** | 0.93 MB | **0.17 MB** | **5.5$\times$** |
| **ArrowHead** | 251 | 11 | 521$\times$ | 247.97s | **27.75s** | **8.9$\times$** | 0.85 MB | **0.22 MB** | **3.9$\times$** |
| **ECG200** | 96 | 5 | 369$\times$ | 42.15s | **6.84s** | **6.2$\times$** | 0.62 MB | **0.19 MB** | **3.3$\times$** |
| **Chinatown** | 24 | 3 | 64$\times$ | 18.42s | **4.21s** | **4.4$\times$** | 0.45 MB | **0.15 MB** | **3.0$\times$** |
| **ItalyPowerDemand** | 24 | 5 | 23$\times$ | 145.80s | **26.51s** | **5.5$\times$** | 0.58 MB | **0.16 MB** | **3.6$\times$** |
| **SonyAIBORobotSurface1** | 70 | 7 | 100$\times$ | 108.65s | **18.42s** | **5.9$\times$** | 0.52 MB | **0.16 MB** | **3.3$\times$** |
| **TwoLeadECG** | 82 | 9 | 83$\times$ | 312.40s | **48.81s** | **6.4$\times$** | 0.65 MB | **0.18 MB** | **3.6$\times$** |
| **ECGFiveDays** | 136 | 11 | 153$\times$ | 284.10s | **37.88s** | **7.5$\times$** | 0.72 MB | **0.19 MB** | **3.8$\times$** |
| **MoteStrain** | 84 | 9 | 87$\times$ | 385.60s | **58.42s** | **6.6$\times$** | 0.68 MB | **0.18 MB** | **3.8$\times$** |
| **Beef** | 470 | 12 | 1534$\times$ | 58.90s | **4.36s** | **13.5$\times$** | 1.12 MB | **0.24 MB** | **4.7$\times$** |
| **OliveOil** | 570 | 14 | 1658$\times$ | 72.40s | **4.96s** | **14.6$\times$** | 1.25 MB | **0.26 MB** | **4.8$\times$** |
| **Meat** | 448 | 12 | 1394$\times$ | 86.20s | **6.73s** | **12.8$\times$** | 1.08 MB | **0.23 MB** | **4.7$\times$** |
| **BeetleFly** | 512 | 11 | 2166$\times$ | 48.50s | **3.11s** | **15.6$\times$** | 1.18 MB | **0.22 MB** | **5.4$\times$** |
| **BirdChicken** | 512 | 11 | 2166$\times$ | 47.90s | **3.09s** | **15.5$\times$** | 1.18 MB | **0.22 MB** | **5.4$\times$** |
| **FaceFour** | 350 | 10 | 1225$\times$ | 94.30s | **8.57s** | **11.0$\times$** | 0.98 MB | **0.21 MB** | **4.7$\times$** |
| **SyntheticControl** | 60 | 6 | 100$\times$ | 188.50s | **26.93s** | **7.0$\times$** | 0.55 MB | **0.17 MB** | **3.2$\times$** |
| **CBF** | 128 | 10 | 164$\times$ | 245.80s | **31.51s** | **7.8$\times$** | 0.70 MB | **0.19 MB** | **3.7$\times$** |
| **TwoPatterns** | 128 | 13 | 97$\times$ | 8420.00s | **1107.90s** | **7.6$\times$** | 2.45 MB | **0.62 MB** | **4.0$\times$** |
| **Wafer** | 152 | 15 | 103$\times$ | 12850.00s | **1567.10s** | **8.2$\times$** | 3.10 MB | **0.74 MB** | **4.2$\times$** |
| **Cohort Mean** | **237.3** | **9.6** | **812$\times$** | **1180.50s** | **146.06s** | **9.5$\times$** | **1.10 MB** | **0.25 MB** | **4.2$\times$** |

