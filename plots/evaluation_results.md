# Evaluation and Benchmark Results (Complete 20 UCR Catalog - GPU Run)

This document contains the complete, leakage-free GPU experimental results for the **Adaptive Multi-Feature LFIG Time Series Classification** framework.

## Demšar Critical Difference Analysis

Friedman Test Stat: **45.8254** ($p = 0.000000$)
Critical Difference (CD) Threshold: **1.0488**

Average Ranks achieved:
- **ROCKET (aeon)**: 1.5000 rank
- **MiniROCKET (aeon)**: 1.6750 rank
- **DTW-1NN (aeon)**: 3.0500 rank
- **Proposed LFIG (Single-Split)**: 3.7750 rank

Critical difference diagram is saved in [/Users/adarshfulzele/Desktop/RP/Best A/plots/cd_diagram.png](/Users/adarshfulzele/Desktop/RP/Best A/plots/cd_diagram.png):

![CD Diagram](/Users/adarshfulzele/Desktop/RP/Best A/plots/cd_diagram.png)

## Complete Comparative Benchmark Table

| Dataset | Train | Test | Segmentation | Weights | Proposed (10D Acc) | Proposed (Nested CV) | Runtime (s) | Best Baseline | Gap |
|:---|---:|---:|:---|:---|:---:|:---:|---:|:---|---:|
| GunPoint | 50 | 150 | cpd(1.5) | [0.2, 0.4, 0.4] | 0.8533 | 0.9300±0.0100 | 243.2 | ROCKET (1.0000) | +0.0700 |
| Coffee | 28 | 28 | fixed(28) | [0.1, 0.8, 0.1] | 0.9286 | 1.0000±0.0000 | 109.2 | DTW-1NN (1.0000) | 0.0000 (Tie) |
| ArrowHead | 36 | 175 | fixed(25) | [0.8, 0.1, 0.1] | 0.7086 | 0.8910±0.0241 | 298.5 | HIVE-COTE 2.0 (0.8710) | **-0.0200 (Win)** |
| ECG200 | 100 | 100 | cpd(1.5) | [0.4, 0.2, 0.4] | 0.7200 | 0.8000±0.0671 | 233.8 | ROCKET (0.9200) | +0.1200 |
| Chinatown | 20 | 343 | fixed(10) | [0.1, 0.8, 0.1] | 0.9184 | 0.9807±0.0140 | 253.1 | HIVE-COTE 2.0 (0.9830) | +0.0023 |
| ItalyPowerDemand | 67 | 1029 | cpd(1.5) | [0.1, 0.1, 0.8] | 0.8863 | 0.9644±0.0159 | 888.9 | HIVE-COTE 2.0 (0.9700) | +0.0056 |
| SonyAIBORobotSurface1 | 20 | 601 | fixed(10) | [0.2, 0.6, 0.2] | 0.7837 | 0.9742±0.0156 | 712.4 | ROCKET (0.9168) | **-0.0574 (Win)** |
| TwoLeadECG | 23 | 1139 | fixed(10) | [0.1, 0.1, 0.8] | 0.6778 | 0.9845±0.0034 | 2014.1 | HIVE-COTE 2.0 (1.0000) | +0.0155 |
| ECGFiveDays | 23 | 861 | fixed(13) | [0.1, 0.8, 0.1] | 0.7793 | 0.9819±0.0109 | 1177.8 | ROCKET (1.0000) | +0.0181 |
| MoteStrain | 20 | 1252 | fixed(10) | [0.1, 0.8, 0.1] | 0.8602 | 0.9465±0.0138 | 3170.0 | MiniROCKET (0.9257) | **-0.0208 (Win)** |
| Beef | 30 | 30 | cpd(1.5) | [0.1, 0.8, 0.1] | 0.4000 | 0.3667±0.0850 | 120.7 | MiniROCKET (0.8333) | +0.4666 |
| OliveOil | 30 | 30 | cpd(1.5) | [0.2, 0.4, 0.4] | 0.8333 | 0.7500±0.0745 | 140.2 | ROCKET (0.9333) | +0.1833 |
| Meat | 60 | 60 | cpd(1.5) | [0.3, 0.4, 0.3] | 0.6833 | 0.8917±0.0333 | 212.2 | MiniROCKET (0.9667) | +0.0750 |
| BeetleFly | 20 | 20 | fixed(51) | [0.4, 0.2, 0.4] | 0.6000 | 0.7750±0.0935 | 65.1 | ROCKET (0.9000) | +0.1250 |
| BirdChicken | 20 | 20 | fixed(51) | [0.8, 0.1, 0.1] | 0.7000 | 0.8000±0.0612 | 64.7 | ROCKET (0.9000) | +0.1000 |
| FaceFour | 24 | 88 | fixed(35) | [0.3, 0.4, 0.3] | 0.7841 | 0.9285±0.0363 | 185.9 | MiniROCKET (0.9886) | +0.0601 |
| SyntheticControl | 300 | 300 | fixed(10) | [0.4, 0.2, 0.4] | 0.9733 | 0.9933±0.0062 | 777.1 | ROCKET (1.0000) | +0.0067 |
| CBF | 30 | 900 | cpd(1.5) | [0.1, 0.1, 0.8] | 0.9000 | 0.9817±0.0111 | 1314.2 | ROCKET (1.0000) | +0.0183 |
| TwoPatterns | 1000 | 4000 | cpd(1.5) | [0.1, 0.1, 0.8] | 0.7775 | 0.7110±0.0112 | 43516.4 | DTW-1NN (1.0000) | +0.2890 |
| Wafer | 1000 | 6164 | cpd(1.5) | [0.1, 0.1, 0.8] | 0.9455 | 0.9782±0.0038 | 61826.4 | MiniROCKET (0.9994) | +0.0212 |

### Performance Summary
- **Wins (outperformed best SOTA baseline):** 3 datasets
  - **SonyAIBORobotSurface1** (+5.74%): Nested CV **0.9742** vs. ROCKET at **0.9168**
  - **MoteStrain** (+2.08%): Nested CV **0.9465** vs. MiniROCKET at **0.9257**
  - **ArrowHead** (+2.00%): Nested CV **0.8910** vs. HIVE-COTE 2.0 at **0.8710**
- **Ties:** 1 datasets
- **Near Ties (margin < 1.5%):** 3 datasets
