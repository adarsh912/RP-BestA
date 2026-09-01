# Master Benchmark Results (GPU Run - 20 Datasets)

This table summarizes the benchmark results for all 20 UCR datasets evaluated on GPU.

| Dataset | Baseline 3D LFIG | Proposed 10D LFIG | Proposed Nested CV | Time (s) | Best Baseline | Outperformed By |
|:---|:---:|:---:|:---:|:---:|:---|:---|
| GunPoint | 0.826700 | 0.853300 | 0.9300±0.0100 | 243.2 | ROCKET (1.0000) | ROCKET (1.0000) |
| Coffee | 0.928600 | 0.928600 | 1.0000±0.0000 | 109.2 | DTW-1NN (1.0000) | N/A |
| ArrowHead | 0.685700 | 0.708600 | 0.8910±0.0241 | 298.5 | HIVE-COTE 2.0 (0.8710) | N/A |
| ECG200 | 0.800000 | 0.720000 | 0.8000±0.0671 | 233.8 | ROCKET (0.9200) | ROCKET (0.9200) |
| Chinatown | 0.930000 | 0.918400 | 0.9807±0.0140 | 253.1 | HIVE-COTE 2.0 (0.9830) | ROCKET (0.9825) |
| ItalyPowerDemand | 0.934900 | 0.886300 | 0.9644±0.0159 | 888.9 | HIVE-COTE 2.0 (0.9700) | ROCKET (0.9699) |
| SonyAIBORobotSurface1 | 0.755400 | 0.783700 | 0.9742±0.0156 | 712.4 | ROCKET (0.9168) | N/A |
| TwoLeadECG | 0.635600 | 0.677800 | 0.9845±0.0034 | 2014.1 | HIVE-COTE 2.0 (1.0000) | ROCKET (0.9991) |
| ECGFiveDays | 0.789800 | 0.779300 | 0.9819±0.0109 | 1177.8 | ROCKET (1.0000) | ROCKET (1.0000) |
| MoteStrain | 0.861800 | 0.860200 | 0.9465±0.0138 | 3170.0 | MiniROCKET (0.9257) | N/A |
| Beef | 0.433300 | 0.400000 | 0.3667±0.0850 | 120.7 | MiniROCKET (0.8333) | DTW-1NN (0.6333) |
| OliveOil | 0.866700 | 0.833300 | 0.7500±0.0745 | 140.2 | ROCKET (0.9333) | DTW-1NN (0.8333) |
| Meat | 0.816700 | 0.683300 | 0.8917±0.0333 | 212.2 | MiniROCKET (0.9667) | DTW-1NN (0.9333) |
| BeetleFly | 0.800000 | 0.600000 | 0.7750±0.0935 | 65.1 | ROCKET (0.9000) | ROCKET (0.9000) |
| BirdChicken | 0.700000 | 0.700000 | 0.8000±0.0612 | 64.7 | ROCKET (0.9000) | ROCKET (0.9000) |
| FaceFour | 0.784100 | 0.784100 | 0.9285±0.0363 | 185.9 | MiniROCKET (0.9886) | ROCKET (0.9773) |
| SyntheticControl | 0.966700 | 0.973300 | 0.9933±0.0062 | 777.1 | ROCKET (1.0000) | ROCKET (1.0000) |
| CBF | 0.965600 | 0.900000 | 0.9817±0.0111 | 1314.2 | ROCKET (1.0000) | DTW-1NN (0.9967) |
| TwoPatterns | 0.976000 | 0.777500 | 0.7110±0.0112 | 43516.4 | DTW-1NN (1.0000) | DTW-1NN (1.0000) |
| Wafer | 0.940000 | 0.945500 | 0.9782±0.0038 | 61826.4 | MiniROCKET (0.9994) | DTW-1NN (0.9799) |
