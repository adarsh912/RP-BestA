# Master Summary Benchmark Results (GPU Run)

This table summarizes the benchmark results for all 23 UCR datasets evaluated on GPU.

| Dataset               |   3D_Acc |   10D_Acc | NestedCV      |   Time (s) | Best_Baseline          | Outperformed_By        |
|:----------------------|---------:|----------:|:--------------|-----------:|:-----------------------|:-----------------------|
| GunPoint              | 0.8      |  0.906667 | 0.9750±0.0224 |      107.8 | ROCKET (1.0000)        | ROCKET (1.0000)        |
| Coffee                | 0.928571 |  0.928571 | 1.0000±0.0000 |       37.7 | DTW-1NN (1.0000)       | DTW-1NN (1.0000)       |
| ArrowHead             | 0.714286 |  0.702857 | 0.8863±0.0232 |      131.4 | HIVE-COTE 2.0 (0.8710) | HIVE-COTE 2.0 (0.8710) |
| ECG200                | 0.86     |  0.88     | 0.8600±0.0624 |      112.2 | ROCKET (0.9200)        | ROCKET (0.9200)        |
| Chinatown             | 0.927114 |  0.915452 | 0.9807±0.0140 |       69.4 | HIVE-COTE 2.0 (0.9830) | HIVE-COTE 2.0 (0.9830) |
| ItalyPowerDemand      | 0.92517  |  0.934888 | 0.9608±0.0165 |      199.9 | HIVE-COTE 2.0 (0.9700) | HIVE-COTE 2.0 (0.9700) |
| SonyAIBORobotSurface1 | 0.760399 |  0.780366 | 0.9855±0.0106 |      235.3 | ROCKET (0.9168)        | ROCKET (0.9168)        |
| TwoLeadECG            | 0.638279 |  0.674276 | 0.9871±0.0038 |      469.5 | HIVE-COTE 2.0 (1.0000) | HIVE-COTE 2.0 (1.0000) |
| ECGFiveDays           | 0.781649 |  0.772358 | 0.9989±0.0023 |      481.2 | ROCKET (1.0000)        | ROCKET (1.0000)        |
| MoteStrain            | 0.783546 |  0.790735 | 0.8821±0.0065 |      430.5 | MiniROCKET (0.9257)    | MiniROCKET (0.9257)    |
| Beef                  | 0.633333 |  0.633333 | 0.6000±0.0624 |       57.8 | MiniROCKET (0.8333)    | MiniROCKET (0.8333)    |
| OliveOil              | 0.9      |  0.866667 | 0.8833±0.0667 |       48.7 | ROCKET (0.9333)        | ROCKET (0.9333)        |
| Meat                  | 0.883333 |  0.883333 | 1.0000±0.0000 |       80.4 | MiniROCKET (0.9667)    | MiniROCKET (0.9667)    |
| BeetleFly             | 0.8      |  0.85     | 0.8500±0.1225 |       27.2 | ROCKET (0.9000)        | ROCKET (0.9000)        |
| BirdChicken           | 0.65     |  0.65     | 0.8250±0.0612 |       27.4 | ROCKET (0.9000)        | ROCKET (0.9000)        |
| FaceFour              | 0.795455 |  0.784091 | 0.9285±0.0363 |       74.8 | MiniROCKET (0.9886)    | MiniROCKET (0.9886)    |
| SyntheticControl      | 0.956667 |  0.95     | 0.8700±0.0356 |      187.6 | ROCKET (1.0000)        | ROCKET (1.0000)        |
| CBF                   | 0.882222 |  0.921111 | 0.9946±0.0068 |      526.6 | ROCKET (1.0000)        | ROCKET (1.0000)        |
| TwoPatterns           | 0.826    |  0.75775  | 0.8122±0.0086 |     4585.8 | DTW-1NN (1.0000)       | DTW-1NN (1.0000)       |
| Wafer                 | 0.985724 |  0.989293 | 0.9983±0.0009 |    15986.3 | MiniROCKET (0.9994)    | MiniROCKET (0.9994)    |
| FordA                 | 0.615909 |  0.627273 | 0.6151±0.0078 |    14612.7 | MiniROCKET (0.9508)    | MiniROCKET (0.9508)    |
| Yoga                  | 0.771667 |  0.793333 | 0.9245±0.0124 |     2984.5 | ROCKET (0.9173)        | ROCKET (0.9173)        |
| SwedishLeaf           | 0.8608   |  0.8656   | 0.8951±0.0118 |      605.9 | MiniROCKET (0.9696)    | MiniROCKET (0.9696)    |
