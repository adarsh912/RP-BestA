# Phase 9: Evaluation and Benchmark Results

## 1. Classification Performance Comparison

| Dataset   | Classifier                 |   Accuracy |   Precision |     Recall |   Macro F1 |   Runtime (s) |   Peak Memory (MB) |
|:----------|:---------------------------|-----------:|------------:|-----------:|-----------:|--------------:|-------------------:|
| GunPoint  | Our Proposed (Hybrid LFIG) |   0.906667 |    0.911797 |   0.905939 |   0.90625  |      11.5989  |           0.762583 |
| GunPoint  | Fast-DTW kNN               |   0.886667 |    0.888126 |   0.887091 |   0.886621 |     165.638   |           0.607797 |
| GunPoint  | HIVE-COTE 2.0              |   1        |  nan        | nan        | nan        |     nan       |         nan        |
| GunPoint  | MultiROCKET                |   1        |  nan        | nan        | nan        |     nan       |         nan        |
| GunPoint  | MiniROCKET                 |   1        |  nan        | nan        | nan        |     nan       |         nan        |
| GunPoint  | DrCIF                      |   0.987    |  nan        | nan        | nan        |     nan       |         nan        |
| GunPoint  | Shapelets                  |   1        |  nan        | nan        | nan        |     nan       |         nan        |
| GunPoint  | DTW (Literature)           |   0.913    |  nan        | nan        | nan        |     nan       |         nan        |
| Coffee    | Our Proposed (Hybrid LFIG) |   1        |    1        |   1        |   1        |       3.68521 |           0.169738 |
| Coffee    | Fast-DTW kNN               |   0.928571 |    0.941176 |   0.923077 |   0.927083 |      36.8964  |           1.0323   |
| Coffee    | HIVE-COTE 2.0              |   1        |  nan        | nan        | nan        |     nan       |         nan        |
| Coffee    | MultiROCKET                |   1        |  nan        | nan        | nan        |     nan       |         nan        |
| Coffee    | MiniROCKET                 |   1        |  nan        | nan        | nan        |     nan       |         nan        |
| Coffee    | DrCIF                      |   0.993    |  nan        | nan        | nan        |     nan       |         nan        |
| Coffee    | Shapelets                  |   0.986    |  nan        | nan        | nan        |     nan       |         nan        |
| Coffee    | DTW (Literature)           |   0.993    |  nan        | nan        | nan        |     nan       |         nan        |

## 2. Ablation Study: 3-Feature Standard LFIG vs. 10-Feature Enhanced LFIG

| Dataset   |   Standard LFIG (3 Feat) Acc |   Our Enhanced LFIG (10 Feat) Acc |   Ablation Acc Drop |   Standard LFIG (3 Feat) F1 |   Our Enhanced LFIG (10 Feat) F1 |
|:----------|-----------------------------:|----------------------------------:|--------------------:|----------------------------:|---------------------------------:|
| GunPoint  |                     0.88     |                          0.906667 |           0.0266667 |                    0.879658 |                          0.90625 |
| Coffee    |                     0.964286 |                          1        |           0.0357143 |                    0.96424  |                          1       |