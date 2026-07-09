# Phase 9: Evaluation and Benchmark Results

## 1. Classification Performance Comparison

| Dataset   | Classifier                |   Accuracy |   Precision |     Recall |   Macro F1 |   Runtime (s) |   Peak Memory (MB) |
|:----------|:--------------------------|-----------:|------------:|-----------:|-----------:|--------------:|-------------------:|
| GunPoint  | Our Proposed (KNN)        |   0.906667 |    0.911797 |   0.905939 |   0.90625  |      11.7189  |          0.764171  |
| GunPoint  | Fast-DTW kNN              |   0.886667 |    0.888126 |   0.887091 |   0.886621 |     165.198   |          0.607797  |
| GunPoint  | HIVE-COTE 2.0             |   1        |  nan        | nan        | nan        |     nan       |        nan         |
| GunPoint  | MultiROCKET               |   1        |  nan        | nan        | nan        |     nan       |        nan         |
| GunPoint  | MiniROCKET                |   1        |  nan        | nan        | nan        |     nan       |        nan         |
| GunPoint  | DrCIF                     |   0.987    |  nan        | nan        | nan        |     nan       |        nan         |
| GunPoint  | Shapelets                 |   1        |  nan        | nan        | nan        |     nan       |        nan         |
| GunPoint  | DTW (Literature)          |   0.913    |  nan        | nan        | nan        |     nan       |        nan         |
| Coffee    | Our Proposed (KNN)        |   1        |    1        |   1        |   1        |       3.72792 |          0.171775  |
| Coffee    | Fast-DTW kNN              |   0.928571 |    0.941176 |   0.923077 |   0.927083 |      40.9873  |          0.928993  |
| Coffee    | HIVE-COTE 2.0             |   1        |  nan        | nan        | nan        |     nan       |        nan         |
| Coffee    | MultiROCKET               |   1        |  nan        | nan        | nan        |     nan       |        nan         |
| Coffee    | MiniROCKET                |   1        |  nan        | nan        | nan        |     nan       |        nan         |
| Coffee    | DrCIF                     |   0.993    |  nan        | nan        | nan        |     nan       |        nan         |
| Coffee    | Shapelets                 |   0.986    |  nan        | nan        | nan        |     nan       |        nan         |
| Coffee    | DTW (Literature)          |   0.993    |  nan        | nan        | nan        |     nan       |        nan         |
| ArrowHead | Our Proposed (KNN)        |   0.828571 |    0.830722 |   0.836113 |   0.828463 |      27.0682  |          0.726465  |
| ArrowHead | Fast-DTW kNN              |   0.72     |    0.720339 |   0.720992 |   0.717589 |     246.097   |          0.922943  |
| ArrowHead | HIVE-COTE 2.0             |   0.871    |  nan        | nan        | nan        |     nan       |        nan         |
| ArrowHead | MultiROCKET               |   0.865    |  nan        | nan        | nan        |     nan       |        nan         |
| ArrowHead | MiniROCKET                |   0.86     |  nan        | nan        | nan        |     nan       |        nan         |
| ArrowHead | DrCIF                     |   0.852    |  nan        | nan        | nan        |     nan       |        nan         |
| ArrowHead | DTW (Literature)          |   0.829    |  nan        | nan        | nan        |     nan       |        nan         |
| ECG200    | Our Proposed (KNN)        |   0.91     |    0.904396 |   0.899306 |   0.901736 |      37.0784  |          1.18041   |
| ECG200    | Fast-DTW kNN              |   0.83     |    0.846667 |   0.782118 |   0.799505 |     124.81    |          0.32106   |
| ECG200    | HIVE-COTE 2.0             |   0.9      |  nan        | nan        | nan        |     nan       |        nan         |
| ECG200    | MultiROCKET               |   0.89     |  nan        | nan        | nan        |     nan       |        nan         |
| ECG200    | MiniROCKET                |   0.88     |  nan        | nan        | nan        |     nan       |        nan         |
| ECG200    | DrCIF                     |   0.87     |  nan        | nan        | nan        |     nan       |        nan         |
| ECG200    | DTW (Literature)          |   0.88     |  nan        | nan        | nan        |     nan       |        nan         |
| Chinatown | Our Proposed (Kernel SVM) |   0.976676 |    0.965306 |   0.977314 |   0.97107  |       4.44683 |          0.695835  |
| Chinatown | Fast-DTW kNN              |   0.96793  |    0.949373 |   0.974601 |   0.960834 |      11.4138  |          0.0727158 |
| Chinatown | HIVE-COTE 2.0             |   0.983    |  nan        | nan        | nan        |     nan       |        nan         |
| Chinatown | MultiROCKET               |   0.981    |  nan        | nan        | nan        |     nan       |        nan         |
| Chinatown | MiniROCKET                |   0.978    |  nan        | nan        | nan        |     nan       |        nan         |
| Chinatown | DrCIF                     |   0.975    |  nan        | nan        | nan        |     nan       |        nan         |
| Chinatown | DTW (Literature)          |   0.965    |  nan        | nan        | nan        |     nan       |        nan         |

## 2. Ablation Study: 3-Feature Standard LFIG vs. 10-Feature Enhanced LFIG

| Dataset   |   Standard LFIG (3 Feat) Acc |   Our Enhanced LFIG (10 Feat) Acc |   Ablation Acc Drop |   Standard LFIG (3 Feat) F1 |   Our Enhanced LFIG (10 Feat) F1 |
|:----------|-----------------------------:|----------------------------------:|--------------------:|----------------------------:|---------------------------------:|
| GunPoint  |                     0.88     |                          0.906667 |           0.0266667 |                    0.879658 |                         0.90625  |
| Coffee    |                     0.964286 |                          1        |           0.0357143 |                    0.96424  |                         1        |
| ArrowHead |                     0.84     |                          0.828571 |          -0.0114286 |                    0.839415 |                         0.828463 |
| ECG200    |                     0.9      |                          0.91     |           0.01      |                    0.89011  |                         0.901736 |
| Chinatown |                     0.976676 |                          0.976676 |           0         |                    0.971251 |                         0.97107  |