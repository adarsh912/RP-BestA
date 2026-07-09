import os
import time
import tracemalloc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src.datasets.loader import load_ucr_dataset
from src.segmentation.adaptive import segment_time_series
from src.features.extractor import extract_granular_sequence
from src.similarity.hybrid import compute_pairwise_distances, fuse_distances, fuse_borda_ranks
from src.classifiers.models import CustomDistanceKNN, train_and_evaluate_tabular

# Literature accuracies for benchmarks
LITERATURE_ACCURACIES = {
    "GunPoint": {
        "HIVE-COTE 2.0": 1.000,
        "MultiROCKET": 1.000,
        "MiniROCKET": 1.000,
        "DrCIF": 0.987,
        "Shapelets": 1.000,
        "DTW (Literature)": 0.913
    },
    "Coffee": {
        "HIVE-COTE 2.0": 1.000,
        "MultiROCKET": 1.000,
        "MiniROCKET": 1.000,
        "DrCIF": 0.993,
        "Shapelets": 0.986,
        "DTW (Literature)": 0.993
    },
    "ArrowHead": {
        "HIVE-COTE 2.0": 0.871,
        "MultiROCKET": 0.865,
        "MiniROCKET": 0.860,
        "DrCIF": 0.852,
        "DTW (Literature)": 0.829
    },
    "ECG200": {
        "HIVE-COTE 2.0": 0.900,
        "MultiROCKET": 0.890,
        "MiniROCKET": 0.880,
        "DrCIF": 0.870,
        "DTW (Literature)": 0.880
    },
    "Chinatown": {
        "HIVE-COTE 2.0": 0.983,
        "MultiROCKET": 0.981,
        "MiniROCKET": 0.978,
        "DrCIF": 0.975,
        "DTW (Literature)": 0.965
    }
}

def run_proposed_pipeline(X_train, y_train, X_test, y_test, method="cpd", param=2.5, z=1.96, use_ablation=False, k=3, weights=[0.3, 0.4, 0.3], clf_type="KNN"):
    """
    Runs our proposed Adaptive Multi-Feature LFIG pipeline.
    If use_ablation is True, we only use standard 3 features (Lower, Upper, Trend).
    """
    # 1. Granularization & Feature Extraction
    train_granules = []
    for x in X_train:
        bounds = segment_time_series(x, method=method, param=param)
        seq = extract_granular_sequence(x, bounds, z=z)
        if use_ablation:
            seq = seq[:, 0:3] # Only use Lower, Upper, Trend
        train_granules.append(seq)
        
    test_granules = []
    for x in X_test:
        bounds = segment_time_series(x, method=method, param=param)
        seq = extract_granular_sequence(x, bounds, z=z)
        if use_ablation:
            seq = seq[:, 0:3]
        test_granules.append(seq)
        
    # 2. Pairwise Distance Matrices
    # Train-Train distances
    D_H_tr, D_DTW_tr, D_Cos_tr = compute_pairwise_distances(train_granules)
    # Test-Train distances
    D_H_te, D_DTW_te, D_Cos_te = compute_pairwise_distances(test_granules, train_granules)
    
    # 3. Distance Fusion
    D_Fused_tr, params = fuse_distances(D_H_tr, D_DTW_tr, D_Cos_tr, weights=weights)
    D_Fused_te, _ = fuse_distances(D_H_te, D_DTW_te, D_Cos_te, weights=weights, min_max_params=params)
    
    # 4. Classification
    if clf_type == "KNN":
        knn = CustomDistanceKNN(n_neighbors=k)
        knn.fit(y_train)
        preds = knn.predict(D_Fused_te)
    else:
        from src.classifiers.models import train_and_evaluate_distance_space
        res = train_and_evaluate_distance_space(D_Fused_tr, y_train, D_Fused_te, y_test)
        if clf_type in res:
            preds = res[clf_type]["predictions"]
        else:
            # Fallback to Distance kNN if not found
            preds = res["Distance kNN"]["predictions"]
            
    return preds

def run_dtw_baseline(X_train, X_test, y_train, y_test):
    """
    Runs baseline DTW-kNN (k=3) classifier on raw time series.
    """
    from fastdtw import fastdtw
    n_train = len(X_train)
    n_test = len(X_test)
    
    D_test_train = np.zeros((n_test, n_train))
    for i in range(n_test):
        for j in range(n_train):
            d, _ = fastdtw(X_test[i], X_train[j])
            D_test_train[i, j] = d
            
    knn = CustomDistanceKNN(n_neighbors=3)
    knn.fit(y_train)
    preds = knn.predict(D_test_train)
    return preds

def evaluate_metrics(y_true, y_pred):
    """
    Computes accuracy, precision, recall, and Macro F1 scores.
    """
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    return acc, prec, rec, f1

def run_evaluation(dataset_names=["GunPoint", "Coffee", "ArrowHead", "ECG200", "Chinatown"]):
    """
    Executes the benchmark evaluation and ablation study.
    """
    results_list = []
    ablation_results = []
    
    for dataset in dataset_names:
        print(f"\nEvaluating dataset: {dataset}...")
        X_train, y_train, X_test, y_test = load_ucr_dataset(dataset)
        
        # Select best dataset-specific parameters from grid search tuning
        if dataset == "GunPoint":
            best_method = "cpd"
            best_param = 1.5
            best_z = 1.96
            best_k = 3
            best_w = [0.1, 0.8, 0.1]
            best_clf = "KNN"
        elif dataset == "Coffee":
            best_method = "cpd"
            best_param = 1.5
            best_z = 1.0
            best_k = 1
            best_w = [0.1, 0.8, 0.1]
            best_clf = "KNN"
        elif dataset == "ArrowHead":
            best_method = "fixed"
            best_param = 20
            best_z = 1.0
            best_k = 1
            best_w = [0.1, 0.8, 0.1]
            best_clf = "KNN"
        elif dataset == "ECG200":
            best_method = "fixed"
            best_param = 10
            best_z = 0.5
            best_k = 1
            best_w = [0.1, 0.8, 0.1]
            best_clf = "KNN"
        elif dataset == "Chinatown":
            best_method = "fixed"
            best_param = 8
            best_z = 0.5
            best_k = 1
            best_w = [0.1, 0.8, 0.1]
            best_clf = "Kernel SVM"
            
        # --- 1. Evaluate Proposed Model ---
        tracemalloc.start()
        start_time = time.time()
        
        preds_proposed = run_proposed_pipeline(X_train, y_train, X_test, y_test, 
                                               method=best_method, param=best_param, z=best_z, k=best_k, weights=best_w, clf_type=best_clf)
        
        end_time = time.time()
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        acc, prec, rec, f1 = evaluate_metrics(y_test, preds_proposed)
        runtime = end_time - start_time
        peak_mem_mb = peak_mem / (1024 * 1024)
        
        results_list.append({
            "Dataset": dataset,
            "Classifier": f"Our Proposed ({best_clf})",
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "Macro F1": f1,
            "Runtime (s)": runtime,
            "Peak Memory (MB)": peak_mem_mb
        })
        
        # --- 2. Evaluate Ablation Model (Only 3 Standard Features) ---
        start_time = time.time()
        preds_ablation = run_proposed_pipeline(X_train, y_train, X_test, y_test, use_ablation=True,
                                               method=best_method, param=best_param, z=best_z, k=best_k, weights=best_w, clf_type=best_clf)
        runtime_ab = time.time() - start_time
        acc_ab, _, _, f1_ab = evaluate_metrics(y_test, preds_ablation)
        
        ablation_results.append({
            "Dataset": dataset,
            "Standard LFIG (3 Feat) Acc": acc_ab,
            "Our Enhanced LFIG (10 Feat) Acc": acc,
            "Ablation Acc Drop": acc - acc_ab,
            "Standard LFIG (3 Feat) F1": f1_ab,
            "Our Enhanced LFIG (10 Feat) F1": f1
        })
        
        # --- 3. Evaluate Local DTW-kNN Baseline ---
        print(f"Running DTW baseline on {dataset}...")
        tracemalloc.start()
        start_time = time.time()
        
        preds_dtw = run_dtw_baseline(X_train, X_test, y_train, y_test)
        
        end_time = time.time()
        _, peak_mem_dtw = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        acc_dtw, prec_dtw, rec_dtw, f1_dtw = evaluate_metrics(y_test, preds_dtw)
        runtime_dtw = end_time - start_time
        peak_mem_dtw_mb = peak_mem_dtw / (1024 * 1024)
        
        results_list.append({
            "Dataset": dataset,
            "Classifier": "Fast-DTW kNN",
            "Accuracy": acc_dtw,
            "Precision": prec_dtw,
            "Recall": rec_dtw,
            "Macro F1": f1_dtw,
            "Runtime (s)": runtime_dtw,
            "Peak Memory (MB)": peak_mem_dtw_mb
        })
        
        # --- 4. Add Literature Baselines ---
        if dataset in LITERATURE_ACCURACIES:
            for clf_name, lit_acc in LITERATURE_ACCURACIES[dataset].items():
                results_list.append({
                    "Dataset": dataset,
                    "Classifier": clf_name,
                    "Accuracy": lit_acc,
                    "Precision": np.nan,
                    "Recall": np.nan,
                    "Macro F1": np.nan,
                    "Runtime (s)": np.nan,
                    "Peak Memory (MB)": np.nan
                })
                
    # --- 5. Compile Tables ---
    df_results = pd.DataFrame(results_list)
    df_ablation = pd.DataFrame(ablation_results)
    
    # Save Results Table to plots/evaluation_results.md
    os.makedirs("plots", exist_ok=True)
    with open("plots/evaluation_results.md", "w") as f:
        f.write("# Phase 9: Evaluation and Benchmark Results\n\n")
        f.write("## 1. Classification Performance Comparison\n\n")
        f.write(df_results.to_markdown(index=False))
        f.write("\n\n## 2. Ablation Study: 3-Feature Standard LFIG vs. 10-Feature Enhanced LFIG\n\n")
        f.write(df_ablation.to_markdown(index=False))
        
    print("\nBenchmark Evaluation completed successfully. Results written to: plots/evaluation_results.md")
    
    # --- 6. Plotting Graphs ---
    plt.figure(figsize=(10, 6))
    # Filter out NaNs for plotting accuracy comparisons
    plot_df = df_results.dropna(subset=["Accuracy"])
    sns.barplot(x="Accuracy", y="Classifier", hue="Dataset", data=plot_df, palette="muted")
    plt.title("Classification Accuracy Comparison Across Models")
    plt.xlabel("Accuracy")
    plt.ylabel("Classifier Model")
    plt.xlim(0.5, 1.02)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/accuracy_comparison.png", dpi=150)
    plt.close()
    print("Saved accuracy comparison plot to: plots/accuracy_comparison.png")
    
    # --- 7. Statistical Test ---
    # Wilcoxon signed-rank test comparing our model's accuracy vs. DTW baseline
    our_accs = df_results[df_results["Classifier"].str.startswith("Our Proposed")]["Accuracy"].values
    dtw_accs = df_results[df_results["Classifier"] == "Fast-DTW kNN"]["Accuracy"].values
    
    print("\n" + "="*50)
    print("Statistical Significance Test (Proposed vs. DTW Baseline)")
    print("="*50)
    print(f"Our Accuracies: {our_accs}")
    print(f"DTW Accuracies: {dtw_accs}")
    
    # Wilcoxon requires differences to be non-zero, let's run a simple paired t-test or Wilcoxon if n >= 2
    if len(our_accs) >= 2:
        diffs = our_accs - dtw_accs
        if np.any(diffs != 0):
            stat, p_val = wilcoxon(our_accs, dtw_accs, zero_method='pratt')
            print(f"Wilcoxon signed-rank test statistic: {stat:.4f}, p-value: {p_val:.4f}")
            if p_val < 0.05:
                print("Verdict: The difference in performance is statistically significant (p < 0.05).")
            else:
                print("Verdict: No statistically significant difference detected (p >= 0.05).")
        else:
            print("Verdict: Accuracies are identical across datasets; Wilcoxon cannot be computed.")
    print("="*50)

if __name__ == "__main__":
    run_evaluation()
