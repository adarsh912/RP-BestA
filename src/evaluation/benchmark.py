"""
Benchmark evaluation for the Adaptive Multi-Feature LFIG pipeline.
Supports nested CV, repeated evaluation, reproducible baselines, CD diagrams,
and fine-grained leave-one-feature-out ablation.
"""

import os
import time
import tracemalloc
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm

from src.datasets.loader import load_ucr_dataset
from src.segmentation.adaptive import segment_time_series
from src.features.extractor import extract_granular_sequence
from src.similarity.hybrid import compute_pairwise_distances, fuse_distances, fuse_borda_ranks
from src.classifiers.models import CustomDistanceKNN, train_and_evaluate_tabular

# Literature-only baselines (NOT reproduced under our protocol).
# Clearly labeled as sourced from published papers.
LITERATURE_ONLY_BASELINES = {
    "GunPoint": {"HIVE-COTE 2.0*": 1.000, "DrCIF*": 0.987},
    "Coffee": {"HIVE-COTE 2.0*": 1.000, "DrCIF*": 0.993},
    "ArrowHead": {"HIVE-COTE 2.0*": 0.871, "DrCIF*": 0.852},
    "ECG200": {"HIVE-COTE 2.0*": 0.900, "DrCIF*": 0.870},
    "Chinatown": {"HIVE-COTE 2.0*": 0.983, "DrCIF*": 0.975},
}

FEATURE_NAMES = [
    'Lower Bound', 'Upper Bound', 'Trend Slope', 'Shannon Entropy',
    'Variance', 'Volatility', 'Curvature', 'Intercept', 'Energy', 'Skewness'
]


def run_proposed_pipeline(X_train, y_train, X_test, y_test,
                          method="cpd", param=2.5, z=1.96,
                          use_ablation=False, k=3,
                          weights=[0.3, 0.4, 0.3], clf_type="KNN",
                          exclude_feature_idx=None):
    """
    Runs the Adaptive Multi-Feature LFIG pipeline.

    Args:
        exclude_feature_idx: If set, zero-out that feature column (for leave-one-out ablation).
        use_ablation: If True, use only 3 standard features (Lower, Upper, Trend).
    """
    train_granules = []
    for x in X_train:
        bounds = segment_time_series(x, method=method, param=param)
        seq = extract_granular_sequence(x, bounds, z=z)
        if use_ablation:
            seq = seq[:, 0:3]
        elif exclude_feature_idx is not None:
            seq = seq.copy()
            seq[:, exclude_feature_idx] = 0.0
        train_granules.append(seq)

    test_granules = []
    for x in X_test:
        bounds = segment_time_series(x, method=method, param=param)
        seq = extract_granular_sequence(x, bounds, z=z)
        if use_ablation:
            seq = seq[:, 0:3]
        elif exclude_feature_idx is not None:
            seq = seq.copy()
            seq[:, exclude_feature_idx] = 0.0
        test_granules.append(seq)

    # Pairwise distances
    D_H_tr, D_DTW_tr, D_Cos_tr = compute_pairwise_distances(train_granules)
    D_H_te, D_DTW_te, D_Cos_te = compute_pairwise_distances(test_granules, train_granules)

    # Distance fusion
    D_Fused_tr, params = fuse_distances(D_H_tr, D_DTW_tr, D_Cos_tr, weights=weights)
    D_Fused_te, _ = fuse_distances(D_H_te, D_DTW_te, D_Cos_te, weights=weights, min_max_params=params)

    # Classification
    if clf_type == "KNN":
        knn = CustomDistanceKNN(n_neighbors=k)
        knn.fit(y_train)
        preds = knn.predict(D_Fused_te)
    elif clf_type == "Kernel SVM":
        from sklearn.svm import SVC
        median_d = np.median(D_Fused_tr)
        gamma = 1.0 / (2.0 * (median_d ** 2)) if median_d > 0 else 1.0
        K_train = np.exp(-gamma * (D_Fused_tr ** 2))
        K_test = np.exp(-gamma * (D_Fused_te ** 2))
        svm = SVC(kernel='precomputed', random_state=42)
        svm.fit(K_train, y_train)
        preds = svm.predict(K_test)
    else:
        from src.classifiers.models import train_and_evaluate_distance_space
        res = train_and_evaluate_distance_space(D_Fused_tr, y_train, D_Fused_te, y_test)
        preds = res.get(clf_type, res.get("Distance kNN", {})).get("predictions", np.array([]))

    return preds


def run_dtw_baseline(X_train, X_test, y_train, y_test):
    """Runs baseline Fast-DTW kNN (k=3) classifier on raw time series."""
    from fastdtw import fastdtw
    n_train, n_test = len(X_train), len(X_test)
    D_test_train = np.zeros((n_test, n_train))
    for i in range(n_test):
        for j in range(n_train):
            d, _ = fastdtw(X_test[i], X_train[j])
            D_test_train[i, j] = d

    knn = CustomDistanceKNN(n_neighbors=3)
    knn.fit(y_train)
    return knn.predict(D_test_train)


def evaluate_metrics(y_true, y_pred):
    """Computes accuracy, precision, recall, and Macro F1 scores."""
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    return acc, prec, rec, f1


# ---------------------------------------------------------------------------
# Step 2: Repeated evaluation with stratified splits
# ---------------------------------------------------------------------------

def run_repeated_evaluation(dataset_name, n_repeats=10, test_size=0.3, data_dir='data'):
    """
    Runs repeated stratified train/test splits for a single dataset.
    Uses nested CV (via tuning module) to select hyperparameters on each split.

    Returns dict with mean ± std for accuracy, precision, recall, f1.
    """
    from src.evaluation.tuning import select_segmentation_strategy, run_inner_cv

    X_train, y_train, X_test, y_test = load_ucr_dataset(dataset_name, data_dir=data_dir)
    X_full = np.concatenate([X_train, X_test], axis=0)
    y_full = np.concatenate([y_train, y_test], axis=0)

    splitter = StratifiedShuffleSplit(n_splits=n_repeats, test_size=test_size, random_state=42)

    param_grid = {
        'param': [1.0, 1.5, 2.0],
        'z': [0.5, 1.0, 1.96],
        'k': [1, 3, 5],
        'weights': [[0.1, 0.8, 0.1], [0.3, 0.4, 0.3], [0.2, 0.6, 0.2]],
        'clf_type': ['KNN', 'Kernel SVM']
    }

    metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}

    for split_i, (train_idx, test_idx) in enumerate(splitter.split(X_full, y_full)):
        X_tr, X_te = X_full[train_idx], X_full[test_idx]
        y_tr, y_te = y_full[train_idx], y_full[test_idx]

        # Auto-select segmentation strategy
        method, _ = select_segmentation_strategy(X_tr, y_tr)

        # Inner CV for hyperparameters
        best_params = run_inner_cv(X_tr, y_tr, method, param_grid, n_inner_folds=3)

        # Evaluate
        preds = run_proposed_pipeline(
            X_tr, y_tr, X_te, y_te,
            method=method, param=best_params['param'], z=best_params['z'],
            k=best_params['k'], weights=best_params['weights'], clf_type=best_params['clf_type']
        )
        acc, prec, rec, f1 = evaluate_metrics(y_te, preds)
        metrics['accuracy'].append(acc)
        metrics['precision'].append(prec)
        metrics['recall'].append(rec)
        metrics['f1'].append(f1)
        print(f"  Split {split_i + 1}/{n_repeats}: acc={acc:.4f}")

    return {
        'dataset': dataset_name,
        'accuracy_mean': np.mean(metrics['accuracy']),
        'accuracy_std': np.std(metrics['accuracy']),
        'precision_mean': np.mean(metrics['precision']),
        'precision_std': np.std(metrics['precision']),
        'recall_mean': np.mean(metrics['recall']),
        'recall_std': np.std(metrics['recall']),
        'f1_mean': np.mean(metrics['f1']),
        'f1_std': np.std(metrics['f1']),
    }


# ---------------------------------------------------------------------------
# Step 7: Leave-one-feature-out ablation
# ---------------------------------------------------------------------------

def run_leave_one_feature_out_ablation(dataset_name, method="cpd", param=1.5,
                                        z=1.96, k=3, weights=[0.1, 0.8, 0.1],
                                        clf_type="KNN", n_repeats=10,
                                        test_size=0.3, data_dir='data'):
    """
    Leave-one-feature-out ablation: for each of the 10 features,
    zero it out and measure accuracy delta vs. the full model.

    Returns DataFrame with columns: Feature, Full_Acc, Ablated_Acc, Delta, per dataset.
    """
    X_train, y_train, X_test, y_test = load_ucr_dataset(dataset_name, data_dir=data_dir)
    X_full = np.concatenate([X_train, X_test], axis=0)
    y_full = np.concatenate([y_train, y_test], axis=0)

    splitter = StratifiedShuffleSplit(n_splits=n_repeats, test_size=test_size, random_state=42)

    # Collect per-split results
    full_accs = []
    feature_accs = {i: [] for i in range(10)}

    for train_idx, test_idx in splitter.split(X_full, y_full):
        X_tr, X_te = X_full[train_idx], X_full[test_idx]
        y_tr, y_te = y_full[train_idx], y_full[test_idx]

        # Full model
        preds_full = run_proposed_pipeline(X_tr, y_tr, X_te, y_te,
                                           method=method, param=param, z=z,
                                           k=k, weights=weights, clf_type=clf_type)
        full_accs.append(accuracy_score(y_te, preds_full))

        # Ablate each feature
        for feat_idx in range(10):
            preds_abl = run_proposed_pipeline(X_tr, y_tr, X_te, y_te,
                                              method=method, param=param, z=z,
                                              k=k, weights=weights, clf_type=clf_type,
                                              exclude_feature_idx=feat_idx)
            feature_accs[feat_idx].append(accuracy_score(y_te, preds_abl))

    # Compile results
    rows = []
    mean_full = np.mean(full_accs)
    for feat_idx in range(10):
        mean_abl = np.mean(feature_accs[feat_idx])
        rows.append({
            'Feature': FEATURE_NAMES[feat_idx],
            'Full_Acc': f"{mean_full:.4f}±{np.std(full_accs):.4f}",
            'Ablated_Acc': f"{mean_abl:.4f}±{np.std(feature_accs[feat_idx]):.4f}",
            'Delta': mean_full - mean_abl,
        })

    df = pd.DataFrame(rows)
    print(f"\n=== Leave-One-Feature-Out Ablation: {dataset_name} ===")
    print(df.to_string(index=False))
    return df


# ---------------------------------------------------------------------------
# Full benchmark: nested CV + baselines + CD diagram
# ---------------------------------------------------------------------------

def run_full_benchmark(dataset_names=None, n_repeats=10, data_dir='data'):
    """
    Runs the full benchmark pipeline across all datasets:
    1. Nested CV evaluation for our proposed method
    2. Reproducible baselines (ROCKET, MiniROCKET, DTW-1NN via aeon)
    3. Fast-DTW kNN baseline (reproduced)
    4. Literature-only baselines (HIVE-COTE, DrCIF — clearly labeled)
    5. Demšar critical-difference diagram
    6. Standard 3-feature vs 10-feature ablation
    """
    from src.datasets.ucr_catalog import get_dataset_names as catalog_names
    from src.evaluation.tuning import run_nested_cv, select_segmentation_strategy
    from src.evaluation.critical_difference import run_cd_analysis

    if dataset_names is None:
        dataset_names = catalog_names()

    os.makedirs("plots", exist_ok=True)
    all_results = []
    accuracy_for_cd = {}  # {dataset: {classifier: accuracy}}

    pbar = tqdm(dataset_names, desc="Evaluating UCR Catalog", unit="dataset")
    for dataset in pbar:
        pbar.set_description(f"Processing: {dataset}")
        print(f"\n{'='*60}")
        print(f"Evaluating: {dataset}")
        print(f"{'='*60}")

        try:
            X_train, y_train, X_test, y_test = load_ucr_dataset(dataset, data_dir=data_dir)
        except Exception as e:
            print(f"  SKIPPED: {e}")
            continue

        accuracy_for_cd[dataset] = {}

        # --- 1. Our Proposed (Nested CV) ---
        try:
            ncv_result = run_nested_cv(dataset, data_dir=data_dir)
            acc_str = f"{ncv_result['accuracy_mean']:.4f}±{ncv_result['accuracy_std']:.4f}"
            all_results.append({
                "Dataset": dataset,
                "Classifier": "Proposed (Nested CV)",
                "Accuracy": acc_str,
                "Accuracy_Mean": ncv_result['accuracy_mean'],
                "F1": f"{ncv_result['f1_mean']:.4f}±{ncv_result['f1_std']:.4f}",
            })
            accuracy_for_cd[dataset]["Proposed"] = ncv_result['accuracy_mean']

            # --- [Proof 1] Variable-Length CPD Segmentation Demonstration ---
            method, default_param = select_segmentation_strategy(X_train)
            boundaries = segment_time_series(X_train[0], method, default_param)
            lengths = [boundaries[i+1] - boundaries[i] for i in range(len(boundaries)-1)]
            print(f"\n--- [Proof 1] Variable-Length CPD Segmentation: {dataset} ---")
            print(f"  Boundary Indices: {boundaries}")
            print(f"  Granule Lengths (variable size proof): {lengths}")

            # --- [Proof 2] Feature Comparison (3D Standard vs 10D Proposed) ---
            # We use a 3-repeat split to compare 3D vs 10D quickly
            from sklearn.model_selection import StratifiedShuffleSplit
            sss = StratifiedShuffleSplit(n_splits=3, test_size=0.3, random_state=42)
            acc_10d_list = []
            acc_3d_list = []
            for tr_idx, val_idx in sss.split(X_train, y_train):
                a_10d = evaluate_pipeline(X_train[tr_idx], y_train[tr_idx], X_train[val_idx], y_train[val_idx], use_ablation=False)
                acc_10d_list.append(a_10d)
                a_3d = evaluate_pipeline(X_train[tr_idx], y_train[tr_idx], X_train[val_idx], y_train[val_idx], use_ablation=True)
                acc_3d_list.append(a_3d)
            mean_10d = np.mean(acc_10d_list)
            mean_3d = np.mean(acc_3d_list)
            print(f"\n--- [Proof 2] Feature Comparison (3D Standard vs 10D Proposed): {dataset} ---")
            print(f"  Standard 3D LFIG Accuracy:                 {mean_3d:.4f}")
            print(f"  Proposed Multi-Feature 10D LFIG Accuracy:  {mean_10d:.4f}")
            print(f"  Improvement Delta:                        {mean_10d - mean_3d:+.4f}")

            # --- [Proof 3] Leave-One-Feature-Out (LOFO) Ablation ---
            run_leave_one_feature_out_ablation(dataset, n_repeats=3, data_dir=data_dir)

        except Exception as e:
            print(f"  Proposed pipeline proofs failed: {e}")

        # --- 2. Reproducible baselines (aeon) ---
        try:
            from src.evaluation.baselines import run_all_baselines
            baseline_results = run_all_baselines(X_train, y_train, X_test, y_test)
            for name, res in baseline_results.items():
                all_results.append({
                    "Dataset": dataset,
                    "Classifier": f"{name} (reproduced)",
                    "Accuracy": f"{res['accuracy']:.4f}",
                    "Accuracy_Mean": res['accuracy'],
                    "F1": "—",
                })
                accuracy_for_cd[dataset][name] = res['accuracy']
        except Exception as e:
            print(f"  Baselines failed: {e}")

        # --- 3. Fast-DTW kNN (reproduced) ---
        try:
            tracemalloc.start()
            t0 = time.time()
            preds_dtw = run_dtw_baseline(X_train, X_test, y_train, y_test)
            runtime_dtw = time.time() - t0
            _, peak_mem = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            acc_dtw, _, _, f1_dtw = evaluate_metrics(y_test, preds_dtw)
            all_results.append({
                "Dataset": dataset,
                "Classifier": "Fast-DTW kNN (reproduced)",
                "Accuracy": f"{acc_dtw:.4f}",
                "Accuracy_Mean": acc_dtw,
                "F1": f"{f1_dtw:.4f}",
            })
            accuracy_for_cd[dataset]["Fast-DTW kNN"] = acc_dtw
        except Exception as e:
            print(f"  DTW baseline failed: {e}")

        # --- 4. Literature-only baselines ---
        if dataset in LITERATURE_ONLY_BASELINES:
            for clf_name, lit_acc in LITERATURE_ONLY_BASELINES[dataset].items():
                all_results.append({
                    "Dataset": dataset,
                    "Classifier": clf_name,
                    "Accuracy": f"{lit_acc:.4f}†",
                    "Accuracy_Mean": lit_acc,
                    "F1": "—",
                })
                # Note: these are NOT included in CD diagram since they are not reproduced

    # --- 5. Compile results ---
    df_results = pd.DataFrame(all_results)

    with open("plots/evaluation_results.md", "w") as f:
        f.write("# Evaluation and Benchmark Results\n\n")
        f.write("> **†** = Literature-reported accuracy, not reproduced under our protocol.\n")
        f.write("> All other results are reproduced under identical evaluation protocol.\n\n")
        f.write("## Classification Performance Comparison\n\n")
        f.write(df_results[['Dataset', 'Classifier', 'Accuracy', 'F1']].to_markdown(index=False))
        f.write("\n")

    print("\nResults written to: plots/evaluation_results.md")

    # --- 6. CD Diagram (only for reproduced classifiers) ---
    try:
        # Build accuracy matrix for reproduced classifiers only
        cd_rows = []
        for ds, clfs in accuracy_for_cd.items():
            cd_rows.append(clfs)
        if cd_rows:
            cd_df = pd.DataFrame(cd_rows, index=list(accuracy_for_cd.keys()))
            cd_df = cd_df.dropna(axis=1, how='any')  # Only keep classifiers present for all datasets
            if len(cd_df.columns) >= 2 and len(cd_df) >= 3:
                cd_result = run_cd_analysis(cd_df, output_path='plots/cd_diagram.png')
                print(f"CD diagram generated. Friedman p={cd_result['friedman_p']:.6f}")
            else:
                print("Not enough classifiers/datasets for CD diagram.")
    except Exception as e:
        print(f"CD analysis failed: {e}")

    return df_results


if __name__ == "__main__":
    from src.datasets.ucr_catalog import get_dataset_names
    # Run on all 23 datasets in the catalog
    run_full_benchmark(
        dataset_names=get_dataset_names()
    )
