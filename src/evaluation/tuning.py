"""
Nested cross-validation and automatic segmentation strategy selection
for the Adaptive Multi-Feature LFIG time series classification pipeline.
"""

import numpy as np
import pandas as pd
from itertools import product
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src.datasets.loader import load_ucr_dataset
from src.segmentation.adaptive import segment_time_series
from src.features.extractor import extract_granular_sequence
from src.similarity.hybrid import compute_pairwise_distances, fuse_distances
from src.classifiers.models import CustomDistanceKNN


def select_segmentation_strategy(X_train, y_train):
    """
    Automatically selects segmentation strategy (CPD vs fixed-window) using
    only training data. Uses cross-sample autocorrelation consistency:
    - Compute mean pairwise lag-1 autocorrelation across training samples
    - High consistency (low variance) → phase-aligned → fixed
    - Low consistency (high variance) → phase-shifted → CPD

    Returns: ('cpd', default_penalty) or ('fixed', default_window_size)
    """
    autocorrs = []
    for x in X_train:
        if len(x) < 3:
            continue
        r = np.corrcoef(x[:-1], x[1:])[0, 1]
        if not np.isnan(r):
            autocorrs.append(r)

    if len(autocorrs) < 2:
        # Not enough data to decide; default to CPD
        return 'cpd', 1.5

    var_autocorr = np.var(autocorrs)
    series_length = X_train.shape[1]

    if var_autocorr > 0.05:
        return 'cpd', 1.5
    else:
        return 'fixed', max(10, series_length // 10)


def _evaluate_pipeline(X_train, y_train, X_test, y_test, method, param, z, k, weights, clf_type):
    """
    Runs the full pipeline on given splits and returns accuracy.
    Pipeline: segment → extract features → compute distances → fuse → classify.
    """
    # Segment and extract features
    gran_train = []
    for x in X_train:
        bounds = segment_time_series(x, method=method, param=param)
        gran_train.append(extract_granular_sequence(x, bounds, z=z))

    gran_test = []
    for x in X_test:
        bounds = segment_time_series(x, method=method, param=param)
        gran_test.append(extract_granular_sequence(x, bounds, z=z))

    # Compute distances
    D_H_train, D_DTW_train, D_Cos_train = compute_pairwise_distances(gran_train)
    D_H_test, D_DTW_test, D_Cos_test = compute_pairwise_distances(gran_test, gran_train)

    # Fuse distances
    D_train_fused, mm_params = fuse_distances(D_H_train, D_DTW_train, D_Cos_train, weights=weights)
    D_test_fused, _ = fuse_distances(D_H_test, D_DTW_test, D_Cos_test, weights=weights, min_max_params=mm_params)

    # Classify
    if clf_type == 'KNN':
        clf = CustomDistanceKNN(n_neighbors=k)
        clf.fit(y_train)
        preds = clf.predict(D_test_fused)
    elif clf_type == 'Kernel SVM':
        from sklearn.svm import SVC
        median_d = np.median(D_train_fused)
        gamma = 1.0 / (2.0 * (median_d ** 2)) if median_d > 0 else 1.0
        K_train = np.exp(-gamma * (D_train_fused ** 2))
        K_test = np.exp(-gamma * (D_test_fused ** 2))
        svm = SVC(kernel='precomputed', random_state=42)
        svm.fit(K_train, y_train)
        preds = svm.predict(K_test)
    else:
        raise ValueError(f"Unknown clf_type: {clf_type}")

    return accuracy_score(y_test, preds)


def run_inner_cv(X_train_fold, y_train_fold, method, param_grid, n_inner_folds=3):
    """
    Runs inner cross-validation loop to select best hyperparameters.

    param_grid: dict with keys 'param', 'z', 'k', 'weights', 'clf_type'
    Uses StratifiedKFold for inner splits.
    Returns best_params dict with keys: param, z, k, weights, clf_type
    """
    skf = StratifiedKFold(n_splits=n_inner_folds, shuffle=True, random_state=42)

    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combos = list(product(*values))

    best_acc = -1.0
    best_params = None
    total = len(combos)

    for idx, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        fold_accs = []

        for train_idx, val_idx in skf.split(X_train_fold, y_train_fold):
            X_tr, X_val = X_train_fold[train_idx], X_train_fold[val_idx]
            y_tr, y_val = y_train_fold[train_idx], y_train_fold[val_idx]

            try:
                acc = _evaluate_pipeline(
                    X_tr, y_tr, X_val, y_val,
                    method=method,
                    param=params['param'],
                    z=params['z'],
                    k=params['k'],
                    weights=params['weights'],
                    clf_type=params['clf_type']
                )
                fold_accs.append(acc)
            except Exception as e:
                print(f"  Inner fold failed for {params}: {e}")
                fold_accs.append(0.0)

        mean_acc = np.mean(fold_accs)
        if mean_acc > best_acc:
            best_acc = mean_acc
            best_params = params

        if (idx + 1) % 10 == 0 or idx == total - 1:
            print(f"  Inner CV: {idx + 1}/{total} combos evaluated, best so far: {best_acc:.4f}")

    print(f"  Inner CV complete. Best accuracy: {best_acc:.4f}, params: {best_params}")
    return best_params


def run_nested_cv(dataset_name, n_outer_folds=5, n_inner_folds=3, data_dir='data'):
    """
    Runs full nested cross-validation for a dataset.

    Outer loop: stratified CV for unbiased performance estimation.
    Inner loop: stratified CV on each training fold for hyperparameter tuning.

    Returns dict with mean/std of accuracy, precision, recall, f1 across outer folds.
    """
    print(f"=== Nested CV for {dataset_name} ===")

    # Load and concatenate
    X_train, y_train, X_test, y_test = load_ucr_dataset(dataset_name, data_dir=data_dir)
    X_full = np.concatenate([X_train, X_test], axis=0)
    y_full = np.concatenate([y_train, y_test], axis=0)
    print(f"Full dataset: {X_full.shape[0]} samples, {X_full.shape[1]} timepoints")

    outer_skf = StratifiedKFold(n_splits=n_outer_folds, shuffle=True, random_state=42)

    # Default param grid
    param_grid = {
        'param': [1.0, 1.5, 2.0, 2.5],
        'z': [0.5, 1.0, 1.96],
        'k': [1, 3, 5],
        'weights': [[0.1, 0.8, 0.1], [0.3, 0.4, 0.3], [0.2, 0.6, 0.2]],
        'clf_type': ['KNN', 'Kernel SVM']
    }

    metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}

    for fold_i, (train_idx, test_idx) in enumerate(outer_skf.split(X_full, y_full)):
        print(f"\n--- Outer Fold {fold_i + 1}/{n_outer_folds} ---")
        X_tr, X_te = X_full[train_idx], X_full[test_idx]
        y_tr, y_te = y_full[train_idx], y_full[test_idx]

        # Auto-select segmentation strategy
        method, default_param = select_segmentation_strategy(X_tr, y_tr)
        print(f"  Auto-selected strategy: {method} (default param={default_param})")

        # Run inner CV
        best_params = run_inner_cv(X_tr, y_tr, method, param_grid, n_inner_folds=n_inner_folds)

        # Evaluate on outer test fold with best params
        try:
            acc = _evaluate_pipeline(
                X_tr, y_tr, X_te, y_te,
                method=method,
                param=best_params['param'],
                z=best_params['z'],
                k=best_params['k'],
                weights=best_params['weights'],
                clf_type=best_params['clf_type']
            )
        except Exception as e:
            print(f"  Outer fold evaluation failed: {e}")
            acc = 0.0

        # Compute detailed metrics via full pipeline predictions
        try:
            gran_train = [extract_granular_sequence(x, segment_time_series(x, method, best_params['param']),
                                                    z=best_params['z']) for x in X_tr]
            gran_test = [extract_granular_sequence(x, segment_time_series(x, method, best_params['param']),
                                                   z=best_params['z']) for x in X_te]
            D_H_tr, D_DTW_tr, D_Cos_tr = compute_pairwise_distances(gran_train)
            D_H_te, D_DTW_te, D_Cos_te = compute_pairwise_distances(gran_test, gran_train)
            D_tr_f, mm_p = fuse_distances(D_H_tr, D_DTW_tr, D_Cos_tr, weights=best_params['weights'])
            D_te_f, _ = fuse_distances(D_H_te, D_DTW_te, D_Cos_te, weights=best_params['weights'],
                                       min_max_params=mm_p)

            if best_params['clf_type'] == 'KNN':
                clf = CustomDistanceKNN(n_neighbors=best_params['k'])
                clf.fit(y_tr)
                preds = clf.predict(D_te_f)
            else:
                from sklearn.svm import SVC
                median_d = np.median(D_tr_f)
                gamma = 1.0 / (2.0 * (median_d ** 2)) if median_d > 0 else 1.0
                svm = SVC(kernel='precomputed', random_state=42)
                svm.fit(np.exp(-gamma * D_tr_f ** 2), y_tr)
                preds = svm.predict(np.exp(-gamma * D_te_f ** 2))

            prec, rec, f1, _ = precision_recall_fscore_support(y_te, preds, average='weighted', zero_division=0)
        except Exception as e:
            print(f"  Detailed metrics failed: {e}")
            prec, rec, f1 = 0.0, 0.0, 0.0

        metrics['accuracy'].append(acc)
        metrics['precision'].append(prec)
        metrics['recall'].append(rec)
        metrics['f1'].append(f1)
        print(f"  Fold {fold_i + 1} accuracy: {acc:.4f}, F1: {f1:.4f}")

    results = {
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
    print(f"\n=== {dataset_name} Results: Acc={results['accuracy_mean']:.4f}±{results['accuracy_std']:.4f} ===")
    return results


def validate_strategy_selection(dataset_names, data_dir='data'):
    """
    Validates auto-segmentation strategy across multiple datasets.
    For each dataset, runs select_segmentation_strategy and logs which strategy was chosen.

    Returns a DataFrame with columns: Dataset, Selected_Strategy, Selected_Param
    """
    rows = []
    for name in dataset_names:
        try:
            X_train, y_train, _, _ = load_ucr_dataset(name, data_dir=data_dir)
            method, param = select_segmentation_strategy(X_train, y_train)
            print(f"{name}: {method} (param={param})")
            rows.append({'Dataset': name, 'Selected_Strategy': method, 'Selected_Param': param})
        except Exception as e:
            print(f"{name}: FAILED ({e})")
            rows.append({'Dataset': name, 'Selected_Strategy': 'error', 'Selected_Param': None})

    return pd.DataFrame(rows)
