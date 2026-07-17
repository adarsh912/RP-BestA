"""
Nested cross-validation and automatic segmentation strategy selection
for the Adaptive Multi-Feature LFIG time series classification pipeline.
Optimized with distance caching to prevent redundant DTW computations during grid search.
"""

import numpy as np
import pandas as pd
from itertools import product
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.svm import SVC

from src.datasets.loader import load_ucr_dataset
from src.segmentation.adaptive import segment_time_series
from src.features.extractor import extract_granular_sequence
from src.similarity.hybrid import compute_pairwise_distances, fuse_distances
from src.classifiers.models import CustomDistanceKNN


def select_segmentation_strategy(X_train):
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
        return 'cpd', 1.5

    var_autocorr = np.var(autocorrs)
    series_length = X_train.shape[1]

    if var_autocorr > 0.05:
        return 'cpd', 1.5
    else:
        return 'fixed', max(10, series_length // 10)


# Hyperparameter search grid
PARAM_GRID = {
    'z': [1.0, 1.96],
    'k': [1, 3],
    'weights': [[0.3, 0.4, 0.3], [0.33, 0.34, 0.33]],
    'clf_type': ['KNN', 'Kernel SVM'],
}


def _compute_distance_cache(X_tr, X_te, method, param, z):
    """Computes and caches distance components to avoid redundant DTW runs."""
    gran_tr = [extract_granular_sequence(x, segment_time_series(x, method, param), z=z) for x in X_tr]
    gran_te = [extract_granular_sequence(x, segment_time_series(x, method, param), z=z) for x in X_te]
    D_H_tr, D_DTW_tr, D_Cos_tr = compute_pairwise_distances(gran_tr)
    D_H_te, D_DTW_te, D_Cos_te = compute_pairwise_distances(gran_te, gran_tr)
    return D_H_tr, D_DTW_tr, D_Cos_tr, D_H_te, D_DTW_te, D_Cos_te


def _classify_from_cache(dist_cache, y_tr, y_te, weights, k, clf_type):
    """Runs distance fusion and classification directly from cached distance matrices."""
    D_H_tr, D_DTW_tr, D_Cos_tr, D_H_te, D_DTW_te, D_Cos_te = dist_cache
    D_tr, params = fuse_distances(D_H_tr, D_DTW_tr, D_Cos_tr, weights)
    D_te, _ = fuse_distances(D_H_te, D_DTW_te, D_Cos_te, weights, params)
    
    if clf_type == 'KNN':
        clf = CustomDistanceKNN(n_neighbors=k).fit(y_tr)
        preds = clf.predict(D_te)
    elif clf_type == 'Kernel SVM':
        med = np.median(D_tr)
        gamma = 1.0 / (2 * med ** 2) if med > 0 else 1.0
        K_tr, K_te = np.exp(-gamma * D_tr ** 2), np.exp(-gamma * D_te ** 2)
        svm = SVC(kernel='precomputed', random_state=42).fit(K_tr, y_tr)
        preds = svm.predict(K_te)
    else:
        raise ValueError(f"Unknown clf_type: {clf_type}")
        
    return preds


def _accuracy_from_cache(dist_cache, y_tr, y_te, weights, k, clf_type):
    """Utility to compute accuracy from cached distances."""
    preds = _classify_from_cache(dist_cache, y_tr, y_te, weights, k, clf_type)
    return accuracy_score(y_te, preds)


def run_inner_cv(X_tr_fold, y_tr_fold, method, default_param, n_inner=3):
    """
    Selects optimal hyperparameters using ONLY the training fold.
    Caches distances for each z level to optimize computation.
    """
    skf = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=42)
    splits = list(skf.split(X_tr_fold, y_tr_fold))
    
    z_values = PARAM_GRID['z']
    other_combos = list(product(PARAM_GRID['k'], PARAM_GRID['weights'], PARAM_GRID['clf_type']))
    
    best_acc, best_params = -1.0, None
    total_combos = len(z_values) * len(other_combos)
    
    print(f"  Running inner CV with {total_combos} parameter combinations...")
    
    for z in z_values:
        caches = []
        # Precompute and cache distances for this z level across all splits
        for inner_tr_idx, inner_val_idx in splits:
            cache = _compute_distance_cache(
                X_tr_fold[inner_tr_idx], X_tr_fold[inner_val_idx],
                method, default_param, z
            )
            caches.append((cache, y_tr_fold[inner_tr_idx], y_tr_fold[inner_val_idx]))
            
        # Loop through other hyperparameters rapidly using the cache
        for k, weights, clf_type in other_combos:
            accs = []
            for cache, y_tr_split, y_val_split in caches:
                try:
                    acc = _accuracy_from_cache(cache, y_tr_split, y_val_split, weights, k, clf_type)
                except Exception:
                    acc = 0.0
                accs.append(acc)
            mean_acc = np.mean(accs)
            if mean_acc > best_acc:
                best_acc = mean_acc
                best_params = {'z': z, 'k': k, 'weights': weights, 'clf_type': clf_type}
                
    return best_params


def run_nested_cv(dataset_name, n_outer=5, n_inner=3, data_dir='data'):
    """
    Runs full nested cross-validation for a dataset.
    Outer loop: 5-fold stratified CV for reporting.
    Inner loop: 3-fold stratified CV on each training fold for tuning.
    """
    print(f"=== Nested CV for {dataset_name} ===")

    # Load dataset
    X_train, y_train, X_test, y_test = load_ucr_dataset(dataset_name, data_dir=data_dir)
    X = np.concatenate([X_train, X_test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0)
    print(f"Dataset shape: {X.shape[0]} samples, {X.shape[1]} timepoints")

    outer = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=42)
    metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}

    for fold_i, (tr_idx, te_idx) in enumerate(outer.split(X, y)):
        print(f"\n--- Outer Fold {fold_i + 1}/{n_outer} ---")
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        # 1. Strategy chosen from training data only
        method, default_param = select_segmentation_strategy(X_tr)
        print(f"  Selected strategy: {method} (default param={default_param})")

        # 2. Hyperparameters chosen by inner CV on training fold only
        best_params = run_inner_cv(X_tr, y_tr, method, default_param, n_inner=n_inner)
        print(f"  Best inner hyperparameters: {best_params}")

        # 3. Final evaluation on the untouched outer test fold
        cache = _compute_distance_cache(X_tr, X_te, method, default_param, best_params['z'])
        preds = _classify_from_cache(
            cache, y_tr, y_te,
            best_params['weights'], best_params['k'], best_params['clf_type']
        )
        
        acc = accuracy_score(y_te, preds)
        prec, rec, f1, _ = precision_recall_fscore_support(y_te, preds, average='macro', zero_division=0)
        
        metrics['accuracy'].append(acc)
        metrics['precision'].append(prec)
        metrics['recall'].append(rec)
        metrics['f1'].append(f1)
        print(f"  Fold {fold_i+1} Accuracy: {acc:.4f}  F1: {f1:.4f}")

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
    """
    rows = []
    for name in dataset_names:
        try:
            X_train, y_train, _, _ = load_ucr_dataset(name, data_dir=data_dir)
            method, param = select_segmentation_strategy(X_train)
            print(f"{name}: {method} (param={param})")
            rows.append({'Dataset': name, 'Selected_Strategy': method, 'Selected_Param': param})
        except Exception as e:
            print(f"{name}: FAILED ({e})")
            rows.append({'Dataset': name, 'Selected_Strategy': 'error', 'Selected_Param': None})

    return pd.DataFrame(rows)
