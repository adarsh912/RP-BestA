"""
Standalone Google Colab Script for Adaptive Multi-Feature LFIG Classification.

This script contains all the core pipeline code (segmentation, LFIG granulation,
feature extraction, hybrid distance fusion with weight learning, and custom kNN classification)
packaged into a single self-contained file. It loads UCR datasets online via the `aeon` library.

Instructions:
1. Copy the entire contents of this file.
2. Paste it into a single cell in a Google Colab notebook.
3. Run the cell.
"""

import sys
import os
import time
import subprocess

# 1. Check and install dependencies if running in Colab
def install_dependencies():
    try:
        import aeon
        import ruptures
        import fastdtw
        print("Dependencies already satisfied.")
    except ImportError:
        print("Installing required packages in Google Colab...")
        packages = ["aeon>=0.7.0", "ruptures>=1.1.5", "fastdtw>=0.3.4", "scikit-learn", "numpy", "scipy", "pandas", "tqdm"]
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + packages)
        print("All packages successfully installed.")

# install_dependencies will be called inside main execution block if needed

# Import packages
import numpy as np
import scipy.stats
import ruptures as rpt
import time
from fastdtw import fastdtw
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from itertools import product
from tqdm import tqdm

# ---------------------------------------------------------------------------
# 2. Pipeline Core Modules
# ---------------------------------------------------------------------------

# --- Segmentation ---
def compute_shannon_entropy(segment, bins=10):
    if len(segment) < 2 or np.std(segment) < 1e-9:
        return 0.0
    counts, _ = np.histogram(segment, bins=bins)
    probs = counts / len(segment)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))

def fixed_segmentation(X_series, window_size=10):
    N = len(X_series)
    boundaries = list(range(0, N, window_size))
    if boundaries[-1] != N:
        boundaries.append(N)
    return boundaries

def cpd_segmentation(X_series, penalty=2.0, model="l2", min_size=3):
    N = len(X_series)
    if N <= min_size * 2:
        return [0, N]
    try:
        signal = np.asarray(X_series, dtype=np.float64).reshape(-1, 1)
        algo = rpt.BottomUp(model=model, min_size=min_size).fit(signal)
        result = algo.predict(pen=penalty)
        if len(result) == 0:
            result = [0, N]
        elif result[0] != 0:
            result = [0] + result
        return result
    except Exception as e:
        return fixed_segmentation(X_series, window_size=max(10, N // 10))

def segment_time_series(X_series, method="cpd", param=2.0, min_size=3):
    if method == "fixed":
        return fixed_segmentation(X_series, window_size=int(param))
    elif method == "cpd":
        return cpd_segmentation(X_series, penalty=float(param), min_size=min_size)
    else:
        raise ValueError(f"Unknown segmentation method: {method}")

# --- LFIG Granulation ---
def construct_lfig_granule(segment, z=1.96):
    L = len(segment)
    tau = np.arange(1, L + 1)
    if L < 2:
        return {
            "slope": 0.0, "intercept": segment[0] if L == 1 else 0.0, "std_residuals": 0.0,
            "lower_bound_mean": segment[0] if L == 1 else 0.0, "upper_bound_mean": segment[0] if L == 1 else 0.0
        }
    a, b = np.polyfit(tau, segment, 1)
    residuals = segment - (a * tau + b)
    sigma = np.std(residuals)
    lower = a * tau + b - z * sigma
    upper = a * tau + b + z * sigma
    return {
        "slope": a,
        "intercept": b,
        "std_residuals": sigma,
        "lower_bound_mean": np.mean(lower),
        "upper_bound_mean": np.mean(upper)
    }

# --- Feature Extraction ---
def extract_granule_features(segment, z=1.96):
    L = len(segment)
    tau = np.arange(1, L + 1)
    g = construct_lfig_granule(segment, z=z)
    entropy = compute_shannon_entropy(segment)
    variance = np.var(segment) if L > 1 else 0.0
    volatility = np.std(np.diff(segment)) if L > 2 else 0.0
    c = np.polyfit(tau, segment, 2)[0] if L >= 3 else 0.0
    energy = float(np.sum(segment ** 2))
    skewness = float(scipy.stats.skew(segment)) if (L >= 3 and variance > 1e-9) else 0.0
    
    return np.array([
        g["lower_bound_mean"],  # 1
        g["upper_bound_mean"],  # 2
        g["slope"],             # 3
        entropy,                # 4
        variance,               # 5
        volatility,             # 6
        c,                      # 7
        g["intercept"],         # 8
        energy,                 # 9
        skewness                # 10
    ])

def extract_granular_sequence(X_series, boundaries, z=1.96):
    seq = []
    for j in range(len(boundaries) - 1):
        start, end = boundaries[j], boundaries[j+1]
        segment = X_series[start:end]
        if len(segment) == 0:
            continue
        seq.append(extract_granule_features(segment, z=z))
    if len(seq) == 0:
        return np.empty((0, 10))
    return np.array(seq)

# --- Similarity & Fusion ---
def interval_hausdorff_dist(L1, U1, L2, U2):
    return max(abs(L1 - L2), abs(U1 - U2))

def sequence_hausdorff_distance(seq_P, seq_Q):
    K, M = len(seq_P), len(seq_Q)
    dist_matrix = np.zeros((K, M))
    for k in range(K):
        for m in range(M):
            dist_matrix[k, m] = interval_hausdorff_dist(
                seq_P[k, 0], seq_P[k, 1], seq_Q[m, 0], seq_Q[m, 1]
            )
    return max(np.max(np.min(dist_matrix, axis=1)), np.max(np.min(dist_matrix, axis=0)))

def cosine_distance(v1, v2):
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    return 1.0 - np.dot(v1, v2) / (n1 * n2) if (n1 > 1e-9 and n2 > 1e-9) else 1.0

def compute_distances(seq_P, seq_Q):
    d_H = sequence_hausdorff_distance(seq_P[:, 0:2], seq_Q[:, 0:2])
    d_DTW, _ = fastdtw(seq_P[:, 2], seq_Q[:, 2], dist=lambda x, y: abs(x - y))
    d_Cos, _ = fastdtw(seq_P, seq_Q, dist=cosine_distance)
    return d_H, d_DTW, d_Cos

def min_max_normalize(D, d_min=None, d_max=None):
    if d_min is None or d_max is None:
        d_min, d_max = np.min(D), np.max(D)
    if abs(d_max - d_min) < 1e-9:
        return np.zeros_like(D), d_min, d_max
    return (D - d_min) / (d_max - d_min), d_min, d_max

def compute_pairwise_distances(X, Y=None):
    N_X = len(X)
    symmetric = Y is None
    N_Y = N_X if symmetric else len(Y)
    D_H, D_DTW, D_Cos = np.zeros((N_X, N_Y)), np.zeros((N_X, N_Y)), np.zeros((N_X, N_Y))
    dataset_Y = X if symmetric else Y
    for i in range(N_X):
        start_j = i if symmetric else 0
        for j in range(start_j, N_Y):
            dh, ddtw, dcos = compute_distances(X[i], dataset_Y[j])
            D_H[i, j] = dh
            D_DTW[i, j] = ddtw
            D_Cos[i, j] = dcos
            if symmetric:
                D_H[j, i] = dh
                D_DTW[j, i] = ddtw
                D_Cos[j, i] = dcos
    return D_H, D_DTW, D_Cos

def fuse_distances(D_H, D_DTW, D_Cos, weights=[0.3, 0.4, 0.3], min_max_params=None):
    w = weights
    if min_max_params is None:
        D_H_norm, min_H, max_H = min_max_normalize(D_H)
        D_DTW_norm, min_DTW, max_DTW = min_max_normalize(D_DTW)
        D_Cos_norm, min_Cos, max_Cos = min_max_normalize(D_Cos)
        params = (min_H, max_H, min_DTW, max_DTW, min_Cos, max_Cos)
    else:
        min_H, max_H, min_DTW, max_DTW, min_Cos, max_Cos = min_max_params
        D_H_norm = np.clip((D_H - min_H) / (max_H - min_H + 1e-9), 0, 1)
        D_DTW_norm = np.clip((D_DTW - min_DTW) / (max_DTW - min_DTW + 1e-9), 0, 1)
        D_Cos_norm = np.clip((D_Cos - min_Cos) / (max_Cos - min_Cos + 1e-9), 0, 1)
        params = min_max_params
    return w[0] * D_H_norm + w[1] * D_DTW_norm + w[2] * D_Cos_norm, params

def learn_fusion_weights_grid(D_H_train, D_DTW_train, D_Cos_train, y_train, k=3, cv=3):
    weight_grid = [
        [0.1, 0.8, 0.1], [0.2, 0.6, 0.2], [0.3, 0.4, 0.3],
        [0.33, 0.34, 0.33], [0.1, 0.1, 0.8], [0.8, 0.1, 0.1],
        [0.4, 0.4, 0.2], [0.2, 0.4, 0.4], [0.4, 0.2, 0.4],
    ]
    y = np.asarray(y_train)
    min_class_size = np.min(np.unique(y, return_counts=True)[1]) if len(y) > 0 else 0
    actual_cv = min(cv, min_class_size)
    if actual_cv < 2:
        skf = KFold(n_splits=2, shuffle=True, random_state=42)
    else:
        skf = StratifiedKFold(n_splits=actual_cv, shuffle=True, random_state=42)
        
    best_weights, best_acc = [0.3, 0.4, 0.3], -1.0
    for w in weight_grid:
        D_fused, _ = fuse_distances(D_H_train, D_DTW_train, D_Cos_train, weights=w)
        accs = []
        for train_idx, val_idx in skf.split(D_fused, y):
            D_val = D_fused[np.ix_(val_idx, train_idx)]
            knn = CustomDistanceKNN(n_neighbors=k)
            knn.fit(y[train_idx])
            preds = knn.predict(D_val)
            accs.append(np.mean(preds == y[val_idx]))
        mean_acc = np.mean(accs)
        if mean_acc > best_acc:
            best_acc, best_weights = mean_acc, w
    return list(best_weights)

# --- Classifiers ---
class CustomDistanceKNN:
    def __init__(self, n_neighbors=3):
        self.n_neighbors = n_neighbors
        self.X_train_labels = None
        
    def fit(self, y_train):
        self.X_train_labels = np.array(y_train)
        self.n_neighbors = min(self.n_neighbors, len(self.X_train_labels))
        return self
        
    def predict(self, D_test_train):
        n_test = D_test_train.shape[0]
        predictions = []
        for i in range(n_test):
            nearest_indices = np.argsort(D_test_train[i])[:self.n_neighbors]
            nearest_labels = self.X_train_labels[nearest_indices]
            unique_labels, counts = np.unique(nearest_labels, return_counts=True)
            predictions.append(unique_labels[np.argmax(counts)])
        return np.array(predictions)

# ---------------------------------------------------------------------------
# 3. Validation Logic
# ---------------------------------------------------------------------------
def select_segmentation_strategy(X_train):
    autocorrs = []
    for x in X_train:
        x_arr = np.asarray(x, dtype=np.float64)
        if len(x_arr) < 3:
            continue
        r = np.corrcoef(x_arr[:-1], x_arr[1:])[0, 1]
        if not np.isnan(r):
            autocorrs.append(float(r))
    if len(autocorrs) < 2:
        return 'cpd', 1.5
    var_autocorr = float(np.var(autocorrs))
    if hasattr(X_train, 'ndim') and X_train.ndim == 2:
        series_length = X_train.shape[1]
    else:
        series_length = int(np.mean([len(x) for x in X_train]))

    if var_autocorr > 0.05:
        return 'cpd', 1.5
    else:
        return 'fixed', max(10, series_length // 10)

def demonstrate_segmentation(x, method, param):
    """Prints segment boundaries and granule lengths to prove variable size (Q1.1/Q2.1)."""
    boundaries = segment_time_series(x, method, param)
    lengths = [boundaries[i+1] - boundaries[i] for i in range(len(boundaries)-1)]
    print(f"\n--- [Proof 1] Variable-Length CPD Segmentation Demonstration ---")
    print(f"Signal Length N: {len(x)}")
    print(f"Segmentation Method: {method.upper()} (param={param})")
    print(f"Detected Boundary Indices: {boundaries}")
    print(f"Number of Granules Created: {len(lengths)}")
    print(f"Granule Lengths (variable size proof): {lengths}")


def run_3d_vs_10d_comparison(train_granules, test_granules, y_train, y_test, weights, params):
    """Evaluates classifier using only 3D standard LFIG features versus 10D (Q3.1/Q3.2)."""
    train_granules_3d = [g[:, 0:3] for g in train_granules]
    test_granules_3d = [g[:, 0:3] for g in test_granules]
    
    D_H_tr_3d, D_DTW_tr_3d, D_Cos_tr_3d = compute_pairwise_distances(train_granules_3d)
    D_H_te_3d, D_DTW_te_3d, D_Cos_te_3d = compute_pairwise_distances(test_granules_3d, train_granules_3d)
    
    D_Fused_tr_3d, params_3d = fuse_distances(D_H_tr_3d, D_DTW_tr_3d, D_Cos_tr_3d, weights=weights)
    D_Fused_te_3d, _ = fuse_distances(D_H_te_3d, D_DTW_te_3d, D_Cos_te_3d, weights=weights, min_max_params=params_3d)
    
    knn = CustomDistanceKNN(n_neighbors=3).fit(y_train)
    preds_3d = knn.predict(D_Fused_te_3d)
    return accuracy_score(y_test, preds_3d)


def run_lofo_ablation_demo(train_granules, test_granules, y_train, y_test, weights, params, full_acc):
    """Performs Leave-One-Feature-Out ablation on the 10 features to show impact (Q3.2)."""
    feature_names = [
        'Lower Bound', 'Upper Bound', 'Trend Slope', 'Shannon Entropy',
        'Variance', 'Volatility', 'Curvature', 'Intercept', 'Energy', 'Skewness'
    ]
    rows = []
    for idx, feat_name in enumerate(feature_names):
        # Copy and zero out the feature at index 'idx'
        tr_abl = []
        for g in train_granules:
            g_copy = g.copy()
            g_copy[:, idx] = 0.0
            tr_abl.append(g_copy)
            
        te_abl = []
        for g in test_granules:
            g_copy = g.copy()
            g_copy[:, idx] = 0.0
            te_abl.append(g_copy)
            
        dh_tr, ddtw_tr, dcos_tr = compute_pairwise_distances(tr_abl)
        dh_te, ddtw_te, dcos_te = compute_pairwise_distances(te_abl, tr_abl)
        
        df_tr, p_abl = fuse_distances(dh_tr, ddtw_tr, dcos_tr, weights=weights)
        df_te, _ = fuse_distances(dh_te, ddtw_te, dcos_te, weights=weights, min_max_params=p_abl)
        
        knn = CustomDistanceKNN(n_neighbors=3).fit(y_train)
        preds_abl = knn.predict(df_te)
        acc_abl = accuracy_score(y_test, preds_abl)
        delta = full_acc - acc_abl
        rows.append({'Feature': feat_name, 'Ablated_Acc': acc_abl, 'Delta': delta})
        
    print("\n--- [Proof 3] Leave-One-Feature-Out (LOFO) Impact Table ---")
    print(f"{'Feature Name':20s} | {'Ablated Acc':12s} | {'Impact (Delta)':15s}")
    print("-" * 55)
    for r in rows:
        print(f"{r['Feature']:20s} | {r['Ablated_Acc']:.4f}      | {r['Delta']:+.4f}")


def _compute_distance_cache(X_tr, X_te, method, param, z):
    """Computes and caches distance matrices for a specific z value."""
    gran_tr = [extract_granular_sequence(x, segment_time_series(x, method, param), z=z) for x in X_tr]
    gran_te = [extract_granular_sequence(x, segment_time_series(x, method, param), z=z) for x in X_te]
    D_H_tr, D_DTW_tr, D_Cos_tr = compute_pairwise_distances(gran_tr)
    D_H_te, D_DTW_te, D_Cos_te = compute_pairwise_distances(gran_te, gran_tr)
    return D_H_tr, D_DTW_tr, D_Cos_tr, D_H_te, D_DTW_te, D_Cos_te


def _classify_from_cache(dist_cache, y_tr, y_te, weights, k, clf_type):
    """Fuses distances and classifies using the cached distance matrices."""
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
        raise ValueError(f"Unknown classifier type: {clf_type}")
    return preds


def _accuracy_from_cache(dist_cache, y_tr, y_te, weights, k, clf_type):
    """Calculates accuracy on cached distance arrays."""
    preds = _classify_from_cache(dist_cache, y_tr, y_te, weights, k, clf_type)
    return accuracy_score(y_te, preds)


PARAM_GRID = {
    'z': [1.0, 1.96],
    'k': [1, 3],
    'weights': [[0.1, 0.8, 0.1], [0.3, 0.4, 0.3], [0.2, 0.6, 0.2]],
    'clf_type': ['KNN', 'Kernel SVM']
}

def run_inner_cv(X_tr_fold, y_tr_fold, method, default_param, n_inner=3):
    """Runs inner CV stratified loop to optimize hyperparameters using caches."""
    min_class_size = np.min(np.unique(y_tr_fold, return_counts=True)[1]) if len(y_tr_fold) > 0 else 0
    actual_inner = min(n_inner, min_class_size)
    
    if actual_inner < 2:
        skf = KFold(n_splits=2, shuffle=True, random_state=42)
    else:
        skf = StratifiedKFold(n_splits=actual_inner, shuffle=True, random_state=42)
        
    splits = list(skf.split(X_tr_fold, y_tr_fold))
    
    z_values = PARAM_GRID['z']
    other_combos = list(product(PARAM_GRID['k'], PARAM_GRID['weights'], PARAM_GRID['clf_type']))
    
    best_acc, best_params = -1.0, None
    for z in z_values:
        caches = []
        for inner_tr_idx, inner_val_idx in splits:
            cache = _compute_distance_cache(
                X_tr_fold[inner_tr_idx], X_tr_fold[inner_val_idx],
                method, default_param, z
            )
            caches.append((cache, y_tr_fold[inner_tr_idx], y_tr_fold[inner_val_idx]))
            
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
                
    if best_params is None:
        best_params = {
            'z': 1.96,
            'k': 1,
            'weights': [0.3, 0.4, 0.3],
            'clf_type': 'KNN'
        }
    return best_params


def run_nested_cv_standalone(X, y, n_outer=5, n_inner=3):
    """Runs a complete 5-fold outer, 3-fold inner nested CV on Colab directly."""
    print("\n--- Running 5-Fold Leakage-Free Nested Cross-Validation ---")
    
    lengths = [len(x) for x in X]
    if len(set(lengths)) == 1:
        X = np.array([np.asarray(x, dtype=np.float64) for x in X], dtype=np.float64)
    else:
        X_arr = np.empty(len(X), dtype=object)
        for i, x in enumerate(X):
            X_arr[i] = np.asarray(x, dtype=np.float64)
        X = X_arr
        
    y = np.asarray(y)
        
    outer = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=42)
    metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
    
    for fold_i, (tr_idx, te_idx) in enumerate(outer.split(X, y)):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        
        # 1. Strategy chosen from training split only
        method, default_param = select_segmentation_strategy(X_tr)
        
        # 2. Hyperparameters chosen by inner CV on training split only
        best_params = run_inner_cv(X_tr, y_tr, method, default_param, n_inner=n_inner)
        
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
        print(f"  Fold {fold_i+1}/{n_outer} Accuracy: {acc:.4f}  F1: {f1:.4f}  (Params: {best_params})")
        
    mean_acc = np.mean(metrics['accuracy'])
    std_acc = np.std(metrics['accuracy'])
    mean_f1 = np.mean(metrics['f1'])
    print(f"--> Nested CV Results: Mean Acc = {mean_acc:.4f} ± {std_acc:.4f} | F1 = {mean_f1:.4f} ± {std_f1:.4f}")
    return {'mean': mean_acc, 'std': std_acc}


def run_evaluation_pipeline(X_train, y_train, X_test, y_test):
    print("Selecting segmentation strategy automatically...")
    method, default_param = select_segmentation_strategy(X_train)
    print(f"Selected strategy: {method} (param={default_param})")

    # Granularize
    print("Extracting multi-feature granules...")
    train_granules = [extract_granular_sequence(x, segment_time_series(x, method, default_param)) for x in X_train]
    test_granules = [extract_granular_sequence(x, segment_time_series(x, method, default_param)) for x in X_test]

    # Pairwise distances
    print("Computing distance matrices...")
    D_H_tr, D_DTW_tr, D_Cos_tr = compute_pairwise_distances(train_granules)
    D_H_te, D_DTW_te, D_Cos_te = compute_pairwise_distances(test_granules, train_granules)

    # Learn optimal weights
    print("Learning optimal distance fusion weights...")
    best_weights = learn_fusion_weights_grid(D_H_tr, D_DTW_tr, D_Cos_tr, y_train, k=3, cv=3)
    print(f"Optimal weights: {best_weights}")

    # Fuse
    D_Fused_tr, params = fuse_distances(D_H_tr, D_DTW_tr, D_Cos_tr, weights=best_weights)
    D_Fused_te, _ = fuse_distances(D_H_te, D_DTW_te, D_Cos_te, weights=best_weights, min_max_params=params)

    # Classify
    print("Classifying test sequences...")
    knn = CustomDistanceKNN(n_neighbors=3)
    knn.fit(y_train)
    preds = knn.predict(D_Fused_te)

    # Metrics
    acc = accuracy_score(y_test, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, preds, average='macro', zero_division=0)
    print(f"\nResults:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  Macro F1:  {f1:.4f}")

    # 1. Variable-Length CPD Segmentation Demonstration
    demonstrate_segmentation(X_train[0], method, default_param)

    # 2. 3D vs 10D Comparison
    acc_3d = run_3d_vs_10d_comparison(train_granules, test_granules, y_train, y_test, best_weights, params)
    print(f"\n--- [Proof 2] Feature Comparison (3D Standard vs 10D Proposed) ---")
    print(f"{'Feature Set':30s} | {'Test Accuracy':15s} | {'Improvement Delta':18s}")
    print("-" * 70)
    print(f"{'Standard 3D LFIG':30s} | {acc_3d:.4f}          | -")
    print(f"{'Proposed Multi-Feature 10D LFIG':30s} | {acc:.4f}          | {acc - acc_3d:+.4f}")

    # 3. LOFO Ablation
    run_lofo_ablation_demo(train_granules, test_granules, y_train, y_test, best_weights, params, acc)

    # 4. Comparative Baselines (DTW-1NN, ROCKET, MiniROCKET, HIVE-COTE 2.0)
    run_baseline_comparison_demo(X_train, y_train, X_test, y_test, acc, "Current Dataset")
    return acc, acc_3d


def run_baseline_comparison_demo(X_train, y_train, X_test, y_test, our_acc, dataset_name="Current Dataset"):
    print(f"\n--- [Proof 4] Comparative Baselines Benchmark ({dataset_name}) ---")
    print(f"{'Classifier / Model':34s} | {'Test Accuracy':15s} | {'Delta vs Proposed':20s}")
    print("-" * 75)
    print(f"{'Our Proposed 10D Adaptive LFIG':34s} | {our_acc:.4f}          | Baseline (0.0000)")

    def _reshape_3d(X):
        if isinstance(X, list) or (isinstance(X, np.ndarray) and X.dtype == object):
            return [x.reshape(1, -1) if hasattr(x, 'ndim') and x.ndim == 1 else x for x in X]
        if hasattr(X, 'ndim') and X.ndim == 2:
            return X.reshape(X.shape[0], 1, X.shape[1])
        return X

    X_tr_3d = _reshape_3d(X_train)
    X_te_3d = _reshape_3d(X_test)

    # 1. DTW-1NN
    try:
        from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier
        clf = KNeighborsTimeSeriesClassifier(n_neighbors=1, distance='dtw')
        clf.fit(X_tr_3d, y_train)
        preds = clf.predict(X_te_3d)
        acc_dtw = accuracy_score(y_test, preds)
        print(f"{'DTW-1NN (aeon)':34s} | {acc_dtw:.4f}          | {acc_dtw - our_acc:+.4f}")
    except Exception as e:
        print(f"{'DTW-1NN (aeon)':34s} | N/A              | -")

    # 2. ROCKET
    try:
        from aeon.classification.convolution_based import RocketClassifier
        clf = RocketClassifier(random_state=42)
        clf.fit(X_tr_3d, y_train)
        preds = clf.predict(X_te_3d)
        acc_rocket = accuracy_score(y_test, preds)
        print(f"{'ROCKET (aeon)':34s} | {acc_rocket:.4f}          | {acc_rocket - our_acc:+.4f}")
    except Exception as e:
        print(f"{'ROCKET (aeon)':34s} | N/A              | -")

    # 3. MiniROCKET
    try:
        from aeon.classification.convolution_based import MiniRocketClassifier
        clf = MiniRocketClassifier(random_state=42)
        clf.fit(X_tr_3d, y_train)
        preds = clf.predict(X_te_3d)
        acc_minirocket = accuracy_score(y_test, preds)
        print(f"{'MiniROCKET (aeon)':34s} | {acc_minirocket:.4f}          | {acc_minirocket - our_acc:+.4f}")
    except Exception as e:
        print(f"{'MiniROCKET (aeon)':34s} | N/A              | -")

    # 4. HIVE-COTE 2.0 Literature
    hc2_dict = {
        'GunPoint': 1.0000, 'Coffee': 1.0000, 'ArrowHead': 0.8710,
        'ECG200': 0.9000, 'Chinatown': 0.9830, 'ItalyPowerDemand': 0.9700,
        'TwoLeadECG': 1.0000, 'ECGFiveDays': 1.0000
    }
    if dataset_name in hc2_dict:
        hc2_acc = hc2_dict[dataset_name]
        print(f"{'HIVE-COTE 2.0 (Literature†)':34s} | {hc2_acc:.4f}          | {hc2_acc - our_acc:+.4f}")


# ---------------------------------------------------------------------------
# 4. Main Execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from aeon.datasets import load_classification
    
    datasets = [
        'GunPoint', 'Coffee', 'ArrowHead', 'ECG200', 'Chinatown', 
        'ItalyPowerDemand', 'SonyAIBORobotSurface1', 'TwoLeadECG', 
        'ECGFiveDays', 'MoteStrain', 'Beef', 'OliveOil', 'Meat', 
        'BeetleFly', 'BirdChicken', 'FaceFour', 'SyntheticControl', 
        'CBF', 'TwoPatterns', 'Wafer', 'FordA', 'Yoga', 'SwedishLeaf'
    ]
    
    master_summary = []
    pbar = tqdm(datasets, desc="UCR Catalog Benchmarks", unit="dataset")
    for name in pbar:
        pbar.set_description(f"Processing: {name}")
        print(f"\n{'='*60}")
        print(f"Dataset: {name}")
        print(f"{'='*60}")
        try:
            X_train, y_train = load_classification(name, split="train")
            X_test, y_test = load_classification(name, split="test")

            # Squeeze channel dimension for univariate time series robustly
            if isinstance(X_train, np.ndarray) and X_train.ndim == 3:
                X_train = X_train.squeeze(axis=1)
            else:
                X_train = [x.squeeze(axis=0) if hasattr(x, 'ndim') and x.ndim == 2 else x for x in X_train]
                
            if isinstance(X_test, np.ndarray) and X_test.ndim == 3:
                X_test = X_test.squeeze(axis=1)
            else:
                X_test = [x.squeeze(axis=0) if hasattr(x, 'ndim') and x.ndim == 2 else x for x in X_test]

            n_train = len(X_train)
            n_test = len(X_test)
            if hasattr(X_train, 'shape') and len(X_train.shape) == 2:
                series_len = X_train.shape[1]
            else:
                series_len = int(np.mean([len(x) for x in X_train]))

            print(f"Train samples: {n_train}, Test samples: {n_test}, Length (mean): {series_len}")
            
            t0 = time.time()
            # 1. Run single-split evaluation with diagnostic proofs (CPD segment lengths, 3D vs 10D table, LOFO ablation, Baselines)
            acc, acc_3d = run_evaluation_pipeline(X_train, y_train, X_test, y_test)
            
            # 2. Run 5-fold leakage-free nested cross-validation
            combined_X = list(X_train) + list(X_test)
            combined_y = np.concatenate([y_train, y_test], axis=0)
            ncv_results = run_nested_cv_standalone(combined_X, combined_y, n_outer=5, n_inner=3)
            
            master_summary.append({
                'Dataset': name,
                '3D_Acc': acc_3d,
                '10D_Acc': acc,
                'NestedCV': f"{ncv_results['mean']:.4f}±{ncv_results['std']:.4f}"
            })
            print(f"Finished {name} in {time.time() - t0:.2f} seconds.")
        except Exception as e:
            print(f"Error processing {name}: {e}")

    if master_summary:
        print(f"\n{'='*80}")
        print("MASTER SUMMARY COMPARATIVE BENCHMARK TABLE")
        print(f"{'='*80}")
        print(f"{'Dataset':22s} | {'3D LFIG Acc':12s} | {'10D LFIG Single Acc':20s} | {'Nested CV Acc (Mean±Std)':25s}")
        print("-" * 80)
        for row in master_summary:
            print(f"{row['Dataset']:22s} | {row['3D_Acc']:.4f}       | {row['10D_Acc']:.4f}               | {row['NestedCV']}")
        print(f"{'='*80}\n")
