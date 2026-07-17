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
        packages = ["aeon>=0.7.0", "ruptures>=1.1.5", "fastdtw>=0.3.4", "scikit-learn", "numpy", "scipy", "pandas"]
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + packages)
        print("All packages successfully installed.")

install_dependencies()

# Import packages
import numpy as np
import scipy.stats
import ruptures as rpt
from fastdtw import fastdtw
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

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
        signal = X_series.reshape(-1, 1)
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
            D_H[i, j] = D_H[j, i] = dh if symmetric else dh
            D_DTW[i, j] = D_DTW[j, i] = ddtw if symmetric else ddtw
            D_Cos[i, j] = D_Cos[j, i] = dcos if symmetric else dcos
            if not symmetric:
                D_H[i, j], D_DTW[i, j], D_Cos[i, j] = dh, ddtw, dcos
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
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
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
        if len(x) < 3:
            continue
        r = np.corrcoef(x[:-1], x[1:])[0, 1]
        if not np.isnan(r):
            autocorrs.append(r)
    if len(autocorrs) < 2:
        return 'cpd', 1.5
    var_autocorr = np.var(autocorrs)
    if var_autocorr > 0.05:
        return 'cpd', 1.5
    else:
        return 'fixed', max(10, X_train.shape[1] // 10)

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

# ---------------------------------------------------------------------------
# 4. Main Execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from aeon.datasets import load_classification
    
    dataset_name = "GunPoint"
    print(f"Loading '{dataset_name}' dataset online via aeon...")
    X_train, y_train = load_classification(dataset_name, split="train")
    X_test, y_test = load_classification(dataset_name, split="test")

    # Squeeze channel dimension for univariate time series
    X_train = X_train.squeeze(axis=1)
    X_test = X_test.squeeze(axis=1)

    print(f"Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}, Length: {X_train.shape[1]}")
    
    t0 = time.time()
    run_evaluation_pipeline(X_train, y_train, X_test, y_test)
    print(f"\nExecution finished in {time.time() - t0:.2f} seconds.")
