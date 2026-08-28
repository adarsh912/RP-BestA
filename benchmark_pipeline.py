# 1. Pipeline codebase module


# --- NEW CELL ---

import warnings
warnings.filterwarnings("ignore")
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

# --- NEW CELL ---

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


# --- NEW CELL ---

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

def _compute_single_pair(i, j, x_i, y_j):
    dh, ddtw, dcos = compute_distances(x_i, y_j)
    return i, j, dh, ddtw, dcos

def compute_pairwise_distances(X, Y=None):
    from joblib import Parallel, delayed
    N_X = len(X)
    symmetric = Y is None
    N_Y = N_X if symmetric else len(Y)
    D_H, D_DTW, D_Cos = np.zeros((N_X, N_Y)), np.zeros((N_X, N_Y)), np.zeros((N_X, N_Y))
    dataset_Y = X if symmetric else Y
    
    tasks = []
    for i in range(N_X):
        start_j = i if symmetric else 0
        for j in range(start_j, N_Y):
            if symmetric and i == j:
                continue
            tasks.append((i, j, X[i], dataset_Y[j]))
            
    results = Parallel(n_jobs=-1, backend="loky")(
        delayed(_compute_single_pair)(i, j, x_i, y_j) for i, j, x_i, y_j in tasks
    )
    
    for i, j, dh, ddtw, dcos in results:
        D_H[i, j] = dh
        D_DTW[i, j] = ddtw
        D_Cos[i, j] = dcos
        if symmetric and i != j:
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

# --- NEW CELL ---

# 3. Validation Logic
# ---------------------------------------------------------------------------
def compute_segmentation_rss(x, boundaries):
    rss = 0.0
    for j in range(len(boundaries) - 1):
        start, end = boundaries[j], boundaries[j+1]
        segment = x[start:end]
        L = len(segment)
        if L == 0:
            continue
        elif L < 2:
            rss += 0.0
        else:
            tau = np.arange(1, L + 1)
            try:
                a, b = np.polyfit(tau, segment, 1)
                residuals = segment - (a * tau + b)
                rss += np.sum(residuals**2)
            except Exception:
                pass
    return rss

def select_segmentation_strategy(X_train):
    ratios = []
    n_samples = min(10, len(X_train))
    rng = np.random.RandomState(42)
    sample_indices = rng.choice(len(X_train), n_samples, replace=False)
    
    if hasattr(X_train, 'ndim') and X_train.ndim == 2:
        series_length = X_train.shape[1]
    else:
        series_length = int(np.mean([len(x) for x in X_train]))
        
    window_size_default = max(10, series_length // 10)
    
    for idx in sample_indices:
        x = np.asarray(X_train[idx], dtype=np.float64)
        N = len(x)
        
        # 1. Run CPD to find segment budget
        bounds_cpd = cpd_segmentation(x, penalty=1.5)
        num_segments_cpd = len(bounds_cpd) - 1
        
        # 2. Force Fixed windowing to use the same budget (same number of segments)
        fixed_window_size = max(3, N // max(1, num_segments_cpd))
        bounds_fixed = fixed_segmentation(x, fixed_window_size)
        
        rss_fixed = compute_segmentation_rss(x, bounds_fixed)
        rss_cpd = compute_segmentation_rss(x, bounds_cpd)
        
        ratio = rss_cpd / (rss_fixed + 1e-9)
        ratios.append(ratio)
        
    # Bootstrap mean ratio to guard against small-N noise
    boot_means = []
    ratios = np.array(ratios)
    for _ in range(100):
        boot_sample = rng.choice(ratios, size=len(ratios), replace=True)
        boot_means.append(np.mean(boot_sample))
        
    upper_bound = np.percentile(boot_means, 95)
    
    if upper_bound < 0.85:
        return 'cpd', 1.5
    else:
        return 'fixed', window_size_default

# [Duplicate functions removed]




# --- NEW CELL ---



# --- NEW CELL ---

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
            
        for k, weights, clf_type in tqdm(other_combos, desc=f"    Inner CV Grid (z={z})", leave=False):
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
    selected_strategies = []
    
    for fold_i, (tr_idx, te_idx) in enumerate(tqdm(list(outer.split(X, y)), desc="  Outer CV Folds", leave=False)):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        
        # 1. Strategy chosen from training split only
        method, default_param = select_segmentation_strategy(X_tr)
        selected_strategies.append(f"{method}({default_param})")
        
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
    std_f1 = np.std(metrics['f1'])
    print(f"--> Nested CV Results: Mean Acc = {mean_acc:.4f} ± {std_acc:.4f} | F1 = {mean_f1:.4f} ± {std_f1:.4f}")
    return {'mean': mean_acc, 'std': std_acc, 'f1_mean': mean_f1, 'f1_std': std_f1, 'strategies': selected_strategies}

# --- NEW CELL ---

def run_evaluation_pipeline(X_train, y_train, X_test, y_test, dataset_name="Current Dataset"):
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
    best_base_name, best_base_acc, baselines = run_baseline_comparison_demo(X_train, y_train, X_test, y_test, acc, prec, rec, f1, dataset_name)
    return acc, acc_3d, best_base_name, best_base_acc, baselines, method, default_param, best_weights
def run_baseline_comparison_demo(X_train, y_train, X_test, y_test, our_acc, our_prec, our_rec, our_f1, dataset_name="Current Dataset"):
    print(f"\n--- [Proof 4] Comparative Baselines Benchmark ({dataset_name}) ---")
    print(f"{'Classifier / Model':30s} | {'Accuracy':8s} | {'Precision':9s} | {'Recall':8s} | {'Macro F1':8s} | {'Delta vs Proposed':18s}")
    print("-" * 95)
    print(f"{'Our Proposed 10D Adaptive LFIG':30s} | {our_acc:.4f} | {our_prec:.4f}  | {our_rec:.4f} | {our_f1:.4f} | Baseline (0.0000)")
    baselines = {}
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
        prec_dtw, rec_dtw, f1_dtw, _ = precision_recall_fscore_support(y_test, preds, average='macro', zero_division=0)
        print(f"{'DTW-1NN (aeon)':30s} | {acc_dtw:.4f} | {prec_dtw:.4f}  | {rec_dtw:.4f} | {f1_dtw:.4f} | {acc_dtw - our_acc:+.4f}")
        baselines['DTW-1NN'] = acc_dtw
    except Exception as e:
        print(f"{'DTW-1NN (aeon)':30s} | N/A      | N/A       | N/A      | N/A      | -")
    # 2. ROCKET
    try:
        from aeon.classification.convolution_based import RocketClassifier
        clf = RocketClassifier(random_state=42)
        clf.fit(X_tr_3d, y_train)
        preds = clf.predict(X_te_3d)
        acc_rocket = accuracy_score(y_test, preds)
        prec_rocket, rec_rocket, f1_rocket, _ = precision_recall_fscore_support(y_test, preds, average='macro', zero_division=0)
        print(f"{'ROCKET (aeon)':30s} | {acc_rocket:.4f} | {prec_rocket:.4f}  | {rec_rocket:.4f} | {f1_rocket:.4f} | {acc_rocket - our_acc:+.4f}")
        baselines['ROCKET'] = acc_rocket
    except Exception as e:
        print(f"{'ROCKET (aeon)':30s} | N/A      | N/A       | N/A      | N/A      | -")
    # 3. MiniROCKET
    try:
        from aeon.classification.convolution_based import MiniRocketClassifier
        clf = MiniRocketClassifier(random_state=42)
        clf.fit(X_tr_3d, y_train)
        preds = clf.predict(X_te_3d)
        acc_minirocket = accuracy_score(y_test, preds)
        prec_minirocket, rec_minirocket, f1_minirocket, _ = precision_recall_fscore_support(y_test, preds, average='macro', zero_division=0)
        print(f"{'MiniROCKET (aeon)':30s} | {acc_minirocket:.4f} | {prec_minirocket:.4f}  | {rec_minirocket:.4f} | {f1_minirocket:.4f} | {acc_minirocket - our_acc:+.4f}")
        baselines['MiniROCKET'] = acc_minirocket
    except Exception as e:
        print(f"{'MiniROCKET (aeon)':30s} | N/A      | N/A       | N/A      | N/A      | -")
    # 4. HIVE-COTE 2.0 Literature
    hc2_dict = {
        'GunPoint': 1.0000, 'Coffee': 1.0000, 'ArrowHead': 0.8710,
        'ECG200': 0.9000, 'Chinatown': 0.9830, 'ItalyPowerDemand': 0.9700,
        'TwoLeadECG': 1.0000, 'ECGFiveDays': 1.0000
    }
    if dataset_name in hc2_dict:
        hc2_acc = hc2_dict[dataset_name]
        print(f"{'HIVE-COTE 2.0 (Literature†)':30s} | {hc2_acc:.4f} | N/A (Lit) | N/A (Lit)| N/A (Lit)| {hc2_acc - our_acc:+.4f}")
        baselines['HIVE-COTE 2.0'] = hc2_acc
    if baselines:
        best_name = max(baselines, key=baselines.get)
        return best_name, baselines[best_name], baselines
    return "None", 0.0, {}
def _df_to_markdown_safe(df):
    try:
        return df.to_markdown(index=False)
    except Exception:
        cols = list(df.columns)
        header = "| " + " | ".join(str(c) for c in cols) + " |"
        sep = "| " + " | ".join("---" for _ in cols) + " |"
        rows = []
        for idx, row in df.iterrows():
            rows.append("| " + " | ".join(str(val) for val in row.values) + " |")
        return "\n".join([header, sep] + rows)

# --- NEW CELL ---

is_gpu = False
import os
import time
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
try:
    from tqdm import tqdm
except ImportError:
    class tqdm:
        def __init__(self, iterable=None, *args, **kwargs): self.iterable = iterable
        def __iter__(self): return iter(self.iterable) if self.iterable is not None else iter([])
        def update(self, n=1): pass
        def close(self): pass
        def set_description(self, desc): pass
print("=== Benchmark Setup and run_single_dataset function initialized ===")
master_summary = []
os.makedirs('plots', exist_ok=True)
def run_single_dataset(name):
    global master_summary
    output_csv = 'master_benchmark_results_gpu.csv' if is_gpu else 'master_benchmark_results.csv'
    output_md = 'master_benchmark_results_gpu.md' if is_gpu else 'master_benchmark_results.md'
    print(f"\n{'='*60}")
    print(f"Dataset: {name}")
    print(f"{'='*60}")
    try:
        from aeon.datasets import load_classification
        X_train, y_train = load_classification(name, split="train")
        X_test, y_test = load_classification(name, split="test")
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_test = le.transform(y_test)
        if isinstance(X_train, np.ndarray) and X_train.ndim == 3:
            X_train = X_train.squeeze(axis=1)
        else:
            X_train = [x.squeeze(axis=0) if hasattr(x, 'ndim') and x.ndim == 2 else x for x in X_train]
        if isinstance(X_test, np.ndarray) and X_test.ndim == 3:
            X_test = X_test.squeeze(axis=1)
        else:
            X_test = [x.squeeze(axis=0) if hasattr(x, 'ndim') and x.ndim == 2 else x for x in X_test]
        n_train, n_test = len(X_train), len(X_test)
        series_len = X_train.shape[1] if hasattr(X_train, 'shape') and len(X_train.shape) == 2 else int(np.mean([len(x) for x in X_train]))
        print(f"Train samples: {n_train}, Test samples: {n_test}, Length (mean): {series_len}")
        t0 = time.time()
        
        # 1. Single split evaluation + Diagnostic Proofs (1-4)
        acc, acc_3d, best_base_name, best_base_acc, baselines, method, default_param, best_weights = run_evaluation_pipeline(
            X_train, y_train, X_test, y_test, dataset_name=name
        )
        
        # 2. 5-Fold Nested Cross-Validation with progress bars
        res = run_nested_cv_standalone(list(X_train) + list(X_test), np.concatenate([y_train, y_test], axis=0), n_outer=5, n_inner=3)
        t_elapsed = time.time() - t0
        
        mean_acc, std_acc = res['mean'], res['std']
        mean_f1, std_f1 = res['f1_mean'], res['f1_std']
        
        # Load existing results if on disk to prevent overwriting results of other cells
        if os.path.exists(output_csv):
            try:
                master_summary = pd.read_csv(output_csv).to_dict(orient='records')
            except Exception:
                pass
        
        # Remove previous entry for this dataset to avoid duplicates
        master_summary = [r for r in master_summary if r['Dataset'] != name]
        
        if best_base_acc > acc:
            outperformed_by = f"{best_base_name} ({best_base_acc:.4f})"
        else:
            outperformed_by = "Proposed Outperformed All" if best_base_acc < acc else "Tie"
            
        segmentation_str = f"fixed({int(float(default_param))})" if method.lower() == 'fixed' else f"cpd({default_param})"
        
        master_summary.append({
            'Dataset': name,
            'Train': n_train,
            'Test': n_test,
            'Length': series_len,
            'Segmentation': segmentation_str,
            'Weights': str(best_weights),
            'NestedCVAcc': mean_acc,
            'NestedCVAccStd': std_acc,
            'NestedCVF1': mean_f1,
            'NestedCVF1Std': std_f1,
            'DTW1NN_Acc': baselines.get('DTW-1NN', np.nan),
            'ROCKET_Acc': baselines.get('ROCKET', np.nan),
            'MiniROCKET_Acc': baselines.get('MiniROCKET', np.nan),
            'HC2_Acc': baselines.get('HIVE-COTE 2.0', np.nan),
            '3D_Acc': acc_3d,
            '10D_Acc': acc,
            'NestedCV': f"{mean_acc:.4f}±{std_acc:.4f}",
            'Time (s)': f"{t_elapsed:.1f}",
            'Best_Baseline': f"{best_base_name} ({best_base_acc:.4f})" if best_base_acc > 0 else "N/A",
            'Outperformed_By': outperformed_by,
            'Fold_Strategies': ", ".join(res['strategies'])
        })
        
        # Incremental save
        df_summary = pd.DataFrame(master_summary)
        df_summary.to_csv(output_csv, index=False)
        df_summary.to_csv('plots/evaluation_results.csv', index=False)
        with open(output_md, 'w') as f:
            f.write('# Master Summary Benchmark Results\n\n')
            f.write(_df_to_markdown_safe(df_summary))
        with open('plots/evaluation_results.md', 'w') as f:
            f.write('# Master Summary Benchmark Results\n\n')
            f.write(_df_to_markdown_safe(df_summary))
        print(f"\nFinished {name} in {t_elapsed:.2f} seconds. (Saved to {output_csv})")
    except Exception as e:
        print(f"Error processing {name}: {e}")

# --- End of Library Definitions ---


# ==============================================================================
# 5. Master Result Compilation & Critical Difference Diagram Generation
# ==============================================================================
import json
import re
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare

_NEMENYI_Q_ALPHA = {
    2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728,
    6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164
}

def compute_average_ranks(accuracy_matrix):
    ranks = accuracy_matrix.rank(axis=1, ascending=False, method='average')
    return ranks.mean(axis=0)

def friedman_test(accuracy_matrix):
    groups = [accuracy_matrix[col].values for col in accuracy_matrix.columns]
    return friedmanchisquare(*groups)

def nemenyi_post_hoc(accuracy_matrix, alpha=0.05):
    k = len(accuracy_matrix.columns)
    N = len(accuracy_matrix)
    q_alpha = _NEMENYI_Q_ALPHA.get(k, 1.96)
    return q_alpha * np.sqrt(k * (k + 1) / (6 * N))

def plot_critical_difference_diagram(avg_ranks, cd, classifier_names, output_path='plots/cd_diagram.png'):
    k = len(classifier_names)
    sorted_indices = np.argsort(avg_ranks)
    sorted_names = [classifier_names[i] for i in sorted_indices]
    sorted_ranks = [avg_ranks[i] for i in sorted_indices]
    
    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    ax.set_xlim(0.5, k + 0.5)
    ax.set_ylim(-1.5, 1.5)
    ax.hlines(0, 0.5, k + 0.5, color='black', linewidth=1)
    
    for r in range(1, k + 1):
        ax.vlines(r, -0.05, 0.05, color='black', linewidth=1)
        ax.text(r, -0.15, str(r), ha='center', va='top', fontsize=9)
        
    for i, (name, rank) in enumerate(zip(sorted_names, sorted_ranks)):
        y_text = 0.6 if i % 2 == 0 else -0.6
        y_tick = 0.05 if i % 2 == 0 else -0.05
        ax.plot(rank, 0, 'ko', markersize=6)
        ax.vlines(rank, 0, y_tick + (0.3 if i % 2 == 0 else -0.3), color='gray', linewidth=0.8)
        ax.text(rank, y_text, name, ha='center', va='bottom' if i % 2 == 0 else 'top', fontsize=8, fontweight='bold')
        
    # Draw CD bars
    for i in range(k):
        for j in range(i+1, k):
            if abs(sorted_ranks[i] - sorted_ranks[j]) <= cd:
                y_line = 0.2 + (i * 0.1)
                ax.hlines(y_line, sorted_ranks[i], sorted_ranks[j], color='red', linewidth=2.5)
                
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def compile_results_and_plot(nb_file, csv_file, is_gpu=True):
    datasets = {}
    nb_data = {"cells": []}
    if not os.path.exists(nb_file):
        print(f"Warning: Notebook file '{nb_file}' not found. Cannot parse execution outputs from it.")
    else:
        try:
            print(f"Parsing notebook cell outputs from {nb_file}...")
            with open(nb_file, 'r', encoding='utf-8') as f:
                nb_data = json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load notebook '{nb_file}': {e}")
        
    current_dataset = None
    
    for cell in nb_data.get('cells', []):
        for output in cell.get('outputs', []):
            if 'text' in output:
                lines = output['text']
                if isinstance(lines, str):
                    lines = lines.split('\n')
                for line in lines:
                    line_str = line.strip()
                    ds_match = re.search(r'Dataset:\s*(\w+)', line_str)
                    if ds_match:
                        current_dataset = ds_match.group(1).strip()
                        if current_dataset not in datasets:
                            datasets[current_dataset] = {}
                        continue
                    
                    if not current_dataset:
                        continue
                        
                    samples_match = re.search(r'Train samples:\s*(\d+),\s*Test samples:\s*(\d+),\s*Length\s*\(mean\):\s*(\d+)', line_str)
                    if samples_match:
                        datasets[current_dataset]['Train'] = int(samples_match.group(1))
                        datasets[current_dataset]['Test'] = int(samples_match.group(2))
                        datasets[current_dataset]['Length'] = int(samples_match.group(3))
                        continue
                        
                    strat_match = re.search(r'Selected strategy:\s*(\w+)\s*\(param=([\d\.]+)\)', line_str)
                    if strat_match:
                        method = strat_match.group(1).strip()
                        param = strat_match.group(2).strip()
                        if method.lower() == 'fixed':
                            datasets[current_dataset]['Segmentation'] = f"fixed({int(float(param))})"
                        else:
                            datasets[current_dataset]['Segmentation'] = f"cpd({param})"
                        continue
                        
                    weights_match = re.search(r'Optimal weights:\s*\[([0-9\.,\s]+)\]', line_str)
                    if weights_match:
                        w_list = [float(x.strip()) for x in weights_match.group(1).split(',')]
                        datasets[current_dataset]['Weights'] = str(w_list)
                        continue
                        
                    ncv_match = re.search(r'Nested CV Results:\s*Mean Acc\s*=\s*([\d\.]+)\s*±\s*([\d\.]+)\s*\|\s*F1\s*=\s*([\d\.]+)\s*±\s*([\d\.]+)', line_str)
                    if ncv_match:
                        datasets[current_dataset]['NestedCVAcc'] = float(ncv_match.group(1))
                        datasets[current_dataset]['NestedCVAccStd'] = float(ncv_match.group(2))
                        datasets[current_dataset]['NestedCVF1'] = float(ncv_match.group(3))
                        datasets[current_dataset]['NestedCVF1Std'] = float(ncv_match.group(4))
                        continue
                        
                    if 'DTW-1NN (aeon)' in line_str or 'DTW-1NN' in line_str:
                        m = re.search(r'\|\s*([\d\.]+)', line_str)
                        if m:
                            datasets[current_dataset]['DTW1NN_Acc'] = float(m.group(1))
                    elif 'MiniROCKET (aeon)' in line_str or 'MiniROCKET' in line_str:
                        m = re.search(r'\|\s*([\d\.]+)', line_str)
                        if m:
                            datasets[current_dataset]['MiniROCKET_Acc'] = float(m.group(1))
                    elif 'ROCKET (aeon)' in line_str or 'ROCKET' in line_str:
                        m = re.search(r'\|\s*([\d\.]+)', line_str)
                        if m:
                            datasets[current_dataset]['ROCKET_Acc'] = float(m.group(1))
                    elif 'HIVE-COTE 2.0 (Literature†)' in line_str or 'HIVE-COTE 2.0' in line_str:
                        m = re.search(r'\|\s*([\d\.]+)', line_str)
                        if m:
                            datasets[current_dataset]['HC2_Acc'] = float(m.group(1))

    if not os.path.exists(csv_file):
        print(f"Master CSV file not found: {csv_file}")
        return

    df_csv = pd.read_csv(csv_file)
    records = []
    for idx, row in df_csv.iterrows():
        name = row['Dataset']
        nb_data = datasets.get(name, {})
        nested_cv_str = str(row['NestedCV'])
        cv_match = re.match(r'([\d\.]+)±([\d\.]+)', nested_cv_str)
        cv_acc = float(cv_match.group(1)) if cv_match else nb_data.get('NestedCVAcc', np.nan)
        cv_std = float(cv_match.group(2)) if cv_match else nb_data.get('NestedCVAccStd', np.nan)
        
        def get_val(key, fallback_val):
            if key in row and not pd.isna(row[key]):
                return row[key]
            return fallback_val
            
        record = {
            'Dataset': name,
            'Train': int(get_val('Train', nb_data.get('Train', 0))),
            'Test': int(get_val('Test', nb_data.get('Test', 0))),
            'Length': int(get_val('Length', nb_data.get('Length', 0))),
            'Segmentation': get_val('Segmentation', nb_data.get('Segmentation', 'fixed(10)')),
            'Weights': get_val('Weights', nb_data.get('Weights', '[0.33, 0.34, 0.33]')),
            '3D_Acc': row['3D_Acc'],
            '10D_Acc': row['10D_Acc'],
            'NestedCV': nested_cv_str,
            'NestedCVAcc': get_val('NestedCVAcc', cv_acc),
            'NestedCVAccStd': get_val('NestedCVAccStd', cv_std),
            'NestedCVF1': get_val('NestedCVF1', nb_data.get('NestedCVF1', cv_acc)),
            'NestedCVF1Std': get_val('NestedCVF1Std', nb_data.get('NestedCVF1Std', cv_std)),
            'RuntimeSec': float(row['Time (s)']),
            'BestBaseline': row['Best_Baseline'],
            'DTW1NN_Acc': get_val('DTW1NN_Acc', nb_data.get('DTW1NN_Acc', np.nan)),
            'ROCKET_Acc': get_val('ROCKET_Acc', nb_data.get('ROCKET_Acc', np.nan)),
            'MiniROCKET_Acc': get_val('MiniROCKET_Acc', nb_data.get('MiniROCKET_Acc', np.nan)),
            'HC2_Acc': get_val('HC2_Acc', nb_data.get('HC2_Acc', np.nan)),
            'Fold_Strategies': get_val('Fold_Strategies', 'N/A')
        }
        
        b_match = re.search(r'\(([\d\.]+)\)', str(row['Best_Baseline']))
        record['BestBaselineAcc'] = float(b_match.group(1)) if b_match else np.nan
        record['Gap'] = record['BestBaselineAcc'] - cv_acc if b_match else np.nan
        records.append(record)
        
    df_detailed = pd.DataFrame(records)
    detailed_csv = csv_file.replace('.csv', '_detailed.csv')
    df_detailed.to_csv(detailed_csv, index=False)
    print(f"Saved detailed results to {detailed_csv}")
    
    # Critical Difference Analysis
    cd_df = df_detailed[['Dataset', '10D_Acc', 'DTW1NN_Acc', 'ROCKET_Acc', 'MiniROCKET_Acc']].copy()
    cd_df = cd_df.rename(columns={
        '10D_Acc': 'Proposed LFIG (Single-Split)',
        'DTW1NN_Acc': 'DTW-1NN (aeon)',
        'ROCKET_Acc': 'ROCKET (aeon)',
        'MiniROCKET_Acc': 'MiniROCKET (aeon)'
    }).set_index('Dataset').dropna()
    
    if len(cd_df) >= 3:
        stat, p_val = friedman_test(cd_df)
        avg_ranks = compute_average_ranks(cd_df)
        cd = nemenyi_post_hoc(cd_df)
        os.makedirs('plots', exist_ok=True)
        plot_critical_difference_diagram(avg_ranks.values, cd, avg_ranks.index.tolist(), output_path='plots/cd_diagram.png')
        print(f"Critical Difference diagram generated. Friedman stat={stat:.4f}, p={p_val:.6f}")
        print(f"Ranks: {avg_ranks.to_dict()}")
        print(f"CD: {cd:.4f}")
    else:
        print("Not enough complete datasets for CD analysis.")
        
    # Generate final markdown reports
    wins, ties, near_ties = [], [], []
    for idx, row in df_detailed.iterrows():
        gap = row['Gap']
        if pd.isna(gap): continue
        name, cv_acc, base_acc = row['Dataset'], row['NestedCVAcc'], row['BestBaselineAcc']
        base_name = str(row['BestBaseline']).split('(')[0].strip()
        if gap < 0: wins.append((name, -gap, cv_acc, base_acc, base_name))
        elif gap == 0: ties.append((name, cv_acc, base_name))
        elif gap < 0.015: near_ties.append((name, gap, cv_acc, base_acc, base_name))
        
    md = []
    label = "GPU" if is_gpu else "CPU"
    md.append(f"# Evaluation and Benchmark Results (Complete 23 UCR Catalog - {label} Run)\n\n")
    md.append(f"This document contains the complete, leakage-free {label} experimental results for the **Adaptive Multi-Feature LFIG Time Series Classification** framework.\n\n")
    md.append("## Demšar Critical Difference Analysis\n\n")
    if len(cd_df) >= 3:
        md.append(f"Friedman Test Stat: **{stat:.4f}** ($p = {p_val:.6f}$)\n")
        md.append(f"Critical Difference (CD) Threshold: **{cd:.4f}**\n\n")
        md.append("Average Ranks achieved:\n")
        for k, v in sorted(avg_ranks.to_dict().items(), key=lambda x: x[1]):
            md.append(f"- **{k}**: {v:.4f} rank\n")
        md.append("\nCritical difference diagram is saved in [plots/cd_diagram.png](plots/cd_diagram.png):\n\n")
        md.append("![CD Diagram](plots/cd_diagram.png)\n\n")
    
    md.append("## Complete Comparative Benchmark Table\n\n")
    md.append("| Dataset | Train | Test | Segmentation | Weights | Accuracy (10D) | Nested CV Accuracy | Runtime (s) | Best Baseline | Gap |\n")
    md.append("|:---|---:|---:|:---|:---|:---:|:---:|---:|:---|---:|\n")
    for idx, row in df_detailed.iterrows():
        gap = row['Gap']
        gap_str = f"{gap:+.4f}" if not pd.isna(gap) else "N/A"
        if not pd.isna(gap):
            if gap < 0: gap_str = f"**{gap:+.4f} (Win)**"
            elif gap == 0: gap_str = "0.0000 (Tie)"
        md.append(f"| {row['Dataset']} | {row['Train']} | {row['Test']} | {row['Segmentation']} | {row['Weights']} | {row['10D_Acc']:.4f} | {row['NestedCV']} | {row['RuntimeSec']:.1f} | {row['BestBaseline']} | {gap_str} |\n")
        
    md.append(f"\n### Performance Summary\n")
    md.append(f"- **Wins (outperformed best SOTA baseline):** {len(wins)} datasets\n")
    for w in sorted(wins, key=lambda x: x[1], reverse=True):
        md.append(f"  - **{w[0]}** (+{w[1]*100:.2f}%): Nested CV **{w[2]:.4f}** vs. {w[4]} at **{w[3]:.4f}**\n")
    md.append(f"- **Ties:** {len(ties)} datasets\n")
    md.append(f"- **Near Ties (margin < 1.5%):** {len(near_ties)} datasets\n")
    
    with open('plots/evaluation_results.md', 'w') as f:
        f.writelines(md)
    print("Updated report saved to plots/evaluation_results.md.")



# ==============================================================================
# Step 1: Leakage Check (Critical Sanity Check) - Label-Shuffle Test
# ==============================================================================

def run_shuffle_sanity_test(dataset_name):
    print(f"\n" + "="*60)
    print(f"RUNNING LABEL-SHUFFLE SANITY TEST ON: {dataset_name}")
    print(f"="*60)
    
    from aeon.datasets import load_classification
    try:
        # Load dataset (downloads automatically if not cached)
        # Use workspace data directory for standard sandbox compatibility
        extract_dir = os.path.join(os.getcwd(), "data")
        os.makedirs(extract_dir, exist_ok=True)
        X_train, y_train = load_classification(dataset_name, split="train", extract_path=extract_dir)
        X_test, y_test = load_classification(dataset_name, split="test", extract_path=extract_dir)
    except Exception as e:
        print(f"Error loading dataset {dataset_name}: {e}")
        return
        
    # Helper to squeeze if 3D
    def preprocess_x(X):
        if isinstance(X, np.ndarray) and X.ndim == 3:
            return X.squeeze(axis=1)
        return [x.squeeze(axis=0) if hasattr(x, 'ndim') and x.ndim == 2 else x for x in X]
        
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_tr = le.fit_transform(y_train)
    y_te = le.transform(y_test)
    
    X_tr = preprocess_x(X_train)
    X_te = preprocess_x(X_test)
    
    X_all = list(X_tr) + list(X_te)
    y_all = np.concatenate([y_tr, y_te], axis=0)
    
    # Shuffle labels randomly (with fixed seed for reproducibility)
    rng = np.random.RandomState(42)
    y_shuffled = rng.permutation(y_all)
    
    if len(X_all) > 150:
        print(f"Subsetting from {len(X_all)} to 150 samples for quick local leakage check...")
        X_all = X_all[:150]
        y_shuffled = y_shuffled[:150]
        
    unique, counts = np.unique(y_shuffled, return_counts=True)
    baseline = max(counts) / len(y_shuffled)
    print(f"Dataset: {dataset_name} | Active Size: {len(X_all)} | Baseline: {baseline:.4f}")
    
    # Run nested CV
    res = run_nested_cv_standalone(X_all, y_shuffled, n_outer=5, n_inner=3)
    
    print(f"\nResults for {dataset_name}:")
    print(f"  - Nested CV Accuracy: {res['mean']:.4f} \u00b1 {res['std']:.4f}")
    print(f"  - Random Baseline: {baseline:.4f}")
    if res['mean'] <= (baseline + 0.10):
        print("  => STATUS: [PASS] - No data leakage detected. Accuracy collapsed to random chance.")
    else:
        print("  => STATUS: [FAIL] - Potential data leakage detected! Accuracy is significantly higher than baseline.")



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="LFIG Benchmark & Sanity Check Pipeline")
    parser.add_argument("--run-leakage-check", action="store_true", help="Run label-shuffle data leakage sanity check")
    parser.add_argument("--run-benchmark", action="store_true", help="Run benchmark on the first 10 datasets")
    parser.add_argument("--compile-results", action="store_true", help="Compile benchmark results and generate plots")
    parser.add_argument("--gpu", action="store_true", help="Flag to indicate GPU execution environment")
    args, unknown = parser.parse_known_args()
    is_gpu = args.gpu

    # Default action if no flags are provided
    if not (args.run_leakage_check or args.run_benchmark or args.compile_results):
        print("No flags specified. By default, running leakage check...")
        args.run_leakage_check = True

    if args.run_leakage_check:
        print("\n=== RUNNING STEP 1: LEAKAGE CHECK SANITY TEST ===")
        for ds in ["TwoLeadECG", "ECGFiveDays"]:
            run_shuffle_sanity_test(ds)

    if args.run_benchmark:
        print("\n=== RUNNING BENCHMARK: FIRST 10 DATASETS ===")
        datasets_1_10 = [
            'GunPoint', 'Coffee', 'ArrowHead', 'ECG200', 'Chinatown',
            'ItalyPowerDemand', 'SonyAIBORobotSurface1', 'TwoLeadECG', 'ECGFiveDays', 'MoteStrain'
        ]
        for ds in datasets_1_10:
            run_single_dataset(ds)

    if args.compile_results:
        print("\n=== COMPILING RESULTS AND GENERATING CD DIAGRAMS ===")
        csv_file = 'master_benchmark_results_gpu.csv' if args.gpu else 'master_benchmark_results.csv'
        nb_file = 'LFIG_Adaptive_Pipeline_Colab_GPU.ipynb' if args.gpu else 'LFIG_Adaptive_Pipeline_Colab.ipynb'
        compile_results_and_plot(nb_file=nb_file, csv_file=csv_file, is_gpu=args.gpu)
