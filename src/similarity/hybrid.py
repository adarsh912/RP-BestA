import numpy as np
from fastdtw import fastdtw

def interval_hausdorff_dist(L1, U1, L2, U2):
    """
    Computes the Hausdorff distance between two intervals [L1, U1] and [L2, U2].
    """
    return max(abs(L1 - L2), abs(U1 - U2))

def sequence_hausdorff_distance(seq_P, seq_Q):
    """
    Computes the Hausdorff distance between two sequences of bounding intervals.
    seq_P and seq_Q contain lower bound in column 0 and upper bound in column 1.
    """
    K = len(seq_P)
    M = len(seq_Q)
    
    # Calculate all pairwise interval distances
    dist_matrix = np.zeros((K, M))
    for k in range(K):
        for m in range(M):
            dist_matrix[k, m] = interval_hausdorff_dist(
                seq_P[k, 0], seq_P[k, 1], seq_Q[m, 0], seq_Q[m, 1]
            )
            
    # H(P, Q) = max( max_k min_m dist(p_k, q_m), max_m min_k dist(p_k, q_m) )
    p_to_q = np.max(np.min(dist_matrix, axis=1))
    q_to_p = np.max(np.min(dist_matrix, axis=0))
    return max(p_to_q, q_to_p)

def cosine_distance(v1, v2):
    """
    Computes cosine distance (1 - cosine_similarity) between two vectors.
    """
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-9 or norm2 < 1e-9:
        return 1.0 # Or maximum distance
    return 1.0 - np.dot(v1, v2) / (norm1 * norm2)

def trend_slope_distance(v1, v2):
    """
    Absolute difference distance between two trend slopes.
    """
    return abs(v1 - v2)

def compute_distances(seq_P, seq_Q):
    """
    Computes the three distance components between two granular sequences:
    1. Hausdorff distance (interval overlap)
    2. DTW on trend slopes
    3. DTW on all features using Cosine distance
    """
    # 1. Hausdorff
    d_H = sequence_hausdorff_distance(seq_P[:, 0:2], seq_Q[:, 0:2])
    
    # 2. DTW on slopes (column index 2)
    slopes_P = seq_P[:, 2]
    slopes_Q = seq_Q[:, 2]
    # fastdtw expects sequences, returns (distance, path)
    d_DTW, _ = fastdtw(slopes_P, slopes_Q, dist=trend_slope_distance)
    
    # 3. Cosine DTW on 10D features
    d_Cos, _ = fastdtw(seq_P, seq_Q, dist=cosine_distance)
    
    return d_H, d_DTW, d_Cos

def min_max_normalize(D, d_min=None, d_max=None):
    """
    Min-max normalizes a distance matrix to [0, 1].
    """
    if d_min is None or d_max is None:
        d_min = np.min(D)
        d_max = np.max(D)
        
    if abs(d_max - d_min) < 1e-9:
        return np.zeros_like(D), d_min, d_max
    return (D - d_min) / (d_max - d_min), d_min, d_max

def compute_pairwise_distances(gran_dataset_X, gran_dataset_Y=None):
    """
    Computes matrices of the three distance components pairwise between dataset samples.
    """
    N_X = len(gran_dataset_X)
    symmetric = gran_dataset_Y is None
    N_Y = N_X if symmetric else len(gran_dataset_Y)
    
    D_H = np.zeros((N_X, N_Y))
    D_DTW = np.zeros((N_X, N_Y))
    D_Cos = np.zeros((N_X, N_Y))
    
    dataset_Y = gran_dataset_X if symmetric else gran_dataset_Y
    
    for i in range(N_X):
        # In symmetric case, we can compute only upper triangle
        start_j = i if symmetric else 0
        for j in range(start_j, N_Y):
            dh, ddtw, dcos = compute_distances(gran_dataset_X[i], dataset_Y[j])
            D_H[i, j] = dh
            D_DTW[i, j] = ddtw
            D_Cos[i, j] = dcos
            
            if symmetric:
                D_H[j, i] = dh
                D_DTW[j, i] = ddtw
                D_Cos[j, i] = dcos
                
    return D_H, D_DTW, D_Cos

def fuse_distances(D_H, D_DTW, D_Cos, weights=[0.3, 0.4, 0.3], min_max_params=None):
    """
    Normalizes and fuses the distance matrices.
    If min_max_params is provided, it contains: (min_H, max_H, min_DTW, max_DTW, min_Cos, max_Cos)
    """
    w = weights
    
    if min_max_params is None:
        # Calculate training min/max values
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
        
    D_Fused = w[0] * D_H_norm + w[1] * D_DTW_norm + w[2] * D_Cos_norm
    return D_Fused, params

def fuse_borda_ranks(D_H, D_DTW, D_Cos):
    """
    Alternative fusion: Borda Count rank aggregation.
    For each test sample (row i), we sort all training samples (columns j) in ascending order.
    The rank (0 to N-1) is assigned. We sum ranks across all three distances.
    """
    N_X, N_Y = D_H.shape
    Fused_Ranks = np.zeros((N_X, N_Y))
    
    for i in range(N_X):
        rank_H = np.argsort(np.argsort(D_H[i]))
        rank_DTW = np.argsort(np.argsort(D_DTW[i]))
        rank_Cos = np.argsort(np.argsort(D_Cos[i]))
        
        Fused_Ranks[i] = rank_H + rank_DTW + rank_Cos
        
    return Fused_Ranks

def learn_fusion_weights(D_H_train, D_DTW_train, D_Cos_train, y_train):
    """
    Learns optimal fusion weights from training data using logistic regression
    on pairwise same-class labels.

    For each pair (i, j) in training set:
    - Label = 1 if y_i == y_j (same class), else 0
    - Features = [normalized_d_H, normalized_d_DTW, normalized_d_Cos]

    Fits LogisticRegression, extracts coefficients, and normalizes them to sum to 1.
    Returns learned weights as list of 3 floats.
    """
    from sklearn.linear_model import LogisticRegression

    D_H_norm, _, _ = min_max_normalize(D_H_train)
    D_DTW_norm, _, _ = min_max_normalize(D_DTW_train)
    D_Cos_norm, _, _ = min_max_normalize(D_Cos_train)

    y = np.asarray(y_train)
    n = len(y)
    features, labels = [], []
    for i in range(n):
        for j in range(i + 1, n):
            features.append([D_H_norm[i, j], D_DTW_norm[i, j], D_Cos_norm[i, j]])
            labels.append(1 if y[i] == y[j] else 0)

    clf = LogisticRegression(max_iter=1000).fit(features, labels)
    raw = np.abs(clf.coef_[0])
    weights = raw / raw.sum()
    return weights.tolist()


def learn_fusion_weights_grid(D_H_train, D_DTW_train, D_Cos_train, y_train, k=3, cv=3):
    """
    Learns optimal fusion weights via grid search on inner CV.

    Weight grid: predefined list of weight combinations that sum to 1.
    For each weight combo, fuse training distances, run KNN(k) with
    StratifiedKFold(cv), and measure mean accuracy.
    Returns best weights as list of 3 floats.
    """
    from sklearn.model_selection import StratifiedKFold
    from src.classifiers.models import CustomDistanceKNN

    weight_grid = [
        [0.1, 0.8, 0.1], [0.2, 0.6, 0.2], [0.3, 0.4, 0.3],
        [0.33, 0.34, 0.33], [0.1, 0.1, 0.8], [0.8, 0.1, 0.1],
        [0.4, 0.4, 0.2], [0.2, 0.4, 0.4], [0.4, 0.2, 0.4],
    ]

    y = np.asarray(y_train)
    min_class_size = np.min(np.unique(y, return_counts=True)[1]) if len(y) > 0 else 0
    actual_cv = min(cv, min_class_size)
    if actual_cv < 2:
        from sklearn.model_selection import KFold
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
            best_acc = mean_acc
            best_weights = w

    return list(best_weights)


if __name__ == "__main__":
    # Test script
    np.random.seed(42)
    # Generate 3 small granular sequences (each representing 4 granules of 10 features)
    seq1 = np.random.normal(0, 1, (4, 10))
    seq2 = np.random.normal(0.5, 1, (4, 10))
    seq3 = np.random.normal(0, 1.2, (4, 10))
    
    # Ensure Lower bound < Upper bound for Hausdorff test
    seq1[:, 0] = seq1[:, 0:2].min(axis=1) - 0.5
    seq1[:, 1] = seq1[:, 0:2].max(axis=1) + 0.5
    seq2[:, 0] = seq2[:, 0:2].min(axis=1) - 0.5
    seq2[:, 1] = seq2[:, 0:2].max(axis=1) + 0.5
    seq3[:, 0] = seq3[:, 0:2].min(axis=1) - 0.5
    seq3[:, 1] = seq3[:, 0:2].max(axis=1) + 0.5
    
    dataset = [seq1, seq2, seq3]
    
    D_H, D_DTW, D_Cos = compute_pairwise_distances(dataset)
    print("Hausdorff Distance Matrix:\n", D_H)
    print("DTW on Slopes Matrix:\n", D_DTW)
    print("Cosine DTW Matrix:\n", D_Cos)
    
    D_Fused, params = fuse_distances(D_H, D_DTW, D_Cos)
    print("Fused Distance Matrix:\n", D_Fused)
    print("Min-Max parameters:", params)
    
    Ranks = fuse_borda_ranks(D_H, D_DTW, D_Cos)
    print("Borda Fused Ranks:\n", Ranks)
    
    # Assert diagonal of distance matrices are zero
    assert abs(D_H[0, 0]) < 1e-9
    assert abs(D_DTW[0, 0]) < 1e-9
    assert abs(D_Cos[0, 0]) < 1e-9
    assert abs(D_Fused[0, 0]) < 1e-9
    print("All assertions passed: Diagonal distances are zero!")
