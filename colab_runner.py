import os
import sys

# Colab Runner and Workspace Recreator for RP BestA
print("=== Google Colab Workspace Recreator & Runner ===")

# 1. Install dependencies
try:
    import numpy
    import scipy
    import pandas
    import sklearn
    import xgboost
    import lightgbm
    import catboost
    import fastdtw
    import ruptures
    import matplotlib
    import seaborn
    import aeon
    print("Dependencies already satisfied.")
except ImportError:
    print("Installing missing dependencies...")
    import subprocess
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", 
        "numpy>=1.20.0", "scipy>=1.7.0", "pandas>=1.3.0", "scikit-learn>=1.0.0", 
        "xgboost>=1.5.0", "lightgbm>=3.3.0", "catboost>=1.0.0", "fastdtw>=0.3.4", 
        "ruptures>=1.1.5", "matplotlib>=3.4.0", "seaborn>=0.11.0", "aeon>=0.7.0", 
        "tabulate>=0.8.9"
    ])
    print("Dependencies installed successfully.")

# 2. Create Directory Structure
directories = [
    "src",
    "src/datasets",
    "src/segmentation",
    "src/granulation",
    "src/features",
    "src/similarity",
    "src/classifiers",
    "src/evaluation",
    "plots"
]

for d in directories:
    os.makedirs(d, exist_ok=True)
print("Created source directory structure.")

# 3. Write Source Files
files_to_create = {}

# --- src/datasets/loader.py ---
files_to_create["src/datasets/loader.py"] = '''import os
import urllib.request
import zipfile
import numpy as np
import pandas as pd

def load_local_ucr_txt(file_path):
    try:
        data = np.loadtxt(file_path)
        y = data[:, 0]
        X = data[:, 1:]
        return X, y
    except Exception as e:
        df = pd.read_csv(file_path, header=None, sep=None, engine='python')
        y = df.iloc[:, 0].values
        X = df.iloc[:, 1:].values
        return X, y

def load_ts_file(file_path):
    X = []
    y = []
    in_data = False
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.lower() == "@data":
                in_data = True
                continue
            if in_data:
                parts = line.split(':')
                if len(parts) < 2:
                    continue
                vals_str = parts[0].split(',')
                vals = [float(v) for v in vals_str]
                label = parts[1].strip()
                X.append(vals)
                try:
                    if '.' in label:
                        y.append(float(label))
                    else:
                        y.append(int(label))
                except ValueError:
                    y.append(label)
    return np.array(X), np.array(y)

def load_ucr_dataset(dataset_name, data_dir="data"):
    dataset_dir = os.path.join(data_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)
    
    train_path_txt = os.path.join(dataset_dir, f"{dataset_name}_TRAIN.txt")
    test_path_txt = os.path.join(dataset_dir, f"{dataset_name}_TEST.txt")
    train_path_tsv = os.path.join(dataset_dir, f"{dataset_name}_TRAIN.tsv")
    test_path_tsv = os.path.join(dataset_dir, f"{dataset_name}_TEST.tsv")
    train_path_ts = os.path.join(dataset_dir, f"{dataset_name}_TRAIN.ts")
    test_path_ts = os.path.join(dataset_dir, f"{dataset_name}_TEST.ts")
    
    if os.path.exists(train_path_txt) and os.path.exists(test_path_txt):
        X_train, y_train = load_local_ucr_txt(train_path_txt)
        X_test, y_test = load_local_ucr_txt(test_path_txt)
        return X_train, y_train, X_test, y_test
    
    if os.path.exists(train_path_tsv) and os.path.exists(test_path_tsv):
        X_train, y_train = load_local_ucr_txt(train_path_tsv)
        X_test, y_test = load_local_ucr_txt(test_path_tsv)
        return X_train, y_train, X_test, y_test

    if os.path.exists(train_path_ts) and os.path.exists(test_path_ts):
        X_train, y_train = load_ts_file(train_path_ts)
        X_test, y_test = load_ts_file(test_path_ts)
        return X_train, y_train, X_test, y_test

    try:
        if dataset_name.lower() == "gunpoint":
            from pyts.datasets import load_gunpoint
            X_train, X_test, y_train, y_test = load_gunpoint(return_X_y=True)
            return X_train, y_train, X_test, y_test
        elif dataset_name.lower() == "coffee":
            from pyts.datasets import load_coffee
            X_train, X_test, y_train, y_test = load_coffee(return_X_y=True)
            return X_train, y_train, X_test, y_test
    except ImportError:
        pass

    try:
        print(f"Attempting to download '{dataset_name}' from the official UCR archive...")
        zip_url = f"https://timeseriesclassification.com/aeon-formatted/{dataset_name}.zip"
        zip_path = os.path.join(dataset_dir, f"{dataset_name}.zip")
        urllib.request.urlretrieve(zip_url, zip_path)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dataset_dir)
            
        os.remove(zip_path)
        
        if os.path.exists(train_path_ts) and os.path.exists(test_path_ts):
            X_train, y_train = load_ts_file(train_path_ts)
            X_test, y_test = load_ts_file(test_path_ts)
            return X_train, y_train, X_test, y_test
    except Exception as e:
        print(f"Official UCR archive download failed: {e}")
        zip_path_temp = os.path.join(dataset_dir, f"{dataset_name}.zip")
        if os.path.exists(zip_path_temp):
            os.remove(zip_path_temp)

    try:
        from sklearn.datasets import fetch_openml
        print(f"Attempting to fetch dataset '{dataset_name}' from OpenML...")
        data = fetch_openml(name=dataset_name, version=1, as_frame=False)
        X = data.data
        y = data.target
        
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        np.savetxt(train_path_txt, np.column_stack((y_train.astype(float), X_train)))
        np.savetxt(test_path_txt, np.column_stack((y_test.astype(float), X_test)))
        
        return X_train, y_train, X_test, y_test
    except Exception as e:
        print(f"OpenML fetch failed: {e}")
        
    mirrors = [
        (f"https://raw.githubusercontent.com/sktime/sktime/main/sktime/datasets/data/{dataset_name}/{dataset_name}_TRAIN.ts", train_path_ts, ".ts"),
        (f"https://raw.githubusercontent.com/hfawaz/cd-diagram/master/{dataset_name}/{dataset_name}_TRAIN.tsv", train_path_tsv, ".tsv"),
        (f"https://raw.githubusercontent.com/ajbagwell/UCR-Time-Series-Archive-2015/master/UCR%20Time%20Series%20Anomaly%20Archive/{dataset_name}/{dataset_name}_TRAIN", train_path_txt, ".txt"),
    ]
    
    for mirror_url, local_path, extension in mirrors:
        try:
            print(f"Trying to fetch train data from mirror: {mirror_url}")
            urllib.request.urlretrieve(mirror_url, local_path)
            
            test_mirror_url = mirror_url.replace("TRAIN", "TEST")
            local_test_path = local_path.replace("TRAIN", "TEST")
            
            print(f"Trying to fetch test data from mirror: {test_mirror_url}")
            urllib.request.urlretrieve(test_mirror_url, local_test_path)
            
            if extension == ".ts":
                X_train, y_train = load_ts_file(local_path)
                X_test, y_test = load_ts_file(local_test_path)
            else:
                X_train, y_train = load_local_ucr_txt(local_path)
                X_test, y_test = load_local_ucr_txt(local_test_path)
            return X_train, y_train, X_test, y_test
        except Exception as e:
            if os.path.exists(local_path):
                os.remove(local_path)
            test_path_temp = local_path.replace("TRAIN", "TEST")
            if os.path.exists(test_path_temp):
                os.remove(test_path_temp)
            continue
            
    raise FileNotFoundError(f"Could not load or download UCR dataset: {dataset_name}")
'''

# --- src/datasets/stats.py ---
files_to_create["src/datasets/stats.py"] = '''import numpy as np
import matplotlib.pyplot as plt
import os

def calculate_dataset_stats(X_train, y_train):
    stats = {
        'n_samples': len(X_train),
        'length': X_train.shape[1],
        'n_classes': len(np.unique(y_train)),
        'distribution': dict(zip(*np.unique(y_train, return_counts=True))),
        'min': float(np.min(X_train)),
        'max': float(np.max(X_train)),
        'mean': float(np.mean(X_train)),
        'std': float(np.std(X_train)),
    }
    return stats
'''

# --- src/datasets/ucr_catalog.py ---
files_to_create["src/datasets/ucr_catalog.py"] = '''import pandas as pd

UCR_CATALOG = [
    ('GunPoint', 'Motion', 50, 150, 150, 2),
    ('Coffee', 'Spectro', 28, 28, 286, 2),
    ('ArrowHead', 'Image', 36, 175, 251, 3),
    ('ECG200', 'ECG', 100, 100, 96, 2),
    ('Chinatown', 'Sensor', 20, 345, 24, 2),
    ('ItalyPowerDemand', 'Sensor', 67, 1029, 24, 2),
    ('SonyAIBORobotSurface1', 'Sensor', 20, 601, 70, 2),
    ('TwoLeadECG', 'ECG', 23, 1139, 82, 2),
    ('ECGFiveDays', 'ECG', 23, 861, 136, 2),
    ('MoteStrain', 'Sensor', 20, 1252, 84, 2),
    ('Beef', 'Spectro', 30, 30, 470, 5),
    ('OliveOil', 'Spectro', 30, 30, 570, 4),
    ('Meat', 'Spectro', 60, 60, 448, 3),
    ('BeetleFly', 'Image', 20, 20, 512, 2),
    ('BirdChicken', 'Image', 20, 20, 512, 2),
    ('FaceFour', 'Image', 24, 88, 350, 4),
    ('SyntheticControl', 'Simulated', 300, 300, 60, 6),
    ('CBF', 'Simulated', 30, 900, 128, 3),
    ('TwoPatterns', 'Simulated', 1000, 4000, 128, 4),
    ('Wafer', 'Sensor', 1000, 6164, 152, 2),
    ('FordA', 'Sensor', 3601, 1320, 500, 2),
    ('Yoga', 'Image', 300, 3000, 426, 2),
    ('SwedishLeaf', 'Image', 500, 625, 128, 15),
]

def get_catalog():
    return [{'name': c[0], 'domain': c[1], 'train_size': c[2], 'test_size': c[3],
             'length': c[4], 'n_classes': c[5]} for c in UCR_CATALOG]

def get_dataset_names(domains=None, max_length=None, max_train_size=None):
    names = []
    for c in UCR_CATALOG:
        name, domain, train_size, _, length, _ = c
        if domains and domain not in domains:
            continue
        if max_length and length > max_length:
            continue
        if max_train_size and train_size > max_train_size:
            continue
        names.append(name)
    return names

def get_catalog_dataframe():
    return pd.DataFrame(get_catalog())
'''

# --- src/segmentation/adaptive.py ---
files_to_create["src/segmentation/adaptive.py"] = '''import numpy as np
import ruptures as rpt

def compute_shannon_entropy(segment, bins=10):
    if len(segment) < 2:
        return 0.0
    if np.std(segment) < 1e-9:
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

def variance_segmentation(X_series, threshold_var=0.1, min_size=3):
    N = len(X_series)
    boundaries = [0]
    start = 0
    for i in range(min_size, N):
        current_segment = X_series[start:i+1]
        if np.var(current_segment) > threshold_var:
            if N - i >= min_size:
                boundaries.append(i)
                start = i
    if boundaries[-1] != N:
        boundaries.append(N)
    return boundaries

def entropy_segmentation(X_series, threshold_ent=2.0, min_size=3, bins=10):
    N = len(X_series)
    boundaries = [0]
    start = 0
    for i in range(min_size, N):
        current_segment = X_series[start:i+1]
        ent = compute_shannon_entropy(current_segment, bins=bins)
        if ent > threshold_ent:
            if N - i >= min_size:
                boundaries.append(i)
                start = i
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
        print(f"CPD failed, falling back to fixed window: {e}")
        return fixed_segmentation(X_series, window_size=max(10, N // 10))

def segment_time_series(X_series, method="cpd", param=2.0, min_size=3):
    if method == "fixed":
        return fixed_segmentation(X_series, window_size=int(param))
    elif method == "variance":
        return variance_segmentation(X_series, threshold_var=float(param), min_size=min_size)
    elif method == "entropy":
        return entropy_segmentation(X_series, threshold_ent=float(param), min_size=min_size)
    elif method == "cpd":
        return cpd_segmentation(X_series, penalty=float(param), min_size=min_size)
    else:
        raise ValueError(f"Unknown segmentation method: {method}")
'''

# --- src/granulation/lfig.py ---
files_to_create["src/granulation/lfig.py"] = '''import numpy as np

def construct_lfig_granule(segment, z=1.96):
    L = len(segment)
    tau = np.arange(1, L + 1)
    if L < 2:
        return {
            "slope": 0.0, "intercept": float(segment[0]) if L == 1 else 0.0,
            "std_residuals": 0.0, "lower_bound_mean": float(segment[0]) if L == 1 else 0.0,
            "upper_bound_mean": float(segment[0]) if L == 1 else 0.0,
            "trend_mean": float(segment[0]) if L == 1 else 0.0,
            "lower_envelope": segment, "upper_envelope": segment, "trend_line": segment, "length": L
        }
    a, b = np.polyfit(tau, segment, 1)
    residuals = segment - (a * tau + b)
    sigma = np.std(residuals)
    trend = a * tau + b
    lower = trend - z * sigma
    upper = trend + z * sigma
    return {
        "slope": float(a), "intercept": float(b), "std_residuals": float(sigma),
        "lower_bound_mean": float(np.mean(lower)), "upper_bound_mean": float(np.mean(upper)),
        "trend_mean": float(np.mean(trend)), "lower_envelope": lower, "upper_envelope": upper,
        "trend_line": trend, "length": L
    }

def granularize_time_series(X_series, boundaries, z=1.96):
    granules = []
    for j in range(len(boundaries) - 1):
        start, end = boundaries[j], boundaries[j+1]
        segment = X_series[start:end]
        if len(segment) == 0:
            continue
        g = construct_lfig_granule(segment, z=z)
        granules.append(g)
    return granules
'''

# --- src/features/extractor.py ---
files_to_create["src/features/extractor.py"] = '''import numpy as np
import scipy.stats
from src.granulation.lfig import construct_lfig_granule
from src.segmentation.adaptive import compute_shannon_entropy

def extract_granule_features(segment, z=1.96):
    L = len(segment)
    tau = np.arange(1, L + 1)
    g = construct_lfig_granule(segment, z=z)
    entropy = compute_shannon_entropy(segment)
    variance = np.var(segment) if L > 1 else 0.0
    volatility = np.std(np.diff(segment)) if L > 2 else 0.0
    if L >= 3:
        c, d, _ = np.polyfit(tau, segment, 2)
    else:
        c = 0.0
    energy = float(np.sum(segment ** 2))
    if L >= 3 and variance > 1e-9:
        skewness = float(scipy.stats.skew(segment))
    else:
        skewness = 0.0
    return np.array([
        g["lower_bound_mean"], g["upper_bound_mean"], g["slope"], entropy,
        variance, volatility, c, g["intercept"], energy, skewness
    ])

def extract_granular_sequence(X_series, boundaries, z=1.96):
    seq = []
    for j in range(len(boundaries) - 1):
        start, end = boundaries[j], boundaries[j+1]
        segment = X_series[start:end]
        if len(segment) == 0:
            continue
        feat = extract_granule_features(segment, z=z)
        seq.append(feat)
    return np.array(seq)

def extract_granular_dataset(X, method="cpd", param=2.5, z=1.96):
    from src.segmentation.adaptive import segment_time_series
    granulated_dataset = []
    for i in range(len(X)):
        bounds = segment_time_series(X[i], method=method, param=param)
        seq = extract_granular_sequence(X[i], bounds, z=z)
        granulated_dataset.append(seq)
    return granulated_dataset
'''

# --- src/features/redundancy.py ---
files_to_create["src/features/redundancy.py"] = '''import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import os

FEATURE_NAMES = [
    'Lower Bound', 'Upper Bound', 'Trend Slope', 'Shannon Entropy',
    'Variance', 'Volatility', 'Curvature', 'Intercept', 'Energy', 'Skewness'
]

def stack_all_granule_features(granular_dataset):
    return np.vstack(granular_dataset)

def compute_feature_correlation(granular_dataset, output_path='plots/feature_correlation_matrix.png'):
    stacked = stack_all_granule_features(granular_dataset)
    df = pd.DataFrame(stacked, columns=FEATURE_NAMES)
    corr = df.corr(method='pearson')
    high_corr_pairs = []
    for i in range(len(FEATURE_NAMES)):
        for j in range(i + 1, len(FEATURE_NAMES)):
            r = corr.iloc[i, j]
            if abs(r) > 0.85:
                high_corr_pairs.append((FEATURE_NAMES[i], FEATURE_NAMES[j], round(r, 4)))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0, vmin=-1, vmax=1, ax=ax)
    ax.set_title('Granule Feature Correlation Matrix')
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return corr.values, high_corr_pairs

def compute_feature_pca(granular_dataset, output_path='plots/feature_pca_variance.png'):
    stacked = stack_all_granule_features(granular_dataset)
    scaled = StandardScaler().fit_transform(stacked)
    pca = PCA().fit(scaled)
    evr = pca.explained_variance_ratio_
    cumulative = np.cumsum(evr)
    n95 = int(np.searchsorted(cumulative, 0.95) + 1)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    components = np.arange(1, len(evr) + 1)
    ax.bar(components, evr, alpha=0.6, label='Individual')
    ax.step(components, cumulative, where='mid', color='red', label='Cumulative')
    ax.axhline(y=0.95, color='gray', linestyle='--', label='95% threshold')
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Explained Variance Ratio')
    ax.set_xticks(components)
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return {'explained_variance_ratio': evr, 'cumulative_variance': cumulative, 'n_components_95': n95}

def compute_vif(granular_dataset):
    stacked = stack_all_granule_features(granular_dataset)
    n_features = stacked.shape[1]
    vifs = []
    for j in range(n_features):
        y = stacked[:, j]
        X = np.delete(stacked, j, axis=1)
        r2 = LinearRegression().fit(X, y).score(X, y)
        vif = 1.0 / (1.0 - r2) if r2 < 1.0 else float('inf')
        vifs.append(vif)
    return pd.DataFrame({'Feature': FEATURE_NAMES, 'VIF': vifs, 'Problematic': [v > 10 for v in vifs]})

def run_redundancy_analysis(dataset_name, method='cpd', param=2.5, z=1.96, data_dir='data', output_dir='plots'):
    from src.datasets.loader import load_ucr_dataset
    from src.features.extractor import extract_granular_dataset
    X_train, y_train, _, _ = load_ucr_dataset(dataset_name, data_dir=data_dir)
    gran = extract_granular_dataset(X_train, method=method, param=param, z=z)
    corr_matrix, high_pairs = compute_feature_correlation(gran, output_path=os.path.join(output_dir, 'feature_correlation_matrix.png'))
    pca_results = compute_feature_pca(gran, output_path=os.path.join(output_dir, 'feature_pca_variance.png'))
    vif_df = compute_vif(gran)
    print(f"=== Redundancy Analysis for {dataset_name} ===")
    print(f"Correlation check: {len(high_pairs)} high correlation pairs.")
    print(f"PCA check: {pca_results['n_components_95']} components explain 95% variance.")
    print("VIF Analysis:\\n", vif_df.to_string(index=False))
    return {'correlation_matrix': corr_matrix, 'high_corr_pairs': high_pairs, 'pca': pca_results, 'vif': vif_df}
'''

# --- src/similarity/hybrid.py ---
files_to_create["src/similarity/hybrid.py"] = '''import numpy as np
from fastdtw import fastdtw

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
    p_to_q = np.max(np.min(dist_matrix, axis=1))
    q_to_p = np.max(np.min(dist_matrix, axis=0))
    return max(p_to_q, q_to_p)

def cosine_distance(v1, v2):
    norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if norm1 < 1e-9 or norm2 < 1e-9:
        return 1.0
    return 1.0 - np.dot(v1, v2) / (norm1 * norm2)

def trend_slope_distance(v1, v2):
    return abs(v1 - v2)

def compute_distances(seq_P, seq_Q):
    d_H = sequence_hausdorff_distance(seq_P[:, 0:2], seq_Q[:, 0:2])
    slopes_P, slopes_Q = seq_P[:, 2], seq_Q[:, 2]
    d_DTW, _ = fastdtw(slopes_P, slopes_Q, dist=trend_slope_distance)
    d_Cos, _ = fastdtw(seq_P, seq_Q, dist=cosine_distance)
    return d_H, d_DTW, d_Cos

def min_max_normalize(D, d_min=None, d_max=None):
    if d_min is None or d_max is None:
        d_min = np.min(D)
        d_max = np.max(D)
    if abs(d_max - d_min) < 1e-9:
        return np.zeros_like(D), d_min, d_max
    return (D - d_min) / (d_max - d_min), d_min, d_max

def compute_pairwise_distances(gran_dataset_X, gran_dataset_Y=None):
    N_X = len(gran_dataset_X)
    symmetric = gran_dataset_Y is None
    N_Y = N_X if symmetric else len(gran_dataset_Y)
    D_H = np.zeros((N_X, N_Y))
    D_DTW = np.zeros((N_X, N_Y))
    D_Cos = np.zeros((N_X, N_Y))
    dataset_Y = gran_dataset_X if symmetric else gran_dataset_Y
    for i in range(N_X):
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
    D_Fused = w[0] * D_H_norm + w[1] * D_DTW_norm + w[2] * D_Cos_norm
    return D_Fused, params

def fuse_borda_ranks(D_H, D_DTW, D_Cos):
    N_X, N_Y = D_H.shape
    Fused_Ranks = np.zeros((N_X, N_Y))
    for i in range(N_X):
        rank_H = np.argsort(np.argsort(D_H[i]))
        rank_DTW = np.argsort(np.argsort(D_DTW[i]))
        rank_Cos = np.argsort(np.argsort(D_Cos[i]))
        Fused_Ranks[i] = rank_H + rank_DTW + rank_Cos
    return Fused_Ranks

def learn_fusion_weights(D_H_train, D_DTW_train, D_Cos_train, y_train):
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
    from sklearn.model_selection import StratifiedKFold
    from src.classifiers.models import CustomDistanceKNN
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
            best_acc = mean_acc
            best_weights = w
    return list(best_weights)
'''

# --- src/classifiers/models.py ---
files_to_create["src/classifiers/models.py"] = '''import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

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

def aggregate_granular_features(granular_dataset):
    agg_features = []
    for seq in granular_dataset:
        if len(seq) == 0:
            agg_features.append(np.zeros(20))
            continue
        mean_feats = np.mean(seq, axis=0)
        std_feats = np.std(seq, axis=0)
        agg_features.append(np.concatenate([mean_feats, std_feats]))
    return np.array(agg_features)

def train_and_evaluate_tabular(X_train_agg, y_train, X_test_agg, y_test, verbose=False):
    unique_labels = np.unique(y_train)
    label_map = {lbl: idx for idx, lbl in enumerate(unique_labels)}
    inv_label_map = {idx: lbl for idx, lbl in enumerate(unique_labels)}
    y_train_mapped = np.array([label_map[y] for y in y_train])
    classifiers = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(kernel='rbf', random_state=42),
        "XGBoost": xgb.XGBClassifier(eval_metric='mlogloss', random_state=42),
        "LightGBM": lgb.LGBMClassifier(random_state=42, verbose=-1),
        "CatBoost": CatBoostClassifier(random_state=42, verbose=0)
    }
    results = {}
    for name, clf in classifiers.items():
        try:
            if name in ["XGBoost", "LightGBM"]:
                clf.fit(X_train_agg, y_train_mapped)
                preds_mapped = clf.predict(X_test_agg)
                preds = np.array([inv_label_map[p] for p in preds_mapped])
            else:
                clf.fit(X_train_agg, y_train)
                preds = clf.predict(X_test_agg)
            acc = accuracy_score(y_test, preds)
            results[name] = {"accuracy": acc, "predictions": preds}
        except Exception as e:
            print(f"Error training {name}: {e}")
    return results

def train_and_evaluate_distance_space(D_train_train, y_train, D_test_train, y_test, verbose=False):
    unique_labels = np.unique(y_train)
    label_map = {lbl: idx for idx, lbl in enumerate(unique_labels)}
    inv_label_map = {idx: lbl for idx, lbl in enumerate(unique_labels)}
    y_train_mapped = np.array([label_map[y] for y in y_train])
    results = {}
    
    # 1. Custom KNN
    knn = CustomDistanceKNN(n_neighbors=3)
    knn.fit(y_train)
    knn_preds = knn.predict(D_test_train)
    results["Distance kNN"] = {"accuracy": accuracy_score(y_test, knn_preds), "predictions": knn_preds}

    # 2. Kernel SVM
    median_d = np.median(D_train_train)
    gamma = 1.0 / (2.0 * (median_d ** 2)) if median_d > 0 else 1.0
    K_train = np.exp(-gamma * (D_train_train ** 2))
    K_test = np.exp(-gamma * (D_test_train ** 2))
    svm = SVC(kernel='precomputed', random_state=42)
    svm.fit(K_train, y_train)
    results["Kernel SVM"] = {"accuracy": accuracy_score(y_test, svm.predict(K_test)), "predictions": svm.predict(K_test)}

    # 3. Distance RF/XGBoost/CatBoost
    dist_classifiers = {
        "Distance RF": RandomForestClassifier(n_estimators=100, random_state=42),
        "Distance XGBoost": xgb.XGBClassifier(eval_metric='mlogloss', random_state=42),
        "Distance CatBoost": CatBoostClassifier(random_state=42, verbose=0)
    }
    for name, clf in dist_classifiers.items():
        try:
            if "XGBoost" in name:
                clf.fit(D_train_train, y_train_mapped)
                preds_mapped = clf.predict(D_test_train)
                preds = np.array([inv_label_map[p] for p in preds_mapped])
            else:
                clf.fit(D_train_train, y_train)
                preds = clf.predict(D_test_train)
            results[name] = {"accuracy": accuracy_score(y_test, preds), "predictions": preds}
        except Exception as e:
            print(f"Error training {name}: {e}")
    return results
'''

# --- src/evaluation/tuning.py ---
files_to_create["src/evaluation/tuning.py"] = '''import numpy as np
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

def _evaluate_pipeline(X_train, y_train, X_test, y_test, method, param, z, k, weights, clf_type):
    gran_train = []
    for x in X_train:
        bounds = segment_time_series(x, method=method, param=param)
        gran_train.append(extract_granular_sequence(x, bounds, z=z))
    gran_test = []
    for x in X_test:
        bounds = segment_time_series(x, method=method, param=param)
        gran_test.append(extract_granular_sequence(x, bounds, z=z))
    D_H_train, D_DTW_train, D_Cos_train = compute_pairwise_distances(gran_train)
    D_H_test, D_DTW_test, D_Cos_test = compute_pairwise_distances(gran_test, gran_train)
    D_train_fused, mm_params = fuse_distances(D_H_train, D_DTW_train, D_Cos_train, weights=weights)
    D_test_fused, _ = fuse_distances(D_H_test, D_DTW_test, D_Cos_test, weights=weights, min_max_params=mm_params)
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
    return accuracy_score(y_test, preds)

def run_inner_cv(X_train_fold, y_train_fold, method, param_grid, n_inner_folds=3):
    skf = StratifiedKFold(n_splits=n_inner_folds, shuffle=True, random_state=42)
    keys, values = list(param_grid.keys()), list(param_grid.values())
    combos = list(product(*values))
    best_acc, best_params = -1.0, None
    for idx, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        fold_accs = []
        for train_idx, val_idx in skf.split(X_train_fold, y_train_fold):
            X_tr, X_val = X_train_fold[train_idx], X_train_fold[val_idx]
            y_tr, y_val = y_train_fold[train_idx], y_train_fold[val_idx]
            try:
                acc = _evaluate_pipeline(X_tr, y_tr, X_val, y_val, method=method, param=params['param'], z=params['z'], k=params['k'], weights=params['weights'], clf_type=params['clf_type'])
                fold_accs.append(acc)
            except Exception as e:
                fold_accs.append(0.0)
        mean_acc = np.mean(fold_accs)
        if mean_acc > best_acc:
            best_acc = mean_acc
            best_params = params
    return best_params

def run_nested_cv(dataset_name, n_outer_folds=5, n_inner_folds=3, data_dir='data'):
    X_train, y_train, X_test, y_test = load_ucr_dataset(dataset_name, data_dir=data_dir)
    X_full = np.concatenate([X_train, X_test], axis=0)
    y_full = np.concatenate([y_train, y_test], axis=0)
    outer_skf = StratifiedKFold(n_splits=n_outer_folds, shuffle=True, random_state=42)
    param_grid = {
        'param': [1.0, 1.5, 2.0],
        'z': [0.5, 1.0, 1.96],
        'k': [1, 3],
        'weights': [[0.1, 0.8, 0.1], [0.3, 0.4, 0.3]],
        'clf_type': ['KNN', 'Kernel SVM']
    }
    metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
    for fold_i, (train_idx, test_idx) in enumerate(outer_skf.split(X_full, y_full)):
        X_tr, X_te = X_full[train_idx], X_full[test_idx]
        y_tr, y_te = y_full[train_idx], y_full[test_idx]
        method, _ = select_segmentation_strategy(X_tr, y_tr)
        best_params = run_inner_cv(X_tr, y_tr, method, param_grid, n_inner_folds=n_inner_folds)
        acc = _evaluate_pipeline(X_tr, y_tr, X_te, y_te, method=method, param=best_params['param'], z=best_params['z'], k=best_params['k'], weights=best_params['weights'], clf_type=best_params['clf_type'])
        metrics['accuracy'].append(acc)
        print(f"  Fold {fold_i + 1} accuracy: {acc:.4f}")
    print(f"Nested CV Accuracy for {dataset_name}: {np.mean(metrics['accuracy']):.4f}±{np.std(metrics['accuracy']):.4f}")
    return {'accuracy_mean': np.mean(metrics['accuracy']), 'accuracy_std': np.std(metrics['accuracy'])}
'''

# --- src/evaluation/baselines.py ---
files_to_create["src/evaluation/baselines.py"] = '''import numpy as np
from sklearn.metrics import accuracy_score

def _ensure_3d(X):
    if X.ndim == 2:
        return X.reshape(X.shape[0], 1, X.shape[1])
    return X

def run_rocket_baseline(X_train, y_train, X_test, y_test):
    from aeon.classification.convolution_based import RocketClassifier
    X_tr, X_te = _ensure_3d(X_train), _ensure_3d(X_test)
    clf = RocketClassifier(random_state=42)
    clf.fit(X_tr, y_train)
    preds = clf.predict(X_te)
    return preds, accuracy_score(y_test, preds)

def run_minirocket_baseline(X_train, y_train, X_test, y_test):
    from aeon.classification.convolution_based import MiniRocketClassifier
    X_tr, X_te = _ensure_3d(X_train), _ensure_3d(X_test)
    clf = MiniRocketClassifier(random_state=42)
    clf.fit(X_tr, y_train)
    preds = clf.predict(X_te)
    return preds, accuracy_score(y_test, preds)

def run_dtw_1nn_baseline(X_train, y_train, X_test, y_test):
    from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier
    X_tr, X_te = _ensure_3d(X_train), _ensure_3d(X_test)
    clf = KNeighborsTimeSeriesClassifier(n_neighbors=1, distance='dtw')
    clf.fit(X_tr, y_train)
    preds = clf.predict(X_te)
    return preds, accuracy_score(y_test, preds)

def run_all_baselines(X_train, y_train, X_test, y_test):
    baselines = {'ROCKET': run_rocket_baseline, 'MiniROCKET': run_minirocket_baseline, 'DTW-1NN': run_dtw_1nn_baseline}
    results = {}
    for name, fn in baselines.items():
        try:
            preds, acc = fn(X_train, y_train, X_test, y_test)
            results[name] = {'accuracy': acc, 'predictions': preds}
        except Exception as e:
            print(f"Baseline {name} failed: {e}")
    return results
'''

# --- src/evaluation/critical_difference.py ---
files_to_create["src/evaluation/critical_difference.py"] = '''import numpy as np
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
    q_alpha = _NEMENYI_Q_ALPHA[k]
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
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
'''

# --- src/evaluation/benchmark.py ---
files_to_create["src/evaluation/benchmark.py"] = '''import os
import time
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from src.datasets.loader import load_ucr_dataset
from src.evaluation.tuning import run_nested_cv

def run_full_benchmark(dataset_names=["GunPoint", "Coffee"]):
    results = []
    for dataset in dataset_names:
        print(f"Benchmarking dataset: {dataset}...")
        res = run_nested_cv(dataset)
        results.append({
            'Dataset': dataset,
            'Proposed Nested CV Acc': f"{res['accuracy_mean']:.4f}±{res['accuracy_std']:.4f}"
        })
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    return df
'''

# Write all __init__.py files
for path in ["src/datasets/__init__.py", "src/segmentation/__init__.py", 
             "src/granulation/__init__.py", "src/features/__init__.py", 
             "src/similarity/__init__.py", "src/classifiers/__init__.py", 
             "src/evaluation/__init__.py"]:
    files_to_create[path] = ""

# Actually create and write all files
for file_path, content in files_to_create.items():
    with open(file_path, "w") as f:
        f.write(content)
print(f"Created/wrote all {len(files_to_create)} source files successfully.")

# 4. Trigger a sample verification run
print("\n=== Running Verification Tests on Colab Workspace ===")
from src.evaluation.tuning import run_nested_cv
from src.features.redundancy import run_redundancy_analysis

try:
    print("\n--- Running Nested CV on GunPoint (Outer=2, Inner=2) ---")
    run_nested_cv("GunPoint", n_outer_folds=2, n_inner_folds=2)
except Exception as e:
    print(f"Nested CV Test failed: {e}")

try:
    print("\n--- Running Redundancy Analysis on GunPoint ---")
    run_redundancy_analysis("GunPoint")
except Exception as e:
    print(f"Redundancy Analysis failed: {e}")

print("\n=== Workspace ready for use! ===")
print("You can run the full benchmark using:")
print('python -c "from src.evaluation.benchmark import run_full_benchmark; run_full_benchmark()"')

