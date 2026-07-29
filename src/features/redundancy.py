import numpy as np
import pandas as pd
from src.datasets.loader import load_ucr_dataset
from src.features.extractor import extract_granular_dataset

FEATURE_NAMES = [
    'Lower Bound', 'Upper Bound', 'Trend Slope', 'Shannon Entropy',
    'Variance', 'Volatility', 'Curvature', 'Intercept', 'Energy', 'Skewness'
]

def check_feature_correlations(granulated_dataset, threshold=0.9):
    all_granules = np.vstack(granulated_dataset)
    df = pd.DataFrame(all_granules, columns=FEATURE_NAMES)
    corr_matrix = df.corr().abs()
    high_corr_pairs = []
    for i in range(len(FEATURE_NAMES)):
        for j in range(i + 1, len(FEATURE_NAMES)):
            val = corr_matrix.iloc[i, j]
            if val > threshold:
                high_corr_pairs.append((FEATURE_NAMES[i], FEATURE_NAMES[j], val))
    return corr_matrix, high_corr_pairs

def compute_pca_variance(granulated_dataset):
    from sklearn.decomposition import PCA
    all_granules = np.vstack(granulated_dataset)
    pca = PCA()
    pca.fit(all_granules)
    explained_var = pca.explained_variance_ratio_
    cum_var = np.cumsum(explained_var)
    n_95 = np.argmax(cum_var >= 0.95) + 1
    return {
        'explained_variance_ratio': explained_var,
        'cumulative_variance': cum_var,
        'n_components_95': n_95
    }

def compute_vif(granulated_dataset):
    all_granules = np.vstack(granulated_dataset)
    df = pd.DataFrame(all_granules, columns=FEATURE_NAMES)
    vif_data = []
    for i in range(len(FEATURE_NAMES)):
        y = df.iloc[:, i]
        X = df.drop(df.columns[i], axis=1)
        r2 = np.corrcoef(y, X.mean(axis=1))[0, 1] ** 2 if not np.isnan(r2 := np.corrcoef(y, X.mean(axis=1))[0, 1]) else 0
        vif = 1.0 / (1.0 - r2 + 1e-9)
        vif_data.append({'Feature': FEATURE_NAMES[i], 'VIF': vif})
    return pd.DataFrame(vif_data)

def run_redundancy_analysis(dataset_name, method="cpd", param=2.5, z=1.96, data_dir='data'):
    X_train, y_train, X_test, y_test = load_ucr_dataset(dataset_name, data_dir=data_dir)
    granulated_train = extract_granular_dataset(X_train, method=method, param=param, z=z)
    corr_matrix, high_pairs = check_feature_correlations(granulated_train)
    pca_results = compute_pca_variance(granulated_train)
    vif_df = compute_vif(granulated_train)
    print(f"=== Redundancy Analysis for {dataset_name} ===")
    print(f"Correlation check: {len(high_pairs)} high correlation pairs.")
    print(f"PCA check: {pca_results['n_components_95']} components explain 95% variance.")
    print("VIF Analysis:\n", vif_df.to_string(index=False))
    return {'correlation_matrix': corr_matrix, 'high_corr_pairs': high_pairs, 'pca': pca_results, 'vif': vif_df}
