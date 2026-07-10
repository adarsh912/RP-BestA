import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

FEATURE_NAMES = [
    'Lower Bound', 'Upper Bound', 'Trend Slope', 'Shannon Entropy',
    'Variance', 'Volatility', 'Curvature', 'Intercept', 'Energy', 'Skewness'
]


def stack_all_granule_features(granular_dataset):
    """
    Stacks all granule feature vectors from all samples into a single 2D array.
    granular_dataset: list of arrays, each (n_granules_i, 10)
    Returns: np.array of shape (total_granules, 10)
    """
    return np.vstack(granular_dataset)


def compute_feature_correlation(granular_dataset, output_path='plots/feature_correlation_matrix.png'):
    """
    Computes and plots Pearson correlation matrix (10x10) of granule features.
    Flags pairs with |r| > 0.85 as highly correlated.
    Returns: (correlation_matrix, high_corr_pairs_list)
    """
    import os
    stacked = stack_all_granule_features(granular_dataset)
    df = pd.DataFrame(stacked, columns=FEATURE_NAMES)
    corr = df.corr(method='pearson')

    # Find highly correlated pairs
    high_corr_pairs = []
    for i in range(len(FEATURE_NAMES)):
        for j in range(i + 1, len(FEATURE_NAMES)):
            r = corr.iloc[i, j]
            if abs(r) > 0.85:
                high_corr_pairs.append((FEATURE_NAMES[i], FEATURE_NAMES[j], round(r, 4)))

    # Plot
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                vmin=-1, vmax=1, xticklabels=FEATURE_NAMES, yticklabels=FEATURE_NAMES, ax=ax)
    ax.set_title('Granule Feature Correlation Matrix')
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Correlation matrix saved to {output_path}")

    return corr.values, high_corr_pairs


def compute_feature_pca(granular_dataset, output_path='plots/feature_pca_variance.png'):
    """
    Runs PCA on stacked granule features.
    Plots cumulative explained variance curve.
    Returns: dict with 'explained_variance_ratio', 'cumulative_variance', 'n_components_95'
    """
    import os
    stacked = stack_all_granule_features(granular_dataset)
    scaled = StandardScaler().fit_transform(stacked)

    pca = PCA().fit(scaled)
    evr = pca.explained_variance_ratio_
    cumulative = np.cumsum(evr)
    n95 = int(np.searchsorted(cumulative, 0.95) + 1)

    # Plot
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    components = np.arange(1, len(evr) + 1)
    ax.bar(components, evr, alpha=0.6, label='Individual')
    ax.step(components, cumulative, where='mid', color='red', label='Cumulative')
    ax.axhline(y=0.95, color='gray', linestyle='--', label='95% threshold')
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Explained Variance Ratio')
    ax.set_title('PCA on Granule Features')
    ax.set_xticks(components)
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"PCA variance plot saved to {output_path}")

    return {
        'explained_variance_ratio': evr,
        'cumulative_variance': cumulative,
        'n_components_95': n95,
    }


def compute_vif(granular_dataset):
    """
    Computes Variance Inflation Factor for each feature.
    VIF_j = 1 / (1 - R^2_j) where R^2_j is from regressing feature j on all others.
    Flags VIF > 10 as problematic multicollinearity.
    Returns: DataFrame with columns ['Feature', 'VIF', 'Problematic']
    """
    stacked = stack_all_granule_features(granular_dataset)
    n_features = stacked.shape[1]
    vifs = []

    for j in range(n_features):
        y = stacked[:, j]
        X = np.delete(stacked, j, axis=1)
        r2 = LinearRegression().fit(X, y).score(X, y)
        vif = 1.0 / (1.0 - r2) if r2 < 1.0 else float('inf')
        vifs.append(vif)

    df = pd.DataFrame({
        'Feature': FEATURE_NAMES,
        'VIF': vifs,
        'Problematic': [v > 10 for v in vifs],
    })
    return df


def run_redundancy_analysis(dataset_name, method='cpd', param=2.5, z=1.96, data_dir='data',
                            output_dir='plots'):
    """
    Full pipeline: load data → granulate → compute correlation, PCA, VIF.
    Prints summary of findings.
    Returns dict with all results.
    """
    import os
    from src.datasets.loader import load_ucr_dataset
    from src.features.extractor import extract_granular_dataset

    os.makedirs(output_dir, exist_ok=True)
    print(f"=== Redundancy Analysis: {dataset_name} ===")

    # Load and granulate
    X_train, y_train, X_test, y_test = load_ucr_dataset(dataset_name, data_dir=data_dir)
    gran = extract_granular_dataset(X_train, method=method, param=param, z=z)
    total = sum(g.shape[0] for g in gran)
    print(f"Total granules from {len(X_train)} training samples: {total}")

    # Correlation
    corr_path = os.path.join(output_dir, 'feature_correlation_matrix.png')
    corr_matrix, high_pairs = compute_feature_correlation(gran, output_path=corr_path)
    print(f"\nHighly correlated pairs (|r| > 0.85): {len(high_pairs)}")
    for fi, fj, r in high_pairs:
        print(f"  {fi} <-> {fj}: r={r:.4f}")

    # PCA
    pca_path = os.path.join(output_dir, 'feature_pca_variance.png')
    pca_results = compute_feature_pca(gran, output_path=pca_path)
    print(f"\nPCA: {pca_results['n_components_95']} components needed for 95% variance")
    print(f"  Cumulative variance: {pca_results['cumulative_variance']}")

    # VIF
    vif_df = compute_vif(gran)
    print(f"\nVariance Inflation Factors:")
    print(vif_df.to_string(index=False))
    problematic = vif_df[vif_df['Problematic']]
    if len(problematic) > 0:
        print(f"\n  WARNING: {len(problematic)} features have VIF > 10 (multicollinearity)")

    return {
        'correlation_matrix': corr_matrix,
        'high_corr_pairs': high_pairs,
        'pca': pca_results,
        'vif': vif_df,
    }
