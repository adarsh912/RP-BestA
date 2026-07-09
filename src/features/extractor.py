import numpy as np
import scipy.stats
from src.granulation.lfig import construct_lfig_granule
from src.segmentation.adaptive import compute_shannon_entropy

def extract_granule_features(segment, z=1.96):
    """
    Extracts a 10-dimensional feature vector for a time series segment.
    
    Features:
    1. Lower bound mean
    2. Upper bound mean
    3. Trend slope (linear coefficient a)
    4. Shannon entropy
    5. Variance
    6. Volatility (std of first differences)
    7. Curvature (quadratic coefficient c in y = c*x^2 + d*x + e)
    8. Intercept (linear coefficient b)
    9. Energy (sum of squares)
    10. Skewness
    """
    L = len(segment)
    tau = np.arange(1, L + 1)
    
    # Standard LFIG construction
    g = construct_lfig_granule(segment, z=z)
    
    # 4. Shannon Entropy
    entropy = compute_shannon_entropy(segment)
    
    # 5. Variance
    variance = np.var(segment) if L > 1 else 0.0
    
    # 6. Volatility
    volatility = np.std(np.diff(segment)) if L > 2 else 0.0
    
    # 7. Curvature (Quadratic fit)
    if L >= 3:
        c, d, _ = np.polyfit(tau, segment, 2)
    else:
        c = 0.0
        
    # 8. Intercept is in g["intercept"]
    
    # 9. Energy
    energy = float(np.sum(segment ** 2))
    
    # 10. Skewness
    if L >= 3 and variance > 1e-9:
        skewness = float(scipy.stats.skew(segment))
    else:
        skewness = 0.0
        
    feature_vector = [
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
    ]
    
    return np.array(feature_vector)

def extract_granular_sequence(X_series, boundaries, z=1.96):
    """
    Transforms a single raw time series into a sequence of 10-dimensional feature vectors.
    """
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
    """
    Transforms an entire dataset of time series into granulated sequences.
    Since segments can have variable lengths, the output is a list of 2D numpy arrays:
    [n_samples] -> each element is an array of shape (n_granules, 10)
    """
    from src.segmentation.adaptive import segment_time_series
    
    granulated_dataset = []
    for i in range(len(X)):
        bounds = segment_time_series(X[i], method=method, param=param)
        seq = extract_granular_sequence(X[i], bounds, z=z)
        granulated_dataset.append(seq)
        
    return granulated_dataset

if __name__ == "__main__":
    # Test script
    np.random.seed(42)
    series = np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.1, 100)
    boundaries = [0, 25, 50, 75, 100]
    
    features = extract_granular_sequence(series, boundaries)
    print("Granular sequence shape:", features.shape)
    print("Sample granule feature vector:")
    feature_names = [
        "Lower Bound Mean", "Upper Bound Mean", "Trend Slope",
        "Shannon Entropy", "Variance", "Volatility",
        "Curvature", "Intercept", "Energy", "Skewness"
    ]
    for name, val in zip(feature_names, features[0]):
        print(f"  {name:20s}: {val:.4f}")
        
    assert features.shape == (4, 10)
    print("Sanity checks passed successfully!")
