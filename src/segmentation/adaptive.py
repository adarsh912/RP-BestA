import numpy as np
import scipy.stats
import ruptures as rpt

def compute_shannon_entropy(segment, bins=10):
    """
    Computes Shannon entropy of a continuous segment by binning its values.
    """
    if len(segment) < 2:
        return 0.0
    # Add a tiny epsilon to variance to handle flat signals
    if np.std(segment) < 1e-9:
        return 0.0
    
    counts, _ = np.histogram(segment, bins=bins)
    probs = counts / len(segment)
    probs = probs[probs > 0] # Filter out zero probabilities
    return -np.sum(probs * np.log2(probs))

def fixed_segmentation(X_series, window_size=10):
    """
    Partitions the time series into fixed-size windows.
    Returns boundaries: [0, t_1, t_2, ..., N]
    """
    N = len(X_series)
    boundaries = list(range(0, N, window_size))
    if boundaries[-1] != N:
        boundaries.append(N)
    return boundaries

def variance_segmentation(X_series, threshold_var=0.1, min_size=3):
    """
    Partitions the time series dynamically based on cumulative variance.
    A new boundary is created when the variance of the current segment exceeds threshold_var.
    """
    N = len(X_series)
    boundaries = [0]
    start = 0
    
    for i in range(min_size, N):
        current_segment = X_series[start:i+1]
        # If the variance of the segment exceeds threshold, split
        if np.var(current_segment) > threshold_var:
            # Check if remaining segment is too small, if so, don't split yet
            if N - i >= min_size:
                boundaries.append(i)
                start = i
                
    if boundaries[-1] != N:
        boundaries.append(N)
    return boundaries

def entropy_segmentation(X_series, threshold_ent=2.0, min_size=3, bins=10):
    """
    Partitions the time series dynamically based on cumulative Shannon entropy.
    A new boundary is created when entropy of the current segment exceeds threshold_ent.
    """
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
    """
    Partitions the time series using Change Point Detection (BottomUp algorithm from ruptures).
    """
    N = len(X_series)
    if N <= min_size * 2:
        return [0, N]
    
    try:
        # Reshape for ruptures (needs 2D array of shape [n_samples, n_features])
        signal = X_series.reshape(-1, 1)
        algo = rpt.BottomUp(model=model, min_size=min_size).fit(signal)
        # Predict with penalty
        result = algo.predict(pen=penalty)
        # Ensure 0 is at the start
        if len(result) == 0:
            result = [0, N]
        elif result[0] != 0:
            result = [0] + result
        return result
    except Exception as e:
        # Fallback to fixed segmentation if ruptures fails
        print(f"CPD failed, falling back to fixed window: {e}")
        return fixed_segmentation(X_series, window_size=max(10, N // 10))

def segment_time_series(X_series, method="cpd", param=2.0, min_size=3):
    """
    Generic wrapper to segment a time series using the specified method.
    """
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

if __name__ == "__main__":
    # Test script on synthetic signal
    import matplotlib.pyplot as plt
    np.random.seed(42)
    
    # Create synthetic signal with 3 regimes (constant, high variance, linear trend)
    regime1 = np.random.normal(0, 0.1, 50)
    regime2 = np.random.normal(0, 0.8, 50)
    regime3 = np.linspace(0, 3, 50) + np.random.normal(0, 0.1, 50)
    signal = np.concatenate([regime1, regime2, regime3])
    
    print(f"Synthetic signal length: {len(signal)}")
    
    # Run segmentations
    bounds_fixed = segment_time_series(signal, "fixed", param=15)
    bounds_var = segment_time_series(signal, "variance", param=0.2)
    bounds_ent = segment_time_series(signal, "entropy", param=1.8)
    bounds_cpd = segment_time_series(signal, "cpd", param=1.5)
    
    print(f"Fixed boundaries: {bounds_fixed} (count={len(bounds_fixed)-1})")
    print(f"Variance boundaries: {bounds_var} (count={len(bounds_var)-1})")
    print(f"Entropy boundaries: {bounds_ent} (count={len(bounds_ent)-1})")
    print(f"CPD boundaries: {bounds_cpd} (count={len(bounds_cpd)-1})")
