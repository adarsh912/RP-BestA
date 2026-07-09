import numpy as np

def construct_lfig_granule(segment, z=1.96):
    """
    Constructs a Linear Fuzzy Information Granule (LFIG) for a single time series segment.
    
    Args:
        segment (np.ndarray): The time series values in the segment.
        z (float): Coverage parameter determining the spread of the fuzzy bounds.
        
    Returns:
        dict: Containing trend parameters, bounds, and fit residuals.
    """
    L = len(segment)
    tau = np.arange(1, L + 1)
    
    # 1. Fit linear regression y = a_j * tau + b_j
    # If the segment is extremely short, handle it
    if L < 2:
        a = 0.0
        b = float(segment[0]) if L == 1 else 0.0
        residuals = np.array([0.0])
        sigma = 0.0
    else:
        a, b = np.polyfit(tau, segment, 1)
        residuals = segment - (a * tau + b)
        sigma = np.std(residuals)
        
    # 2. Compute envelopes (bounds)
    lower_envelope = a * tau + b - z * sigma
    upper_envelope = a * tau + b + z * sigma
    
    # 3. Compute summary statistics (means)
    mean_lower = np.mean(lower_envelope)
    mean_upper = np.mean(upper_envelope)
    mean_trend = a * np.mean(tau) + b
    
    return {
        "slope": a,
        "intercept": b,
        "std_residuals": sigma,
        "lower_bound_mean": mean_lower,
        "upper_bound_mean": mean_upper,
        "trend_mean": mean_trend,
        "lower_envelope": lower_envelope,
        "upper_envelope": upper_envelope,
        "trend_line": a * tau + b,
        "length": L
    }

def granularize_time_series(X_series, boundaries, z=1.96):
    """
    Granularizes a time series using a list of segment boundaries.
    
    Args:
        X_series (np.ndarray): Univariate time series.
        boundaries (list): List of segment boundaries [0, t_1, t_2, ..., N].
        z (float): Coverage parameter.
        
    Returns:
        list: List of granule dictionaries.
    """
    granules = []
    for j in range(len(boundaries) - 1):
        start, end = boundaries[j], boundaries[j+1]
        segment = X_series[start:end]
        if len(segment) == 0:
            continue
        granule = construct_lfig_granule(segment, z=z)
        granule["start"] = start
        granule["end"] = end
        granules.append(granule)
    return granules

if __name__ == "__main__":
    # Quick sanity check
    segment = np.array([1.2, 1.5, 1.8, 2.2, 2.1, 2.5, 3.0])
    granule = construct_lfig_granule(segment, z=1.5)
    print("Granule constructed successfully:")
    print(f"Slope (Trend): {granule['slope']:.3f}")
    print(f"Lower Mean: {granule['lower_bound_mean']:.3f}")
    print(f"Upper Mean: {granule['upper_bound_mean']:.3f}")
    print(f"Trend Mean: {granule['trend_mean']:.3f}")
    # Verify bounds properties: lower <= trend <= upper
    assert granule['lower_bound_mean'] <= granule['trend_mean'] <= granule['upper_bound_mean']
    print("Assertion passed: Bounds are mathematically sound!")
