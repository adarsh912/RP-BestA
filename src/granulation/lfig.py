import numpy as np

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
