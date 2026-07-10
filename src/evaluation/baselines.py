"""
Reproducible baseline classifiers using the aeon library for time series classification.
"""

import warnings
import numpy as np
from sklearn.metrics import accuracy_score


def _ensure_3d(X):
    """Reshape to (n_samples, 1, n_timepoints) if needed for aeon's univariate format."""
    if X.ndim == 2:
        return X.reshape(X.shape[0], 1, X.shape[1])
    return X


def run_rocket_baseline(X_train, y_train, X_test, y_test):
    """
    Runs ROCKET classifier from aeon.
    Returns: (predictions, accuracy)
    """
    from aeon.classification.convolution_based import RocketClassifier

    X_tr, X_te = _ensure_3d(X_train), _ensure_3d(X_test)
    clf = RocketClassifier(random_state=42)
    clf.fit(X_tr, y_train)
    preds = clf.predict(X_te)
    return preds, accuracy_score(y_test, preds)


def run_minirocket_baseline(X_train, y_train, X_test, y_test):
    """
    Runs MiniROCKET classifier from aeon.
    Returns: (predictions, accuracy)
    """
    from aeon.classification.convolution_based import MiniRocketClassifier

    X_tr, X_te = _ensure_3d(X_train), _ensure_3d(X_test)
    clf = MiniRocketClassifier(random_state=42)
    clf.fit(X_tr, y_train)
    preds = clf.predict(X_te)
    return preds, accuracy_score(y_test, preds)


def run_dtw_1nn_baseline(X_train, y_train, X_test, y_test):
    """
    Runs 1-NN with DTW distance from aeon.
    Returns: (predictions, accuracy)
    """
    from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier

    X_tr, X_te = _ensure_3d(X_train), _ensure_3d(X_test)
    clf = KNeighborsTimeSeriesClassifier(n_neighbors=1, distance='dtw')
    clf.fit(X_tr, y_train)
    preds = clf.predict(X_te)
    return preds, accuracy_score(y_test, preds)


def run_all_baselines(X_train, y_train, X_test, y_test):
    """
    Runs all available baselines. Returns dict mapping name → {accuracy, predictions}.
    Catches ImportError for unavailable classifiers and logs warnings.
    """
    baselines = {
        'ROCKET': run_rocket_baseline,
        'MiniROCKET': run_minirocket_baseline,
        'DTW-1NN': run_dtw_1nn_baseline,
    }

    results = {}
    for name, fn in baselines.items():
        try:
            preds, acc = fn(X_train, y_train, X_test, y_test)
            results[name] = {'accuracy': acc, 'predictions': preds}
            print(f"{name}: accuracy={acc:.4f}")
        except ImportError as e:
            warnings.warn(f"{name} unavailable (missing dependency): {e}")
        except Exception as e:
            warnings.warn(f"{name} failed: {e}")

    return results
