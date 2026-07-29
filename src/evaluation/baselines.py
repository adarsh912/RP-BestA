import numpy as np
from sklearn.metrics import accuracy_score

def _ensure_3d(X):
    if isinstance(X, list) or (isinstance(X, np.ndarray) and X.dtype == object):
        return [x.reshape(1, -1) if x.ndim == 1 else x for x in X]
    if hasattr(X, 'ndim') and X.ndim == 2:
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
