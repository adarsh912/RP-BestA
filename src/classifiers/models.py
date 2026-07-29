import numpy as np
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
