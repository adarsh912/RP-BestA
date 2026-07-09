import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

class CustomDistanceKNN:
    """
    K-Nearest Neighbors classifier utilizing a custom precomputed distance matrix.
    """
    def __init__(self, n_neighbors=3):
        self.n_neighbors = n_neighbors
        self.X_train_labels = None
        
    def fit(self, y_train):
        self.X_train_labels = np.array(y_train)
        # Prevent index out of bounds if training sample count is less than n_neighbors
        self.n_neighbors = min(self.n_neighbors, len(self.X_train_labels))
        return self
        
    def predict(self, D_test_train):
        """
        D_test_train: shape (n_test_samples, n_train_samples)
                      Contains distances from test samples to training samples.
        """
        n_test = D_test_train.shape[0]
        predictions = []
        
        for i in range(n_test):
            # Sort training sample indices by distance to this test sample
            nearest_indices = np.argsort(D_test_train[i])[:self.n_neighbors]
            nearest_labels = self.X_train_labels[nearest_indices]
            
            # Majority vote
            unique_labels, counts = np.unique(nearest_labels, return_counts=True)
            predictions.append(unique_labels[np.argmax(counts)])
            
        return np.array(predictions)

def aggregate_granular_features(granular_dataset):
    """
    Aggregates variable-length granular feature sequences into a fixed-length
    20-dimensional feature vector (10 means, 10 stds).
    """
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
    """
    Trains and evaluates Random Forest, XGBoost, LightGBM, CatBoost, SVM,
    and standard tabular classifiers on the aggregated 20D feature space.
    """
    # Map labels to 0-indexed integers for XGBoost and LightGBM
    unique_labels = np.unique(y_train)
    label_map = {lbl: idx for idx, lbl in enumerate(unique_labels)}
    inv_label_map = {idx: lbl for idx, lbl in enumerate(unique_labels)}
    
    y_train_mapped = np.array([label_map[y] for y in y_train])
    y_test_mapped = np.array([label_map[y] for y in y_test])
    
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
            # XGBoost and LightGBM require mapped labels
            if name in ["XGBoost", "LightGBM"]:
                clf.fit(X_train_agg, y_train_mapped)
                preds_mapped = clf.predict(X_test_agg)
                preds = np.array([inv_label_map[p] for p in preds_mapped])
            else:
                clf.fit(X_train_agg, y_train)
                preds = clf.predict(X_test_agg)
                
            acc = accuracy_score(y_test, preds)
            results[name] = {
                "accuracy": acc,
                "predictions": preds
            }
            if verbose:
                print(f"{name} Tabular Accuracy: {acc:.4f}")
        except Exception as e:
            print(f"Error training {name}: {e}")
            
    return results

def train_and_evaluate_distance_space(D_train_train, y_train, D_test_train, y_test, verbose=False):
    """
    Trains and evaluates classifiers using the Fused Distance Matrix columns as features.
    Plus the custom Distance kNN.
    """
    # Pre-process labels for boosting
    unique_labels = np.unique(y_train)
    label_map = {lbl: idx for idx, lbl in enumerate(unique_labels)}
    inv_label_map = {idx: lbl for idx, lbl in enumerate(unique_labels)}
    
    y_train_mapped = np.array([label_map[y] for y in y_train])
    y_test_mapped = np.array([label_map[y] for y in y_test])
    
    results = {}
    
    # 1. Custom kNN (k=3)
    knn = CustomDistanceKNN(n_neighbors=3)
    knn.fit(y_train)
    knn_preds = knn.predict(D_test_train)
    results["Distance kNN"] = {
        "accuracy": accuracy_score(y_test, knn_preds),
        "predictions": knn_preds
    }
    if verbose:
        print(f"Distance kNN Accuracy: {results['Distance kNN']['accuracy']:.4f}")
        
    # 2. Kernel SVM (precomputed)
    # Convert distance to similarity kernel K = exp(-gamma * D^2)
    # Compute median distance to set gamma
    median_d = np.median(D_train_train)
    gamma = 1.0 / (2.0 * (median_d ** 2)) if median_d > 0 else 1.0
    
    K_train = np.exp(-gamma * (D_train_train ** 2))
    K_test = np.exp(-gamma * (D_test_train ** 2))
    
    svm = SVC(kernel='precomputed', random_state=42)
    svm.fit(K_train, y_train)
    svm_preds = svm.predict(K_test)
    results["Kernel SVM"] = {
        "accuracy": accuracy_score(y_test, svm_preds),
        "predictions": svm_preds
    }
    if verbose:
        print(f"Kernel SVM Accuracy: {results['Kernel SVM']['accuracy']:.4f}")
        
    # 3. Distance-as-features models (RF, XGBoost, CatBoost)
    # Here the features of sample i is its distance to all training samples
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
                
            acc = accuracy_score(y_test, preds)
            results[name] = {
                "accuracy": acc,
                "predictions": preds
            }
            if verbose:
                print(f"{name} Accuracy: {acc:.4f}")
        except Exception as e:
            print(f"Error training {name}: {e}")
            
    return results

if __name__ == "__main__":
    # Test on synthetic data
    np.random.seed(42)
    y_train = np.array([1, 1, 2, 2, 3, 3])
    y_test = np.array([1, 2, 3])
    
    # Distance matrices
    D_train = np.random.uniform(0.1, 1.0, (6, 6))
    np.fill_diagonal(D_train, 0)
    # Ensure symmetric
    D_train = (D_train + D_train.T) / 2
    
    D_test = np.random.uniform(0.1, 1.0, (3, 6))
    
    print("Testing classifiers on distance space...")
    res = train_and_evaluate_distance_space(D_train, y_train, D_test, y_test, verbose=True)
    assert "Distance kNN" in res
    assert "Kernel SVM" in res
    print("Classification test passed successfully!")
