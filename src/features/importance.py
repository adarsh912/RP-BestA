import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from src.datasets.loader import load_ucr_dataset
from src.features.extractor import extract_granular_dataset

def analyze_feature_importance(dataset_name="GunPoint", save_dir="plots"):
    """
    Performs feature importance analysis by converting granulated sequences 
    into global aggregated statistics and training a Random Forest.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Load and granulate data
    X_train, y_train, _, _ = load_ucr_dataset(dataset_name)
    print(f"Granularizing {len(X_train)} samples from {dataset_name}...")
    gran_X = extract_granular_dataset(X_train, method="cpd", param=2.5)
    
    # 2. Aggregate features across granules for each sample
    # We compute the mean and std of each of the 10 features across the granules.
    agg_features = []
    for seq in gran_X:
        # seq has shape (n_granules, 10)
        mean_feats = np.mean(seq, axis=0)
        std_feats = np.std(seq, axis=0)
        agg_features.append(np.concatenate([mean_feats, std_feats]))
        
    agg_features = np.array(agg_features) # shape: (n_samples, 20)
    
    # Feature Names
    base_names = [
        "Lower Bound", "Upper Bound", "Trend Slope", "Shannon Entropy", 
        "Variance", "Volatility", "Curvature", "Intercept", "Energy", "Skewness"
    ]
    feature_names = [f"{name} (Mean)" for name in base_names] + [f"{name} (Std)" for name in base_names]
    
    # 3. Fit Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(agg_features, y_train)
    
    importances = rf.feature_importances_
    
    # Create DataFrame for plotting
    df_imp = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)
    
    # Print results
    print("=" * 60)
    print(f"Feature Importance on {dataset_name} (Random Forest)")
    print("=" * 60)
    for idx, row in df_imp.head(10).iterrows():
        print(f"  {row['Feature']:25s}: {row['Importance']:.4f}")
    print("=" * 60)
    
    # 4. Plot importances
    plt.figure(figsize=(10, 8))
    sns.barplot(x="Importance", y="Feature", data=df_imp, palette="viridis", hue="Feature", legend=False)
    plt.title(f"Random Forest Feature Importance on {dataset_name} (Aggregated Granules)")
    plt.xlabel("Gini Importance")
    plt.ylabel("Aggregated Feature")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f"{dataset_name}_feature_importance.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved feature importance plot to: {save_path}")

if __name__ == "__main__":
    analyze_feature_importance()
