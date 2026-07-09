import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.datasets.loader import load_ucr_dataset

def calculate_dataset_stats(X, y):
    """
    Computes basic statistics of the time series dataset.
    """
    stats = {}
    stats['n_samples'] = len(X)
    stats['series_length'] = X.shape[1]
    
    # Class distribution
    unique_classes, counts = np.unique(y, return_counts=True)
    stats['n_classes'] = len(unique_classes)
    stats['classes'] = list(unique_classes)
    stats['class_counts'] = dict(zip(unique_classes, counts))
    
    # Global metrics
    stats['min_val'] = float(np.min(X))
    stats['max_val'] = float(np.max(X))
    stats['mean_val'] = float(np.mean(X))
    stats['std_val'] = float(np.std(X))
    
    # Checking for missing values
    stats['n_missing'] = int(np.isnan(X).sum())
    
    return stats

def plot_dataset_samples(X, y, dataset_name, num_samples_per_class=3, save_dir="plots"):
    """
    Plots a few time series samples grouped by class.
    """
    os.makedirs(save_dir, exist_ok=True)
    unique_classes = np.unique(y)
    
    plt.figure(figsize=(12, 4 * len(unique_classes)))
    
    for i, label in enumerate(unique_classes):
        plt.subplot(len(unique_classes), 1, i + 1)
        class_indices = np.where(y == label)[0]
        selected_indices = np.random.choice(class_indices, min(num_samples_per_class, len(class_indices)), replace=False)
        
        for idx in selected_indices:
            plt.plot(X[idx], alpha=0.7, label=f"Sample {idx}")
        
        # Plot mean series for this class
        mean_series = np.mean(X[class_indices], axis=0)
        plt.plot(mean_series, color='black', linewidth=2.5, linestyle='--', label="Class Mean")
        
        plt.title(f"{dataset_name} - Class {label}")
        plt.xlabel("Time Step")
        plt.ylabel("Value")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{dataset_name}_samples.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved dataset visualization to: {save_path}")

def print_stats_table(stats_dict, dataset_name):
    """
    Prints a nicely formatted text table of the dataset statistics.
    """
    print("=" * 50)
    print(f"Dataset Statistics: {dataset_name}")
    print("=" * 50)
    print(f"Number of Samples  : {stats_dict['n_samples']}")
    print(f"Time Series Length : {stats_dict['series_length']}")
    print(f"Number of Classes  : {stats_dict['n_classes']}")
    print(f"Classes            : {stats_dict['classes']}")
    print("-" * 50)
    print("Class Distribution:")
    for cls, count in stats_dict['class_counts'].items():
        print(f"  Class {cls}: {count} ({count/stats_dict['n_samples']*100:.1f}%)")
    print("-" * 50)
    print(f"Global Range       : [{stats_dict['min_val']:.3f}, {stats_dict['max_val']:.3f}]")
    print(f"Global Mean (Std)  : {stats_dict['mean_val']:.3f} ({stats_dict['std_val']:.3f})")
    print(f"Missing Values     : {stats_dict['n_missing']}")
    print("=" * 50)

if __name__ == "__main__":
    # Test script using GunPoint
    try:
        X_train, y_train, X_test, y_test = load_ucr_dataset("GunPoint")
        stats = calculate_dataset_stats(X_train, y_train)
        print_stats_table(stats, "GunPoint (Train)")
        plot_dataset_samples(X_train, y_train, "GunPoint")
    except Exception as e:
        print(f"Stats calculation failed: {e}")
