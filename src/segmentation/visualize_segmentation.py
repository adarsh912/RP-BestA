import os
import numpy as np
import matplotlib.pyplot as plt
from src.datasets.loader import load_ucr_dataset
from src.segmentation.adaptive import segment_time_series

def compare_segmentations(dataset_name="GunPoint", index=0, save_dir="plots"):
    """
    Compares the 4 segmentation algorithms on a specific time series index from a dataset.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Load dataset
    X_train, y_train, _, _ = load_ucr_dataset(dataset_name)
    series = X_train[index]
    label = y_train[index]
    
    # Run segmentations
    bounds_fixed = segment_time_series(series, "fixed", param=15)
    bounds_var = segment_time_series(series, "variance", param=0.2)
    bounds_ent = segment_time_series(series, "entropy", param=2.0)
    bounds_cpd = segment_time_series(series, "cpd", param=2.5)
    
    # Plotting
    methods = [
        ("Fixed Window (W=15)", bounds_fixed, "blue"),
        ("Variance Window (th=0.2)", bounds_var, "orange"),
        ("Entropy Window (th=2.0)", bounds_ent, "green"),
        ("Change Point Detection (pen=2.5)", bounds_cpd, "red")
    ]
    
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    
    for i, (name, bounds, color) in enumerate(methods):
        ax = axes[i]
        ax.plot(series, color="gray", alpha=0.8, linewidth=1.5, label="Raw Series")
        
        # Plot boundaries
        for b in bounds:
            ax.axvline(x=b, color=color, linestyle="--", alpha=0.7, linewidth=1.2)
            
        # Draw segment indicators (shading alternating segments)
        for j in range(len(bounds) - 1):
            if j % 2 == 0:
                ax.axvspan(bounds[j], bounds[j+1], color=color, alpha=0.08)
                
        ax.set_title(f"{name} (Number of Segments: {len(bounds)-1})")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left")
        
    axes[-1].set_xlabel("Time Step")
    plt.suptitle(f"Segmentation Comparison on {dataset_name} (Sample Index: {index}, Class: {label})", fontsize=14, y=0.98)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f"{dataset_name}_segmentation_comparison.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved segmentation comparison plot to: {save_path}")

if __name__ == "__main__":
    compare_segmentations()
