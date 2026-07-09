import os
import numpy as np
import matplotlib.pyplot as plt
from src.datasets.loader import load_ucr_dataset
from src.segmentation.adaptive import segment_time_series
from src.granulation.lfig import granularize_time_series

def plot_lfig_granulation(dataset_name="GunPoint", index=0, method="cpd", param=2.5, z=1.96, save_dir="plots"):
    """
    Plots the LFIG trend lines and fuzzy envelopes on a sample time series.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Load dataset
    X_train, y_train, _, _ = load_ucr_dataset(dataset_name)
    series = X_train[index]
    label = y_train[index]
    
    # Segment series
    boundaries = segment_time_series(series, method=method, param=param)
    
    # Construct granules
    granules = granularize_time_series(series, boundaries, z=z)
    
    # Plotting
    plt.figure(figsize=(12, 6))
    
    # Plot raw signal
    plt.plot(series, color="dimgray", alpha=0.6, linewidth=1.5, label="Raw Time Series")
    
    # Plot trend lines and bounds for each segment
    for j, g in enumerate(granules):
        x_range = np.arange(g["start"], g["end"])
        
        # Plot Trend Line
        plt.plot(x_range, g["trend_line"], color="crimson", linewidth=2.5, 
                 label="LFIG Trend" if j == 0 else "")
        
        # Plot Envelopes (bounds)
        plt.plot(x_range, g["lower_envelope"], color="navy", linestyle=":", alpha=0.8,
                 label="Fuzzy Envelopes" if j == 0 else "")
        plt.plot(x_range, g["upper_envelope"], color="navy", linestyle=":", alpha=0.8)
        
        # Shade the uncertainty region
        plt.fill_between(x_range, g["lower_envelope"], g["upper_envelope"], 
                         color="royalblue", alpha=0.15, label="Fuzzy Granule Spread" if j == 0 else "")
        
        # Plot vertical boundary lines
        plt.axvline(x=g["start"], color="black", linestyle="--", alpha=0.4, linewidth=1.0)
    
    # Last boundary line
    plt.axvline(x=series.shape[0] - 1, color="black", linestyle="--", alpha=0.4, linewidth=1.0)
    
    plt.title(f"LFIG Construction on {dataset_name} (Sample Index: {index}, Class: {label})\n"
              f"Segmentation: {method.upper()} (param={param}), Coverage: z={z}", fontsize=13)
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.grid(True, alpha=0.25)
    plt.legend(loc="upper left")
    
    save_path = os.path.join(save_dir, f"{dataset_name}_lfig_granulation.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved LFIG granulation plot to: {save_path}")

if __name__ == "__main__":
    plot_lfig_granulation()
