import numpy as np
import matplotlib.pyplot as plt
import os

def calculate_dataset_stats(X_train, y_train):
    stats = {
        'n_samples': len(X_train),
        'length': X_train.shape[1],
        'n_classes': len(np.unique(y_train)),
        'distribution': dict(zip(*np.unique(y_train, return_counts=True))),
        'min': float(np.min(X_train)),
        'max': float(np.max(X_train)),
        'mean': float(np.mean(X_train)),
        'std': float(np.std(X_train)),
    }
    return stats
