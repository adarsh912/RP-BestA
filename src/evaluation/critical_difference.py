import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare

_NEMENYI_Q_ALPHA = {
    2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728,
    6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164
}

def compute_average_ranks(accuracy_matrix):
    ranks = accuracy_matrix.rank(axis=1, ascending=False, method='average')
    return ranks.mean(axis=0)

def friedman_test(accuracy_matrix):
    groups = [accuracy_matrix[col].values for col in accuracy_matrix.columns]
    return friedmanchisquare(*groups)

def nemenyi_post_hoc(accuracy_matrix, alpha=0.05):
    k = len(accuracy_matrix.columns)
    N = len(accuracy_matrix)
    q_alpha = _NEMENYI_Q_ALPHA[k]
    return q_alpha * np.sqrt(k * (k + 1) / (6 * N))

def plot_critical_difference_diagram(avg_ranks, cd, classifier_names, output_path='plots/cd_diagram.png'):
    k = len(classifier_names)
    sorted_indices = np.argsort(avg_ranks)
    sorted_names = [classifier_names[i] for i in sorted_indices]
    sorted_ranks = [avg_ranks[i] for i in sorted_indices]
    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    ax.set_xlim(0.5, k + 0.5)
    ax.set_ylim(-1.5, 1.5)
    ax.hlines(0, 0.5, k + 0.5, color='black', linewidth=1)
    for r in range(1, k + 1):
        ax.vlines(r, -0.05, 0.05, color='black', linewidth=1)
        ax.text(r, -0.15, str(r), ha='center', va='top', fontsize=9)
    for i, (name, rank) in enumerate(zip(sorted_names, sorted_ranks)):
        y_text = 0.6 if i % 2 == 0 else -0.6
        y_tick = 0.05 if i % 2 == 0 else -0.05
        ax.plot(rank, 0, 'ko', markersize=6)
        ax.vlines(rank, 0, y_tick + (0.3 if i % 2 == 0 else -0.3), color='gray', linewidth=0.8)
        ax.text(rank, y_text, name, ha='center', va='bottom' if i % 2 == 0 else 'top', fontsize=8, fontweight='bold')
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
