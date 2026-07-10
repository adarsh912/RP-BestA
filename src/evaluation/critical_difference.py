"""
Demšar-style critical difference diagrams for multi-dataset classifier comparison.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare, rankdata


# Nemenyi q-alpha values for alpha=0.05 (k = number of classifiers)
_NEMENYI_Q_ALPHA = {
    2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728,
    6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164,
}


def compute_average_ranks(accuracy_matrix):
    """
    Given a DataFrame where rows=datasets, columns=classifiers, values=accuracies,
    computes the average rank of each classifier across datasets.
    Lower rank = better (rank 1 = highest accuracy).

    Returns: Series of average ranks indexed by classifier name.
    """
    # Rank each row (dataset) in descending accuracy order (best = rank 1)
    ranks = accuracy_matrix.rank(axis=1, ascending=False, method='average')
    return ranks.mean(axis=0)


def friedman_test(accuracy_matrix):
    """
    Runs Friedman chi-square test on accuracy_matrix.
    Returns: (statistic, p_value)
    """
    # friedmanchisquare expects each group as a separate array
    groups = [accuracy_matrix[col].values for col in accuracy_matrix.columns]
    stat, p_val = friedmanchisquare(*groups)
    return stat, p_val


def nemenyi_post_hoc(accuracy_matrix, alpha=0.05):
    """
    Runs Nemenyi post-hoc test after a significant Friedman test.
    Returns critical difference value CD.

    CD = q_alpha * sqrt(k*(k+1) / (6*N))
    """
    k = len(accuracy_matrix.columns)  # number of classifiers
    N = len(accuracy_matrix)           # number of datasets

    if k not in _NEMENYI_Q_ALPHA:
        raise ValueError(f"Nemenyi q-alpha not available for k={k}. Supported: 2-10.")

    q_alpha = _NEMENYI_Q_ALPHA[k]
    cd = q_alpha * np.sqrt(k * (k + 1) / (6 * N))
    return cd


def plot_critical_difference_diagram(avg_ranks, cd, classifier_names, output_path='plots/cd_diagram.png'):
    """
    Plots a horizontal critical difference diagram.

    - X-axis: ranks (1 to k)
    - Each classifier is plotted at its average rank
    - Classifiers whose rank difference < CD are connected with a thick bar (clique)
    - Alternates names above and below to avoid overlap

    Saves to output_path.
    """
    import os
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    k = len(classifier_names)
    sorted_indices = np.argsort(avg_ranks)
    sorted_names = [classifier_names[i] for i in sorted_indices]
    sorted_ranks = [avg_ranks[i] for i in sorted_indices]

    fig, ax = plt.subplots(figsize=(max(8, k * 1.5), 4), dpi=150)

    # Draw the rank axis
    ax.set_xlim(0.5, k + 0.5)
    ax.set_ylim(-1.5, 1.5)
    ax.hlines(0, 0.5, k + 0.5, color='black', linewidth=1)

    # Tick marks for integer ranks
    for r in range(1, k + 1):
        ax.vlines(r, -0.05, 0.05, color='black', linewidth=1)
        ax.text(r, -0.15, str(r), ha='center', va='top', fontsize=9)

    # Plot classifier names alternating above/below
    for i, (name, rank) in enumerate(zip(sorted_names, sorted_ranks)):
        if i % 2 == 0:
            y_text, y_tick = 0.6, 0.05
            va = 'bottom'
        else:
            y_text, y_tick = -0.6, -0.05
            va = 'top'

        ax.plot(rank, 0, 'ko', markersize=6)
        ax.vlines(rank, 0, y_tick + (0.3 if i % 2 == 0 else -0.3), color='gray', linewidth=0.8)
        ax.text(rank, y_text, name, ha='center', va=va, fontsize=8, fontweight='bold')

    # Draw cliques (groups of classifiers not significantly different)
    # Find all cliques: maximal groups where all pairwise rank diffs < CD
    cliques = []
    for i in range(len(sorted_ranks)):
        for j in range(i + 1, len(sorted_ranks)):
            if sorted_ranks[j] - sorted_ranks[i] < cd:
                # Check if this pair extends an existing clique
                merged = False
                for clique in cliques:
                    if i in clique and sorted_ranks[j] - sorted_ranks[min(clique)] < cd:
                        clique.add(j)
                        merged = True
                        break
                if not merged:
                    cliques.append({i, j})

    # Draw clique bars
    y_bar = 1.1
    for clique in cliques:
        if len(clique) < 2:
            continue
        left = sorted_ranks[min(clique)]
        right = sorted_ranks[max(clique)]
        ax.hlines(y_bar, left, right, color='red', linewidth=3, alpha=0.7)
        y_bar += 0.15

    # CD annotation
    ax.text(0.5, 1.35, f'CD = {cd:.3f}', ha='left', va='bottom', fontsize=9, style='italic',
            transform=ax.transData)

    ax.set_xlabel('Average Rank', fontsize=10)
    ax.set_title('Critical Difference Diagram', fontsize=12, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_yaxis().set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"CD diagram saved to {output_path}")


def run_cd_analysis(accuracy_matrix, output_path='plots/cd_diagram.png', alpha=0.05):
    """
    Full pipeline: compute ranks → Friedman test → Nemenyi post-hoc → plot CD diagram.
    Returns dict with avg_ranks, friedman_stat, friedman_p, cd_value.
    """
    avg_ranks = compute_average_ranks(accuracy_matrix)
    stat, p_val = friedman_test(accuracy_matrix)

    print(f"Friedman test: statistic={stat:.4f}, p-value={p_val:.6f}")

    cd = None
    if p_val < alpha:
        cd = nemenyi_post_hoc(accuracy_matrix, alpha=alpha)
        print(f"Nemenyi CD (alpha={alpha}): {cd:.4f}")

        classifier_names = list(accuracy_matrix.columns)
        avg_ranks_list = [avg_ranks[name] for name in classifier_names]
        plot_critical_difference_diagram(avg_ranks_list, cd, classifier_names, output_path=output_path)
    else:
        print("Friedman test not significant — no post-hoc analysis needed.")

    return {
        'avg_ranks': avg_ranks,
        'friedman_stat': stat,
        'friedman_p': p_val,
        'cd_value': cd,
    }
