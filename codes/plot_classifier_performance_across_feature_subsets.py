import matplotlib.pyplot as plt
import numpy as np

feature_subsets = ['3 Features', '5 Features', '7 Features', 'Best Features']

markers = {'NORS': '^', 'SMOTE': '*', 'CBUTE': 'o'}  # Triangle, Star, Circle
metric_list = ['Recall', 'G-Mean', 'AUC']

def plot_comparison_clean(classifier, norm_method, data, filename):
    """
    Plot classifier performance across feature subsets.

    For a given classifier (e.g., CART, RF, GB, AB, KNN, Naive Bayes) and
    normalization method (MinMax or Z-Score), this function creates a
    3-panel figure:

      - Panel 1: Recall vs. feature subset
      - Panel 2: G-Mean vs. feature subset
      - Panel 3: AUC vs. feature subset

    In each panel, the three oversampling strategies (NORS, SMOTE, CBUTE)
    are plotted as separate lines for direct comparison.

    The figure is saved to `filename` and also displayed.
    """

def plot_comparison_clean(classifier, norm_method, data, filename):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True)
    
    for ax, metric in zip(axes, metric_list):
        for oversampling in ['NORS', 'SMOTE', 'CBUTE']:
            values = data[classifier][norm_method][oversampling][metric]
            ax.plot(
                feature_subsets,
                values,
                marker=markers[oversampling],
                label=oversampling,
                linestyle='-'
            )

        ax.set_title(metric)
        ax.set_ylim(-5, 105)
        ax.set_xlabel('Feature Subsets')
        ax.set_ylabel('Value (%)')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()

    fig.suptitle(f'{classifier} Performance Across Feature Subsets ({norm_method})')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(filename)
    plt.show()

# Generate plots for each classifier
for classifier in classifiers_data.keys():
    plot_comparison_clean(
        classifier, 'MinMax', classifiers_data,
        f'{classifier}_minmax_comparison_clean.png'
    )
    plot_comparison_clean(
        classifier, 'Z-Score', classifiers_data,
        f'{classifier}_zscore_comparison_clean.png'
    )
