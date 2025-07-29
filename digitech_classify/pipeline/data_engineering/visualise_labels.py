from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_pred_label(y_bin, mlb, top_n=None):
    """
    Plot frequency of individual labels in multilabel binarized matrix.
    
    Args:
        y_bin (np.ndarray): Binarized label matrix.
        mlb (MultiLabelBinarizer): Fitted binarizer with classes_.
        top_n (int, optional): Plot only the top N most frequent labels.
    """
    label_counts = y_bin.sum(axis=0)
    label_series = pd.Series(label_counts, index=mlb.classes_)
    label_series = label_series.sort_values(ascending=False)

    if top_n:
        label_series = label_series.head(top_n)

    plt.figure(figsize=(12, 6))
    sns.barplot(x=label_series.values, y=label_series.index, orient='h')
    plt.title('Label Frequency')
    plt.xlabel('Count')
    plt.ylabel('Label')
    plt.tight_layout()
    plt.show()





def plot_label_combinations(y_bin, mlb, top_n=20):
    """
    Plot frequency of most common label combinations.
    
    Args:
        y_bin (np.ndarray): Binarized label matrix.
        mlb (MultiLabelBinarizer): Fitted binarizer with classes_.
        top_n (int): Number of most frequent combinations to show.
    """
    label_combos = []
    for row in y_bin:
        labels = tuple(sorted([label for label, present in zip(mlb.classes_, row) if present]))
        label_combos.append(labels)

    combo_counts = Counter(label_combos)
    top_combos = combo_counts.most_common(top_n)

    combo_labels = [' + '.join(c) if c else 'NO LABELS' for c, _ in top_combos]
    combo_freqs = [f for _, f in top_combos]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=combo_freqs, y=combo_labels, orient='h')
    plt.title('Top Label Combinations')
    plt.xlabel('Frequency')
    plt.ylabel('Label Combination')
    plt.tight_layout()
    plt.show()






