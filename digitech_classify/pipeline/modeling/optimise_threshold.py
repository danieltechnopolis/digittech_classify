import numpy as np
from sklearn.metrics import precision_recall_curve

def optimize_multilabel_thresholds(y_true, y_prob, class_names=None, metric='f1', verbose=True):
    """
    Optimize classification thresholds for each class in multilabel classification.
        
    Returns:
    --------
    optimal_thresholds : array, shape (n_classes,)
        Optimal threshold for each class
    """
    
    n_classes = y_true.shape[1]
    optimal_thresholds = np.zeros(n_classes)
    
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(n_classes)]
    
    if verbose:
        print("Optimizing thresholds per class...")
        print("-" * 60)
    
    for i in range(n_classes):
        precision, recall, thresholds = precision_recall_curve(y_true[:, i], y_prob[:, i])
        
        if metric == 'f1':
            scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        elif metric == 'precision':
            scores = precision
        elif metric == 'recall':
            scores = recall
        else:
            raise ValueError("metric must be 'f1', 'precision', or 'recall'")
        
        # Find optimal threshold
        optimal_idx = np.argmax(scores)
        optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
        optimal_thresholds[i] = optimal_threshold
        
        if verbose:
            print(f"{class_names[i]:<40} | Threshold: {optimal_threshold:.3f} | {metric.upper()}: {scores[optimal_idx]:.3f}")
    
    return optimal_thresholds


def apply_thresholds(y_prob, thresholds):
    """
    Apply custom thresholds to probability predictions.
    
    Returns:
    --------
    y_pred : array, shape (n_samples, n_classes)
        Binary predictions using optimized thresholds
    """
    return (y_prob >= thresholds).astype(int)



def extract_probabilities_from_mlp(mlp_proba_output):
    """
    Extract probabilities from MLPClassifier predict_proba output.
    For multilabel, MLP returns probabilities directly in correct format.
    
    Parameters:
    -----------
    mlp_proba_output : array, shape (n_samples, n_classes)
        Output from MLPClassifier.predict_proba() or cross_val_predict
        
    Returns:
    --------
    y_prob : array, shape (n_samples, n_classes)
        Probabilities (already in correct format for multilabel MLP)
    """
    return mlp_proba_output