
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, hamming_loss, jaccard_score
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_val_score,
    cross_validate,
    train_test_split,
)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from skmultilearn.model_selection import iterative_train_test_split


def split_data(X, y, test_size=0.2, random_state=42, stratify=False):
    """
    Split data into train and test sets, with stratification for multiclass or iterative stratification for multilabel data.
    """
    if stratify:
        if y.ndim > 1 and y.shape[1] > 1:  # multilabel 
            # skmultilearn expects numpy arrays
            X_np = np.array(X)
            y_np = np.array(y)
            X_train, y_train, X_test, y_test = iterative_train_test_split(
                X_np, y_np, test_size
            )
           
            return X_train, X_test, y_train, y_test
        else:
            return train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
    else:
        return train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

    

def train_logreg(X_train, y_train, max_iter=1000, random_state=42, multilabel=True, **kwargs):

    """Train a logistic regression model for single-label or multilabel classification
    """
    base_clf = LogisticRegression(
        solver='lbfgs',
        max_iter=max_iter,
        random_state=random_state,
        **kwargs
    )
    
    if multilabel:
        clf = OneVsRestClassifier(base_clf)
    else:
        clf = base_clf
    
    clf.fit(X_train, y_train)
    return clf


def evaluate_model(clf, X_test, y_test, target_names=None, multilabel=True, show_report=True):
    """Evaluate the classifier for single-label or multilabel classification.
    
    
    Returns:
        Dictionary containing accuracy/scores, classification report, and predictions
    """
    preds = clf.predict(X_test)
    
    if multilabel:
        # Multilabel metrics
        accuracy = jaccard_score(y_test, preds, average='samples')
        hamming = hamming_loss(y_test, preds)
        
        if show_report:
            print(f"Jaccard Score (samples): {accuracy:.4f}")
            print(f"Hamming Loss: {hamming:.4f}")
            print("\nClassification Report:")
            print(classification_report(y_test, preds, target_names=target_names, zero_division=0))
        
        report = classification_report(y_test, preds, target_names=target_names, output_dict=True, zero_division=0)
        
        return {
            'jaccard_score': accuracy,
            'hamming_loss': hamming,
            'classification_report': report,
            'predictions': preds
        }
    else:
        # Single-label metrics
        accuracy = clf.score(X_test, y_test)
        report = classification_report(y_test, preds, target_names=target_names, output_dict=True)
        
        if show_report:
            print(f"Test accuracy: {accuracy:.4f}")
            print(classification_report(y_test, preds, target_names=target_names))
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'predictions': preds
        }




def crossval_scores(model, X, y, cv=5, scoring='accuracy', random_state=42, multilabel=True, show_results=True):
    """Compute k-fold cross-validation scores for the given model.
    
    Args:
        model: The model to evaluate (can be a base model that will be wrapped in OneVsRestClassifier)
        X, y: Features and labels
        cv: Number of cross-validation folds
        scoring: Scoring metric ('accuracy', 'f1_samples', 'jaccard_samples' for multilabel)
        random_state: Random state for reproducibility
        multilabel: If True, uses appropriate CV strategy for multilabel data
        show_results: If True, prints the results
    
    Returns:
        Array of cross-validation scores
    """
    # Wrap model in OneVsRestClassifier for multilabel classification
    if multilabel and not hasattr(model, 'estimators_'):
        cv_model = OneVsRestClassifier(model)
    else:
        cv_model = model
    
    if multilabel:
        # For multilabel, we use regular KFold instead of StratifiedKFold
        kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
        
        # Use appropriate scoring for multilabel
        if scoring == 'accuracy':
            scoring = 'jaccard_samples'  # More appropriate for multilabel
    else:
        kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    
    scores = cross_val_score(cv_model, X, y, cv=kf, scoring=scoring)

    if show_results:
        print(f"{cv}-fold cross-validation {scoring} scores:", scores)
        print(f"Mean {scoring}: {np.mean(scores):.4f}")
        print(f"Std {scoring}: {np.std(scores):.4f}")

    return scores


def multilabel_crossval_scores(model, X, y, cv=5, scoring_metrics=None, random_state=42, show_results=True):
    """Comprehensive cross-validation for multilabel classification with multiple metrics.
    
    Args:
        model: The base model to evaluate
        X, y: Features and labels
        cv: Number of cross-validation folds
        scoring_metrics: List of scoring metrics to evaluate
        random_state: Random state for reproducibility
        show_results: If True, prints the results
    
    Returns:
        Dictionary of cross-validation scores for each metric
    """
    from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
    
    if scoring_metrics is None:
        scoring_metrics = ['jaccard_samples', 'f1_samples', 'precision_samples', 'recall_samples']
    
    # Wrap model in OneVsRestClassifier
    cv_model = OneVsRestClassifier(model)
    
    # Use regular KFold for multilabel
    kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    
    # Create custom scorers with zero_division=0 to suppress warnings
    custom_scoring = {}
    if 'jaccard_samples' in scoring_metrics:
        custom_scoring['jaccard_samples'] = make_scorer(
            lambda y_true, y_pred: jaccard_score(y_true, y_pred, average='samples', zero_division=0)
        )
    if 'f1_samples' in scoring_metrics:
        custom_scoring['f1_samples'] = make_scorer(
            lambda y_true, y_pred: f1_score(y_true, y_pred, average='samples', zero_division=0)
        )
    if 'precision_samples' in scoring_metrics:
        custom_scoring['precision_samples'] = make_scorer(
            lambda y_true, y_pred: precision_score(y_true, y_pred, average='samples', zero_division=0)
        )
    if 'recall_samples' in scoring_metrics:
        custom_scoring['recall_samples'] = make_scorer(
            lambda y_true, y_pred: recall_score(y_true, y_pred, average='samples', zero_division=0)
        )
    
    # Perform cross-validation with custom scoring
    scores = cross_validate(cv_model, X, y, cv=kf, scoring=custom_scoring, return_train_score=False)
    
    if show_results:
        print(f"{cv}-fold multilabel cross-validation results:")
        for metric in scoring_metrics:
            if f'test_{metric}' in scores:
                test_scores = scores[f'test_{metric}']
                print(f"{metric}: {np.mean(test_scores):.4f} (+/- {np.std(test_scores) * 2:.4f})")
    
    return scores


def train_and_evaluate_multilabel(X, y, target_names=None, test_size=0.2, cv=5, 
                                 max_iter=1000, random_state=42, show_results=True):
    """Complete training and evaluation pipeline for multilabel classification.
    
    This function follows the proper ML protocol:
    1. Train-test split
    2. Cross-validation on training set
    3. Final evaluation on test set
    4. Probability outputs
    
    Args:
        X: Features
        y: Binary label matrix (samples x classes)
        target_names: List of class names
        test_size: Proportion of data for testing
        cv: Number of cross-validation folds
        max_iter: Maximum iterations for LogisticRegression
        random_state: Random state for reproducibility
        show_results: If True, prints detailed results
    
    Returns:
        Dictionary containing the trained model, predictions, probabilities, and evaluation metrics
    """
    # Step 1: Train-test split
    X_train, X_test, y_train, y_test = split_data(
        X, y, test_size=test_size, random_state=random_state, stratify=False
    )
    
    if show_results:
        print(f"Training set size: {X_train.shape[0]}")
        print(f"Test set size: {X_test.shape[0]}")
        print(f"Number of features: {X_train.shape[1]}")
        print(f"Number of classes: {y_train.shape[1]}")
        print("-" * 50)
    
    # Step 2: Cross-validation on training set
    base_model = LogisticRegression(max_iter=max_iter, random_state=random_state)
    
    if show_results:
        print("Performing cross-validation on training set...")
    
    cv_scores = multilabel_crossval_scores(
        base_model, X_train, y_train, cv=cv, 
        random_state=random_state, show_results=show_results
    )
    
    if show_results:
        print("-" * 50)
    
    # Step 3: Train final model on full training set
    clf = train_logreg(X_train, y_train, max_iter=max_iter, 
                     random_state=random_state, multilabel=True)
        
    # Step 5: Evaluate on test set
    if show_results:
        print("Final evaluation on test set:")
    
    eval_results = evaluate_model(
        clf, X_test, y_test, target_names=target_names, 
        multilabel=True, show_report=show_results
    )
    
    return {
        'model': clf,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'cv_scores': cv_scores,
        'predictions': eval_results['predictions'],
        'evaluation_metrics': eval_results,
        'target_names': target_names
    }



def train_random_forest(X_train, y_train, n_estimators=100, max_depth=None, random_state=42, multilabel=True, **kwargs):
    """
    Train a Random Forest for single-label or multilabel classification.
    
    Args:
        X_train: Training features
        y_train: Training labels (binary matrix for multilabel, 1D array for single-label)
        n_estimators: Number of trees in the forest
        max_depth: Maximum depth of the trees
        random_state: Random state for reproducibility
        multilabel: If True, uses OneVsRestClassifier for multilabel
        **kwargs: Additional arguments for RandomForestClassifier
    
    Returns:
        Trained classifier
    """
    base_clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,         # Use all available cores (scales well for large data)
        **kwargs
    )
    
    if multilabel:
        clf = OneVsRestClassifier(base_clf, n_jobs=-1)
    else:
        clf = base_clf
    
    clf.fit(X_train, y_train)
    return clf




def train_mlp(X_train, y_train, hidden_layer_sizes=(100,), max_iter=200, random_state=42, multilabel=True, **kwargs):
    """
    Train a Multi-layer Perceptron (MLP) for single-label or multilabel classification.
    
    Args:
        X_train: Training features
        y_train: Training labels (binary matrix for multilabel, 1D array for single-label)
        hidden_layer_sizes: Tuple defining the architecture of the MLP
        max_iter: Maximum number of iterations for training
        random_state: Random state for reproducibility
        multilabel: If True, uses OneVsRestClassifier for multilabel
        **kwargs: Additional arguments for MLPClassifier
    
    Returns:
        Trained classifier
    """
   
    
    base_clf = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        max_iter=max_iter,
        random_state=random_state,
        **kwargs
    )
    
    if multilabel:
        clf = OneVsRestClassifier(base_clf)
    else:
        clf = base_clf
    
    clf.fit(X_train, y_train)
    return clf