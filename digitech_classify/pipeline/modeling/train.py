
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split


def split_data(X, y, test_size=0.2, random_state=42, stratify=True):

    """Split data into train and test sets, with optional stratification."""
    if stratify:
        return train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
    else:
        return train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
    

def train_model(X_train, y_train, max_iter=1000, random_state=42, **kwargs):

    """Train  multinomial logistic regression model."""

    clf = LogisticRegression(
        solver='lbfgs',
        max_iter=max_iter,
        random_state=random_state,
        **kwargs
    )
    clf.fit(X_train, y_train)
    return clf


def evaluate_model(clf, X_test, y_test, show_report=True):

    """Evaluate the classifier, print and return accuracy and classification report."""

    preds = clf.predict(X_test)
    accuracy = clf.score(X_test, y_test)
    report = classification_report(y_test, preds, output_dict=True)
    if show_report:
        print("Test accuracy:", accuracy)
        print(classification_report(y_test, preds))
    return accuracy, report, preds



def train_ensemble_model(X_train, y_train, X_val, y_val):
    """Train an ensemble of models for better performance."""
    models = {
        'lr': LogisticRegression(multi_class='multinomial', max_iter=2000, random_state=42),
        'rf': RandomForestClassifier(n_estimators=100, random_state=42),
        'gb': GradientBoostingClassifier(n_estimators=100, random_state=42),
    }
    
    # Train each model and collect predictions
    predictions = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions[name] = model.predict_proba(X_val)
    
    # Simple averaging ensemble
    ensemble_pred = np.mean(list(predictions.values()), axis=0)
    return models, ensemble_pred



def crossval_scores(model, X, y, cv=5, scoring='accuracy', random_state=42, show_results=True):
    """Compute k-fold cross-validation scores for the given model."""
    kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    scores = cross_val_score(model, X, y, cv=kf, scoring=scoring)

    if show_results:
        print(f"{cv}-fold cross-validation {scoring} scores:", scores)
        print(f"Mean {scoring}: {np.mean(scores):.4f}")

    return scores