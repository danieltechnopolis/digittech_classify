#%%
import ast

import joblib
import numpy as np
from scipy.stats import loguniform, uniform
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV, cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from skmultilearn.model_selection import IterativeStratification

from digitech_classify.pipeline.config import INTERIM_DATA_DIR, MODELS_DIR, PROCESSED_DATA_DIR
from digitech_classify.pipeline.modeling.optimise_threshold import (
    apply_thresholds,
    optimize_multilabel_thresholds,
)
from digitech_classify.pipeline.modeling.train import split_data

#%%
data = np.load(PROCESSED_DATA_DIR / "training_set_multilabel_all-mpnet-base-v2.npz", allow_pickle=True)
model_name = "mlp_multilabel_mpnet"

print("Keys in loaded npz file:", data.files)
for key in data.files:
    print(f"{key}: shape={data[key].shape}, dtype={data[key].dtype}")

#%%
X = data["embeddings"]             
y = data["sector_label"]   # Labels as string representations of lists
org_ids = data["org_ID"]  

print("Embeddings shape:", X.shape)
print("Labels shape:", y.shape)
print("First few labels:", y[:5])
print("org_ids dtype:", org_ids.dtype)
print("y dtype:", y.dtype)

# Parse labels for all samples (still as lists)
parsed_labels = [ast.literal_eval(label_str) for label_str in y]

# Now perform the train/test split, using multilabel binarized y for stratification
mlb = MultiLabelBinarizer()
y_encoded = mlb.fit_transform(parsed_labels)

# 80/20 hold-out split
X_train, X_test, y_train_encoded, y_test_encoded = split_data(X, y_encoded, stratify=True)
print(f"Total training samples: {X_train.shape[0]}")
print(f"Total test samples: {X_test.shape[0]}")

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

parsed_labels = np.array(parsed_labels, dtype=object)
parsed_labels_train = parsed_labels[:X_train.shape[0]]
parsed_labels_test = parsed_labels[X_train.shape[0]:]

print(f"Train set: {np.sum(y_train_encoded.sum(axis=1) == 0)} samples with all-zero labels")
print(f"Test set: {np.sum(y_test_encoded.sum(axis=1) == 0)} samples with all-zero labels")



#%%
param_dist = {
    'hidden_layer_sizes': [
        (1024, 512), (1536, 768), (2048, 1024),
        (1024, 256), (768, 384), (1536, 512),
        
        (1024, 512, 256), (1536, 768, 384), (2048, 1024, 512),
        (1024, 768, 256), (1536, 1024, 512), (768, 512, 256),
    
        (1536, 1024, 512, 256), (2048, 1024, 512, 256),
        (1024, 768, 512, 256), (1536, 768, 384, 128)
    ],
    'activation': ['relu', 'tanh'],  
    'alpha': loguniform(1e-6, 1e-1),  
    'learning_rate_init': loguniform(5e-5, 5e-3), 
    'max_iter': [3000, 4000, 5000], 
    'learning_rate': ['constant', 'adaptive'],  # Added adaptive option
    'batch_size': ['auto', 200, 500],  
    'early_stopping': [True],
    'validation_fraction': uniform(0.1, 0.15),  
    'n_iter_no_change': [15, 20, 25],  
    
    # Solver
    'solver': ['adam']  
}


#%%
mlp = MLPClassifier(random_state=42)

stratifier = IterativeStratification(n_splits=5, order=2)
cv_splits = list(stratifier.split(X_train, y_train_encoded))

# Randomized search with multilabel-aware cross-validation (refer to skmultilearn documentation)
random_search = RandomizedSearchCV(
    estimator=mlp,
    param_distributions=param_dist,
    n_iter=50,
    scoring='f1_macro',
    cv=cv_splits,
    verbose=2,
    n_jobs=4,
    random_state=42
)


random_search.fit(X_train, y_train_encoded)
print("Best params:", random_search.best_params_)
print("Best score:", random_search.best_score_)


best_mlp = random_search.best_estimator_

#%%

y_pred_cv = cross_val_predict(
    best_mlp, 
    X_train, 
    y_train_encoded, 
    cv=IterativeStratification(n_splits=5, order=1), 
    method='predict',
    n_jobs=2
)

    

print("Cross-validated evaluation report:")
print(classification_report(y_train_encoded, y_pred_cv, target_names=mlb.classes_, zero_division=0))


y_test_pred = best_mlp.predict(X_test) # type: ignore
print("Hold-out test set performance:")
print(classification_report(y_test_encoded, y_test_pred, target_names=mlb.classes_, zero_division=0))


# save the best model and multilabel encoding 
model_name = model_name
model_path = MODELS_DIR / f"{model_name}.joblib"
joblib.dump(best_mlp, model_path)
print(f"Best MLP model saved: {model_path}")


# Save the MultiLabelBinarizer
mlb_path = MODELS_DIR / f"{model_name}__mlb.joblib"
joblib.dump(mlb, mlb_path)
print(f"MultiLabelBinarizer saved: {mlb_path}")

scaler_path = MODELS_DIR / f"{model_name}__scaler.joblib"
joblib.dump(scaler, scaler_path)
print(f"Scaler saved to: {scaler_path}")


metadata = {
    'model_type': 'MLPClassifier_RandomizedSearchCV_Best',
    'best_params': random_search.best_params_,
    'best_cv_score': random_search.best_score_,
    'feature_dim': X_train.shape[1],
    'n_classes': len(mlb.classes_),
    'class_names': list(mlb.classes_),
    'training_samples': X_train.shape[0],
    'test_samples': X_test.shape[0] if 'X_test' in locals() else None,
    'cv_folds': 5,
    'scoring_metric': 'f1_samples',
    'n_iter': random_search.n_iter # type: ignore
}

metadata_path = MODELS_DIR / f"{model_name}_metadata.joblib"
joblib.dump(metadata, metadata_path)
print(f"Model metadata saved to: {metadata_path}")


print("\nBest cross-validated F1 (samples):", random_search.best_score_)
print("Best hyperparameters:", random_search.best_params_)




#%% Get probability predictions for threshold optimization
y_prob_cv = cross_val_predict(
    best_mlp, X_train, y_train_encoded, 
    cv=IterativeStratification(n_splits=5, order=1), 
    method='predict_proba', n_jobs=2
)

# Optimize thresholds
optimal_thresholds = optimize_multilabel_thresholds(
    y_train_encoded, y_prob_cv, 
    class_names=mlb.classes_, verbose=True
)

# Apply to test set
y_test_prob = best_mlp.predict_proba(X_test) # type: ignore
y_test_pred_optimized = apply_thresholds(y_test_prob, optimal_thresholds)

# Evaluate with optimized thresholds
print("\nOptimal threshold results")
print(classification_report(y_test_encoded, y_test_pred_optimized, target_names=mlb.classes_, zero_division=0))

# Save thresholds with your model
metadata['optimal_thresholds'] = optimal_thresholds.tolist()
# %%
