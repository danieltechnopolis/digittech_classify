#%%
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

from digitech_classify.pipeline.config import INTERIM_DATA_DIR, MODELS_DIR, PROCESSED_DATA_DIR

MODEL = "mlp_multilabel_mpnet"
RUN = 2


model_path = MODELS_DIR / f"{MODEL}.joblib"
scaler_path = MODELS_DIR / f"{MODEL}__scaler.joblib"
test_path = INTERIM_DATA_DIR / "company_embeddings_all-MiniLM-L6-v2.npz"
mlb_path = MODELS_DIR / f"{MODEL}__mlb.joblib"
output_path = PROCESSED_DATA_DIR / f"{MODEL}_full_test{RUN}_pred.csv"



test_data = np.load(test_path, allow_pickle=True)
X = test_data["embeddings"]    # shape (N, 384)
org_ids = test_data["org_ID"]    



#%%

model = joblib.load(model_path)
mlb = joblib.load(mlb_path)
scaler = joblib.load(scaler_path)

test_data = np.load(test_path, allow_pickle=True)
X = test_data["embeddings"]    # shape (N, 384)
org_ids = test_data["org_ID"]  

X = scaler.transform(X)

print(f"Model loaded from {model_path}")
print(f"MultiLabelBinarizer loaded from {mlb_path}")  
print("Test data loaded:")
print(f"Embeddings shape: {X.shape}")
print(f"org_IDs shape: {org_ids.shape}")  

#%%

BATCH_SIZE = 5000
THRESHOLD = 0.8

y_proba = []  

# Predict in batches to avoid memory issues
def predict_in_batches(model, X, batch_size=5000, threshold=THRESHOLD):
    y_pred = []
    for i in tqdm(range(0, X.shape[0], batch_size), desc="Predicting batches"):
        batch_X = X[i:i+batch_size]
        # For scikit-learn multilabel MLP, model.predict_proba returns a (N, C) array
        batch_proba = model.predict_proba(batch_X)
        batch_pred = (batch_proba >= threshold).astype(int)
        y_pred.append(batch_pred)
    return np.vstack(y_pred)

# Concatenate all batch predictions
y_pred = predict_in_batches(model, X, batch_size=BATCH_SIZE, threshold=THRESHOLD)

#%%
# Decode binary predictions to labels 
def multilabel_decode(pred, mlb, batch_size=5000):
    labels = []
    for i in range(0, len(pred), batch_size):
        batch = pred[i:i+batch_size]
        batch_labels = mlb.inverse_transform(batch)
        labels.extend(batch_labels)
    return labels


y_pred_labels = multilabel_decode(y_pred, mlb, batch_size=BATCH_SIZE)

# build df Package results
results_df = pd.DataFrame({
    'org_ID': org_ids,
    'predicted_labels': [list(labels) for labels in y_pred_labels],
})
#%%

# Filter out companies with empty predictions
results_df = results_df[results_df['predicted_labels'].apply(lambda x: len(x) > 0)]



#%%
# Save the results
results_df.to_csv(output_path, index=False)
print("Predictions saved.")

# %%
