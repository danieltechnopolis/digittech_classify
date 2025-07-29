#%%
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

from digitech_classify.pipeline.config import INTERIM_DATA_DIR, MODELS_DIR, PROCESSED_DATA_DIR

data_path = INTERIM_DATA_DIR / "unlabelled_data.csv"

model_name = "oneVrest_logreg_v1"
model_path = MODELS_DIR / f"{model_name}.joblib"
mlb_path = MODELS_DIR / f"{model_name}__mlb.joblib"


model = joblib.load(model_path)
mlb = joblib.load(mlb_path)

#%%

unlabelled_data = np.load(INTERIM_DATA_DIR / "company_embeddings_all-MiniLM-L6-v2.npz", allow_pickle=True)
X = unlabelled_data["embeddings"]    # shape (N, 384)
org_ids = unlabelled_data["org_ID"]      

#%%

batch_size = 5000
y_pred = []
y_proba = []  # Store probabilities for custom thresholding


for i in tqdm(range(0, X.shape[0], batch_size), desc="Predicting batches"):
    batch_X = X[i:i+batch_size]
    batch_proba = model.predict_proba(batch_X)
    batch_pred = (batch_proba >= 0.8).astype(int)  # set threshold 
    y_pred.append(batch_pred)

# Concatenate all batch predictions
y_pred = np.vstack(y_pred)

#%%
# Decode binary predictions to labels 
batch_size = 5000
y_pred_labels = []

for i in range(0, len(y_pred), batch_size):
    batch = y_pred[i:i+batch_size]
    batch_labels = mlb.inverse_transform(batch)
    y_pred_labels.extend(batch_labels)

# build df Package results
results_df = pd.DataFrame({
    'org_ID': org_ids,
    'predicted_labels': [list(labels) for labels in y_pred_labels],
})
#%%

# Filter out companies with empty predictions
results_df = results_df[results_df['predicted_labels'].apply(lambda x: len(x) > 0)]

# Save the results
results_df.to_csv(PROCESSED_DATA_DIR / "full_test_LogReg_predv1.csv", index=False)
print("Predictions saved.")

# %%
