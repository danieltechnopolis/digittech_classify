#%%
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from digitech_classify.pipeline.config import PROCESSED_DATA_DIR
from digitech_classify.pipeline.features import embed_texts_transformer

# %%
data_df = pd.read_csv(PROCESSED_DATA_DIR / "training_set_multilabel.csv")


model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = embed_texts_transformer(data_df['search_text'].astype(str).tolist(), model)


# %%
sector_label=np.array(data_df['all_labels'].values, dtype='str')

# %%
assert len(embeddings) == len(data_df['org_ID']) == len(sector_label)
assert embeddings.dtype == np.float32

# %%


np.savez(PROCESSED_DATA_DIR / 'training_set_multilabel_all-MiniLM-L6-v2.npz',
         embeddings=embeddings,
         org_ID=np.array(data_df['org_ID'].values, dtype='str'),
         sector_label=np.array(data_df['all_labels'].values))

print(f"Saved embeddings and labels to {PROCESSED_DATA_DIR }")



# %%
