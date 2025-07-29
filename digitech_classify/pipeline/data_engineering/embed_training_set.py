#%%
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from digitech_classify.pipeline.config import PROCESSED_DATA_DIR
from digitech_classify.pipeline.data_engineering.features import embed_texts_transformer

# %%
data_df = pd.read_csv(PROCESSED_DATA_DIR / "multilabelled_training_set_enriched_v2.csv")


model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = embed_texts_transformer(data_df['search_text'].astype(str).tolist(), model)


# %%
sector_label=np.array(data_df['all_sectors'].values, dtype=object)

# %%
assert len(embeddings) == len(data_df['org_ID']) == len(sector_label)
assert embeddings.dtype == np.float32

# %%


np.savez(PROCESSED_DATA_DIR / 'training_setv2_multilabel_all-MiniLM-L6-v2.npz',
         embeddings=embeddings,
         org_ID=np.array(data_df['org_ID'].values, dtype='str'),
         sector_label=np.array(data_df['all_sectors'].values, dtype=object)
       )

print(f"Saved embeddings and labels to {PROCESSED_DATA_DIR }")



# %%
