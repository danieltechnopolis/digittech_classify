# %%
import sys
sys.path.append("../../digitech_classify")

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

from digitech_classify.pipeline.config import DATA_DIR, INTERIM_DATA_DIR
from digitech_classify.pipeline.data_engineering.features import (
    embed_keywords_transformer,
)

# %%
kw_df = pd.read_excel(DATA_DIR / "keywords_combined_digital/Keywords_Combined.xlsx", sheet_name="Sheet1")
 

# %%
kw_df['Keyword'] = kw_df['Keyword'].astype(str).str.strip().str.lower()
kw_df = kw_df[kw_df['yes/no'] == 'yes']
kw_df = kw_df.drop(columns=['yes/no', 'Subcluster', 'Cluster'])
kw_df['Sector'] = kw_df['Sector'].astype(str).str.strip().str.lower()

ms_df = kw_df[kw_df['Sector'] == 'microelectronics and semiconductors'].copy()

# %%
model = SentenceTransformer('all-MiniLM-L6-v2')
keywords_ms = ms_df['Keyword'].tolist()
keyword_vectors = embed_keywords_transformer(keywords_ms, model)



# %%
save_path = INTERIM_DATA_DIR / "keywords_lemma.csv"
kw_df.to_csv(save_path, index=False)



# %%
np.savez(INTERIM_DATA_DIR / "keywords_MS_embeddings.npz",
         embeddings=keyword_vectors,
         keywords_ms=keywords_ms)