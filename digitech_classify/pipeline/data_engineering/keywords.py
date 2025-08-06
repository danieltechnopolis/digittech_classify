# %%
import sys
sys.path.append("../../digitech_classify")

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from digitech_classify.pipeline.config import DATA_DIR, INTERIM_DATA_DIR
from digitech_classify.pipeline.data_engineering.features import (
    embed_keywords_transformer,
)

# %%
kw_df = pd.read_excel(DATA_DIR / "keywords_combined_digital/Keywords_Combined_v2.xlsx", sheet_name="Sheet1")
 

# %%
kw_df['Keyword'] = kw_df['Keyword'].astype(str).str.strip().str.lower()
kw_df = kw_df[kw_df['yes/no'] == 'yes']
kw_df = kw_df.drop(columns=['yes/no'])
kw_df['sector'] = kw_df['sector'].astype(str).str.strip().str.lower()

# %%
model = SentenceTransformer('all-mpnet-base-v2')
keywords = kw_df['semantic_search'].tolist()
keyword_vectors = embed_keywords_transformer(keywords, model)


# %%
np.savez(INTERIM_DATA_DIR / "keywords_semantic_all-mpnet-base-v2.npz",
         embeddings=keyword_vectors,
         keywords=keywords,
         sectors=kw_df['sector'].tolist()
       )


# %%
