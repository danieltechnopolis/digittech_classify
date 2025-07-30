# %%
import sys
sys.path.append("../../digitech_classify")

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from digitech_classify.pipeline.config import INTERIM_DATA_DIR
from digitech_classify.pipeline.data_engineering.features import (
         embed_keywords_transformer,
         embed_texts_transformer,
)

# %%
data_path = INTERIM_DATA_DIR / "company_lemma_full_description.csv"
company_df = pd.read_csv(data_path)


keywords_path = INTERIM_DATA_DIR / "keywords_lemma.csv"
keyword_df = pd.read_csv(keywords_path)

# %%
model = SentenceTransformer('all-MiniLM-L6-v2')


texts = company_df[['org_ID', 'features_lemma']].dropna().astype(str)
embeddings = embed_texts_transformer(texts['features_lemma'].tolist(), model)


# %%
print(f"Embedding shape: {embeddings.shape}")

# %%
keywords = keyword_df['keywords_lemma'].tolist()
keyword_vectors = embed_keywords_transformer(keywords, model)


# %%
np.savez(INTERIM_DATA_DIR / 'keywords_embeddings_all-MiniLM-L6-v2.npz',
         embeddings=keyword_vectors,
         keywords=keywords)

# %%
np.savez(INTERIM_DATA_DIR / 'company_embeddings_all-MiniLM-L6-v2.npz',
         embeddings=embeddings,
         org_ID=np.array(texts['org_ID'].values))


