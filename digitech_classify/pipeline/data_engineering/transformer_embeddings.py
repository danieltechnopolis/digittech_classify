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
data_path = INTERIM_DATA_DIR / "cleaned_companies_text.csv"
company_df = pd.read_csv(data_path)
company_df = company_df.drop(columns=['processed_text', 'data_source'], errors='ignore')
company_df = company_df.dropna(subset=['search_text'])
#%%
keywords_path = INTERIM_DATA_DIR / "keywords_lemma.csv"
keyword_df = pd.read_csv(keywords_path)

# %%
model = SentenceTransformer('all-mpnet-base-v2')


texts = company_df[['org_ID', 'search_text']].dropna().astype(str)
embeddings = embed_texts_transformer(texts['search_text'].tolist(), model, batch_size=32)

reader = pd.read_csv('company.csv', chunksize=10000)  # Adjust chunksize to fit RAM
for idx, chunk in enumerate(reader):
    chunk = chunk.dropna(subset=['search_text'])
    texts = chunk['search_text'].astype(str).tolist()
    ids = chunk['org_ID'].tolist()
    embs = embed_texts_transformer(texts, model)
    np.save(f'embeddings_part_{idx}.npy', embs)
    pd.Series(ids).to_csv(f'ids_part_{idx}.csv', index=False)
    print(f"Saved chunk {idx}")
3

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


