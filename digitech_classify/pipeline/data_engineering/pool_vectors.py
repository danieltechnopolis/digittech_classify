#%%

import sys
sys.path.append("../../digitech_classify")

import faiss
import numpy as np
import pandas as pd

from digitech_classify.pipeline.config import DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR
from digitech_classify.pipeline.data_engineering.features import (
    apply_sector_threshold,
    build_faiss_index,
    build_keyword_tag_df,
    pool_embeddings,
    search_top_k,
)

#%%
kw_data = np.load(INTERIM_DATA_DIR / "keywords_semantic_all-mpnet-base-v2.npz", allow_pickle=True)
keyword_vectors = kw_data["embeddings"]
keyword_texts = kw_data["keywords"] 


comp_data = np.load(PROCESSED_DATA_DIR / "company_embeddings_mpnet.npz", allow_pickle=True)
company_vectors = comp_data["embeddings"]
org_ids = comp_data["org_ID"]  


kw_df = pd.read_excel(DATA_DIR / "keywords_combined_digital/Keywords_Combined_v2.xlsx", sheet_name="Sheet1")


kw_df['Keyword'] = kw_df['Keyword'].astype(str).str.strip().str.lower()
kw_df = kw_df[kw_df['yes/no'] == 'yes']
kw_df = kw_df.drop(columns=['yes/no'])
kw_df['sector'] = kw_df['sector'].astype(str).str.strip().str.lower()

#%%
kw_map = dict(zip(kw_df['semantic search'], kw_df['sector']))
sectors = [kw_map.get(k, 'other') for k in keyword_texts]
unique_sectors = sorted(set(sectors))
sector_vectors, sector_names = [], []
#%%

sector_vectors, sector_names = pool_embeddings(keyword_vectors, sectors)
faiss.normalize_L2(sector_vectors) 

sector_index = build_faiss_index(sector_vectors)

print(f"Sector index: {sector_index.ntotal} vectors, {sector_index.d} dimensions")
#%%

faiss.normalize_L2(company_vectors)
D, I = search_top_k(sector_index, company_vectors, top_k=5, batch_size=10000)


tagged_sector_df = build_keyword_tag_df(D, I, org_ids, sector_names, sim_threshold=0.4)





# %%
company_path = INTERIM_DATA_DIR / "cleaned_companies_text.csv"
descriptions_df = pd.read_csv(company_path, usecols=['org_ID','organisation_name', 'search_text'])  
# %%
merged_df = tagged_sector_df.merge(
    descriptions_df[['org_ID', 'organisation_name', 'search_text']],
    on='org_ID',
    how='left'
)
# %%
save_path= PROCESSED_DATA_DIR / "company_tagged_pooled_similarity.csv"
merged_df.to_csv(save_path, index=False)
print(f"Saved tagged sector data to {save_path}")

# %%
