#%%

import pandas as pd

from digitech_classify.pipeline.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

p_df = pd.read_csv(PROCESSED_DATA_DIR / 'final_dataset.csv')


full_df = pd.read_csv(RAW_DATA_DIR / "cb_net0_companies_concat.csv",
        usecols=['org_ID', 'organisation_name', 'short_description', 'data_source'],
        dtype={'org_ID': 'string', 'organisation_name': 'string', 'short_description': 'string', 'data_source': 'string'},
        index_col=False)  



#%%

negatives_df = full_df[~full_df['org_ID'].isin(p_df['org_ID'])].copy()

# %%
sampled_negatives = negatives_df.sample(n=22000, random_state=42).copy()
# %%
sampled_negatives['labels'] = [set() for _ in range(len(sampled_negatives))]

# %%
sampled_negatives.to_csv(RAW_DATA_DIR / 'sampled_negatives.csv', index=False)
# %%
