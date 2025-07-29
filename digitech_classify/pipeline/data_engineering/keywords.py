# %%
import sys
sys.path.append("../../digitech_classify")

from digitech_classify.pipeline.config import DATA_DIR, INTERIM_DATA_DIR
from digitech_classify.pipeline.data_engineering.features import batch_lemmatize, nlp_fast, nlp_full
import pandas as pd




# %%
kw_df = pd.read_excel(DATA_DIR / "keywords_combined_digital/Keywords_Combined.xlsx", sheet_name="Sheet1")
 

# %%
kw_df['Keyword'] = kw_df['Keyword'].astype(str).str.strip().str.lower()
kw_df = kw_df[kw_df['yes/no'] == 'yes']
kw_df = kw_df.drop(columns=['yes/no', 'Subcluster', 'Cluster'])
kw_df['Sector'] = kw_df['Sector'].astype(str).str.strip().str.lower()

# %%

kw_df['keywords_lemma'] = batch_lemmatize(
    kw_df['Keyword'].tolist(), model=nlp_fast, batch_size=5000
)

# %%
save_path = INTERIM_DATA_DIR / "keywords_lemma.csv"
kw_df.to_csv(save_path, index=False)


