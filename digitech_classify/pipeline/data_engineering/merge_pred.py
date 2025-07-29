import pandas as pd

from digitech_classify.pipeline.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

# Load the predictions
predictions_df = pd.read_csv(PROCESSED_DATA_DIR / "full_test_logReg_predv1.csv")


companies_df = pd.read_csv(RAW_DATA_DIR / "cb_net0_companies_concat.csv",
        usecols=['org_ID', 'organisation_name', 'short_description', 'description'],
        dtype={'org_ID': 'string', 'organisation_name': 'string', 'short_description': 'string', 'description': 'string'},
        index_col=False)  



# Merge predictions with company data
merged_df = predictions_df.merge(companies_df, on='org_ID', how='left')



# Save the merged DataFrame
merged_df.to_csv(PROCESSED_DATA_DIR / "merged_predictions_oneVrest_logreg_v1.csv", index=False)


