import pandas as pd


def normalise_label(val):
    """Simple normalization: lowercases and strips."""
    if pd.isnull(val):
        return None
    return str(val).strip().lower()



def extract_sectors(df, sector_col1='digital_sector', sector_col2='digital_sector_glassAI'):
    def combine(row):
        sectors = set()
        s1 = row[sector_col1]
        s2 = row[sector_col2]
        if pd.notnull(s1):
            sectors.add(str(s1).strip().lower())
        if pd.notnull(s2):
            sectors.add(str(s2).strip().lower())
        sectors = set([s for s in sectors if s and s != 'nan'])
        return sectors
    df['sectors_combined'] = df.apply(combine, axis=1)
    return df



def aggregate_labels(row):
    """Aggregate normalized labels from both columns, discarding None."""
    labels = set()
    s1 = normalise_label(row['col1'])
    s2 = normalise_label(row['col2'])
    if s1:
        labels.add(s1)
    if s2:      
        labels.add(s2)
    return list(labels)



def aggregate_by_sector(df, org_col='org_ID', name_col='organisation_name', text_col='search_text'):
    agg = (
        df.groupby(org_col, as_index=False)
        .agg({
            name_col: 'first',
            'sectors_combined': lambda x: set().union(*x),
            text_col: 'first'
        })
        .rename(columns={'sectors_combined': 'labels'})
    )
    return agg