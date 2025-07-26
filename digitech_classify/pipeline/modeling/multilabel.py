import pandas as pd


def normalise_label(val):
    """Simple normalization: lowercases and strips."""
    if pd.isnull(val):
        return None
    return str(val).strip().lower()


def aggregate_labels(row):
    """Aggregate normalized labels from both columns, discarding None."""
    labels = set()
    s1 = normalise_label(row['sector'])
    s2 = normalise_label(row['digital_sector_glassAI'])
    if s1:
        labels.add(s1)
    if s2:      
        labels.add(s2)
    return list(labels)


def prepare_multilabel_df(df,
                          sector_col='sector',
                          glassai_col='digital_sector_glassAI',
                          groupby_cols=['org_ID', 'organisation_name', 'search_text']):
    """
    Returns a grouped DataFrame with a multi-label list per company,
    using simple normalization and no mapping.
    """
    df = df.copy()
    df['all_labels'] = df.apply(aggregate_labels, axis=1)
    grouped_df = (
        df.groupby(groupby_cols, as_index=False)
          .agg({'all_labels': lambda x: list(set().union(*x))})
    )
    return grouped_df