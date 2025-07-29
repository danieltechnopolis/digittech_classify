import pandas as pd

from digitech_classify.pipeline.config import PROCESSED_DATA_DIR
from digitech_classify.pipeline.data_engineering.multilabel import (
    aggregate_by_sector,
    extract_sectors,
)

glass_AI = pd.read_excel(PROCESSED_DATA_DIR / 'glassAI_crunchbase_overlap.xlsx')


sector_mapping = {
    'Cloud to Edge to IoT': 'cloud-edge-iot',
    'Data Analytics Technologies': 'data analytics',
    'Artificial Intelligence': 'artificial intelligence',
    'Blockchain Technologies': 'blockchain',
    'Photonics Technologies': 'photonics',
    'Quantum Technologies': 'quantum technologies',
    'Robotics Technologies': 'robotics',
    'Advanced Digital Communications and Connectivity': 'advanced digital communications and connectivity',
    'High Performance Computing': 'high performance computing',
    'Next Generation Internet and Extended Reality': 'next generation internet and extended reality',
    'Microelectronics, High Frequency Chips and Semiconductors': 'microelectronics, high frequency chips and semiconductors'
    
}


glass_AI['digital_sector_glassAI'] = (
    glass_AI['digital_sector_glassAI']
    .replace(sector_mapping)
    .str.strip()
    .str.lower()
)

glass_AI['digital_sector_glassAI'] = glass_AI['digital_sector_glassAI'].replace(
    'advanced and high performance computing', 'high performance computing'
)

glass_AI['digital_sector_glassAI'] = glass_AI['digital_sector_glassAI'].replace("blockchain, distributed ledger and digital identity technologies"
    , 'blockchain'
)

glass_AI['digital_sector_glassAI'] = glass_AI['digital_sector_glassAI'].replace("quantum"
    , 'quantum technologies'
)

glass_AI['digital_sector_glassAI'] = glass_AI['digital_sector_glassAI'].replace("microelectronics, high frequency chips and semiconductors"
    , 'microelectronics and semiconductors'
)

df_sectors_labelled = extract_sectors(df=glass_AI, sector_col1='digital_sector', sector_col2='digital_sector_glassAI')

df_sectors_labelled_agg = aggregate_by_sector(df_sectors_labelled, org_col='org_ID', text_col='search_text')


save_path = PROCESSED_DATA_DIR / 'training_data_multilabel_agg.xlsx'