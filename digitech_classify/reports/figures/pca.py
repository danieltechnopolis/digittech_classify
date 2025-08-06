
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

from digitech_classify.pipeline.config import DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR

kw_df = pd.read_excel(DATA_DIR / "keywords_combined_digital/Keywords_Combined_v2.xlsx", sheet_name="Sheet1")
kw_data = np.load(INTERIM_DATA_DIR / "keywords_semantic_all-mpnet-base-v2.npz", allow_pickle=True)
keyword_vectors = kw_data["embeddings"]
keyword_texts = kw_data["keywords"]

print(keyword_vectors.dtype, keyword_vectors.shape, keyword_vectors.flags['C_CONTIGUOUS'])      



kw_df['Keyword'] = kw_df['Keyword'].astype(str).str.strip().str.lower()
kw_df = kw_df[kw_df['yes/no'] == 'yes']
kw_df = kw_df.drop(columns=['yes/no'])
kw_df['sector'] = kw_df['sector'].astype(str).str.strip().str.lower()


# Create mapping from keyword_texts to sectors using kw_df
# First create a mapping from semantic search to sector
semantic_to_sector = dict(zip(kw_df['semantic search'], kw_df['sector']))

# Map each keyword_text to its sector, defaulting to 'other' if not found
sectors = [semantic_to_sector.get(kw, 'other') for kw in keyword_texts]

# Apply PCA to keyword vectors for 3D visualization
pca = PCA(n_components=3)
keyword_pca = pca.fit_transform(keyword_vectors)



# Plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')


sector_list = list(set(sectors))

palette = plt.cm.get_cmap('tab20')(np.linspace(0, 1, len(sector_list)))
sector_color = {sector: palette[i] for i, sector in enumerate(sector_list)}
colors = [sector_color[s] for s in sectors]

ax.scatter(keyword_pca[:, 0], keyword_pca[:, 1], keyword_pca[:, 2], c=colors, alpha=0.7)

# Create legend
legend_elements = [plt.Line2D([0], [0], marker='o', color='w',  # type: ignore
                             markerfacecolor=sector_color[sector], markersize=8, 
                             label=sector) for sector in sector_list]
ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1))

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3') # type: ignore 
ax.set_title('Keyword Embeddings (all-mpnet-base-v2) colored by sector)')
plt.tight_layout()
plt.show()