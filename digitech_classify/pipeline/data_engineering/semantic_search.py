#%%
import sys
sys.path.append("../../digitech_classify")


import faiss
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

from digitech_classify.pipeline.config import DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR

# %%
# Load keyword vectors
keyword_data = np.load(INTERIM_DATA_DIR / "keywords_semantic_all-mpnet-base-v2.npz", allow_pickle=True)
keyword_vectors = keyword_data["embeddings"]
keywords = keyword_data["keywords"]

# Load company vectors
comp_data = np.load(PROCESSED_DATA_DIR / "company_embeddings_mpnet.npz", allow_pickle=True)
company_vectors = comp_data["embeddings"]
org_ids = comp_data["org_ID"]

print(company_vectors.dtype, company_vectors.shape, company_vectors.flags['C_CONTIGUOUS'])


# %% [Normalization]
faiss.normalize_L2(company_vectors)
faiss.normalize_L2(keyword_vectors)

# %% [FAISS PCA]
def run_faiss_pca(vectors, n_components=2):
    pca_matrix = faiss.PCAMatrix(vectors.shape[1], n_components)
    pca_matrix.train(vectors)
    assert pca_matrix.is_trained
    reduced = pca_matrix.apply_py(vectors)
    return reduced

company_pca_2d = run_faiss_pca(company_vectors, n_components=2)

# %% [FAISS K-means]
def run_faiss_kmeans(vectors, n_clusters=10, n_iter=20, seed=123):
    d = vectors.shape[1]
    kmeans = faiss.Kmeans(d, n_clusters, niter=n_iter, verbose=True, seed=seed)
    kmeans.train(vectors)
    cluster_assignments = kmeans.index.search(vectors, 1)[1].flatten()
    return cluster_assignments, kmeans.centroids

company_clusters, centroids = run_faiss_kmeans(company_vectors, n_clusters=10)



# %% [Plotting: PCA]
def plot_pca_scatter(pca_2d, color_labels, title='PCA Scatter'):
    plt.figure(figsize=(7,6))
    scatter = plt.scatter(pca_2d[:,0], pca_2d[:,1], c=color_labels, s=3, cmap='tab10', alpha=0.7)
    plt.colorbar(scatter, label="Cluster")
    plt.title(title)
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.tight_layout()
    plt.show()

plot_pca_scatter(company_pca_2d, company_clusters, title="FAISS PCA (2D) with K-means clusters")



# %% [Plotting: K-means centroids]
def plot_centroids_on_pca(pca_2d, cluster_labels, centroids, pca_matrix, title='PCA + Centroids'):
    centroids_2d = pca_matrix.apply_py(centroids)
    plt.figure(figsize=(7,6))
    plt.scatter(pca_2d[:,0], pca_2d[:,1], c=cluster_labels, s=3, cmap='tab10', alpha=0.4, label='Companies')
    plt.scatter(centroids_2d[:,0], centroids_2d[:,1], c='red', s=60, marker='X', label='Centroids')
    plt.title(title)
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Run centroid plot
pca_matrix = faiss.PCAMatrix(company_vectors.shape[1], 2)
pca_matrix.train(company_vectors)
plot_centroids_on_pca(company_pca_2d, company_clusters, centroids, pca_matrix)




# %%
keyword_clusters = run_faiss_kmeans(keyword_vectors, n_clusters=10)
keyword_pca_2d = run_faiss_pca(keyword_vectors, n_components=2)

keyword_clusters = np.array(keyword_clusters)

# %%
def plot_keyword_clusters(pca_2d, clusters, keywords, sample_size=1500):
    idx = np.random.choice(len(pca_2d), size=min(sample_size, len(pca_2d)), replace=False)
    plt.figure(figsize=(8, 7))
    scatter = plt.scatter(pca_2d[idx, 0], pca_2d[idx, 1], c=clusters[idx], s=18, cmap='tab10', alpha=0.8)
    plt.title("Keyword Embeddings (FAISS PCA + K-means Clusters)")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.colorbar(scatter, label="Cluster")
    # Optionally annotate a few keywords:
    for i in idx[:20]:
        plt.annotate(str(keywords[i]), (pca_2d[i,0], pca_2d[i,1]), fontsize=8, alpha=0.7)
    plt.tight_layout()
    plt.show()

plot_keyword_clusters(keyword_pca_2d, keyword_clusters, keywords)
# %%
