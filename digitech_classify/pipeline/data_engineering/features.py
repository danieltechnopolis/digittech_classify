import faiss
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import spacy
from spacy.matcher import Matcher
from tqdm import tqdm


def get_spacy_models():
    nlp_full = spacy.load("en_core_web_sm")
    nlp_fast = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    return nlp_full, nlp_fast


nlp_full, nlp_fast = get_spacy_models()



def extract_noun_phrases(texts, model=nlp_full, batch_size=5000, n_process=1):
    results = []
    for doc in tqdm(model.pipe(texts, batch_size=batch_size, n_process=n_process), total=len(texts), desc="Noun-phrase extraction"):
        key_phrases = set(chunk.text for chunk in doc.noun_chunks)
        results.append(' '.join(key_phrases))
    return results



def batch_lemmatize(texts, model=nlp_fast, batch_size=5000):
    processed = []
    with tqdm(total=len(texts), desc="Processing texts") as pbar:
        for doc in model.pipe(texts, batch_size=batch_size):
            tokens = [token.lemma_.lower() for token in doc 
                      if not token.is_stop and not token.is_punct 
                      and not token.is_space and len(token.text) > 1]
            processed.append(' '.join(tokens))
            pbar.update(1)
    return processed


def get_embedding_model():

    return spacy.load('en_core_web_lg', disable=['ner', 'parser'])


def embed_texts(texts, model=None, batch_size=1000, n_process=1):
    if model is None:
        model = get_embedding_model()
    vectors = []
    for doc in tqdm(model.pipe(texts, batch_size=batch_size, n_process=n_process), total=len(texts), desc="Embedding texts"):
        vectors.append(doc.vector)
    return np.vstack(vectors)


def embed_keywords_transformer(keywords, model):
    return model.encode(keywords, normalize_embeddings=True)


def embed_texts_transformer(texts, model, batch_size=1000):
    # Always work with a list, not a Series avoid iloc/index bugs
    if isinstance(texts, pd.Series):
        texts = texts.tolist()
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]  
        emb = model.encode(batch, show_progress_bar=True, normalize_embeddings=True)
        embeddings.append(emb)
    return np.vstack(embeddings)


def load_vectors(npz_path, vec_key, id_key=None):
    data = np.load(npz_path, allow_pickle=True)
    vectors = np.ascontiguousarray(data[vec_key], dtype=np.float32)
    ids = None
    if id_key is not None:
        ids = np.array(data[id_key], copy=True)
    return vectors, ids


def build_faiss_index(vectors):
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors) # type: ignore
    return index


def search_top_k(index, query_vectors, top_k=5, batch_size=10000):
    num_queries = query_vectors.shape[0]
    all_D, all_I = [], []
    for i in range(0, num_queries, batch_size):
        batch = np.ascontiguousarray(query_vectors[i:i+batch_size], dtype=np.float32)
        # No need to normalize again (already normalized)
        D, I = index.search(batch, top_k)
        all_D.append(D)
        all_I.append(I)
    return np.vstack(all_D), np.vstack(all_I)



def build_keyword_tag_df(D, indices, org_ids, keyword_texts, sim_threshold=0.8):
    mask = D >= sim_threshold
    company_idx, kw_rank = np.where(mask)
    if len(company_idx) == 0:
        print("No matches found above threshold")
        return pd.DataFrame(columns=['org_ID', 'keyword', 'similarity'])
    return pd.DataFrame({
        'org_ID': np.array(org_ids)[company_idx],
        'keyword': np.array(keyword_texts)[indices[company_idx, kw_rank]],
        'similarity': D[company_idx, kw_rank]
    })



def create_keyword_features(texts, keywords_by_sector, nlp, top_k=50):
    """Create keyword-based features for each text."""
    # Build sector-specific matchers
    matchers = {}
    for sector, keywords in keywords_by_sector.items():
        matcher = Matcher(nlp.vocab)
        patterns = [nlp.make_doc(kw) for kw in keywords]
        matcher.add(f"{sector}_KEYWORDS", patterns)
        matchers[sector] = matcher
    
    # Extract features
    features = []
    for text in tqdm(texts):
        doc = nlp(text)
        feature_vec = []
        
        # Count matches per sector
        for sector, matcher in matchers.items():
            matches = matcher(doc)
            feature_vec.append(len(matches))  # Raw count
            feature_vec.append(len(matches) / len(doc))  

        features.append(feature_vec)
    
    return np.array(features)



def combine_embeddings_with_features(embeddings, keyword_features, alpha=0.1):
    """Combine embeddings with keyword features."""
    # Normalize keyword features
    scaler = StandardScaler()
    keyword_features_norm = scaler.fit_transform(keyword_features)
    
    # Weight and concatenate
    weighted_features = keyword_features_norm * alpha
    return np.hstack([embeddings, weighted_features])



