# src/semantic_search.py

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# In-memory storage
srs_chunks = []
test_cases = []
test_case_embeddings = []

def embed_text(text):
    """Embed a single string."""
    embedding = embedder.encode([text])[0]
    return embedding.astype(np.float32)

def build_index(srs_chunk_list, test_case_list):
    """
    Store all inputs and precompute embeddings for test cases.
    """
    global srs_chunks, test_cases, test_case_embeddings

    srs_chunks = srs_chunk_list
    test_cases = test_case_list

    # Embed only test case descriptions
    test_case_embeddings = []
    for tc in test_cases:
        combined = tc.get("requirement", "") + " " + tc.get("test_case", "")  # Adjust keys as needed
        embedding = embed_text(combined)
        test_case_embeddings.append(embedding)

def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def match_srs_to_testcases(similarity_threshold=0.75):
    """
    For each SRS chunk, check if it is already covered by a test case.
    Return unmatched SRS chunks that need test generation.
    """
    unmatched = []

    for chunk in srs_chunks:
        chunk_vec = embed_text(chunk)
        matched = False

        for tc_vec in test_case_embeddings:
            sim = cosine_similarity(chunk_vec, tc_vec)
            if sim >= similarity_threshold:
                matched = True
                break

        if not matched:
            unmatched.append(chunk)

    return unmatched

def search_unmatched_requirements(query_text, top_k=3):
    """
    Searches the most relevant unmatched SRS chunks.
    Used during /query route for custom test case generation.
    """
    unmatched = match_srs_to_testcases()
    if not unmatched:
        return []

    query_vec = embed_text(query_text).reshape(1, -1)
    chunk_vectors = [embed_text(c) for c in unmatched]

    index = faiss.IndexFlatL2(query_vec.shape[1])
    index.add(np.array(chunk_vectors))
    distances, indices = index.search(query_vec, top_k)

    results = [unmatched[i] for i in indices[0]]
    return results
