import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

def fhe_poly_sign_simd(x_array, iterations=10):
    """
    SIMD version of FHE sign evaluation.
    Processes an entire array (representing a packed ciphertext) at once.
    """
    res = x_array
    for _ in range(iterations):
        res = 1.5 * res - 0.5 * (res**3)
    return (res + 1.0) / 2.0

class ScalableFHEServer:
    def __init__(self):
        # We store the database as a single matrix to simulate a packed CKKS ciphertext
        self.db_matrix = None
        self.chunk_ids = []
        
    def store_batch(self, chunk_ids, vectors):
        self.chunk_ids.extend(chunk_ids)
        if self.db_matrix is None:
            self.db_matrix = np.array(vectors)
        else:
            self.db_matrix = np.vstack((self.db_matrix, vectors))

    def search_simd(self, query_vector, sensitivity=1.2):
        """
        O(1) execution time in FHE due to SIMD packing.
        Calculates all dot products and masks simultaneously.
        """
        n = len(self.chunk_ids)
        if n == 0: return [], []

        # 1. SIMD Dot Product (One FHE operation across thousands of slots)
        # In FHE: Encrypted Query Vector * Encrypted DB Matrix
        scores = np.dot(self.db_matrix, query_vector)
        
        # 2. SIMD Statistical Thresholding
        inv_n = 1.0 / n
        mean_score = np.sum(scores) * inv_n
        
        # Calculate variance using vector math
        variance = np.sum((scores - mean_score)**2) * inv_n
        
        # 3. Create the evaluation threshold array
        # diff_array = scores - threshold
        threshold = mean_score + (sensitivity * variance)
        diff_array = (scores - threshold) * 0.5
        
        # 4. FHE Dampening to prevent explosion (Vectorized)
        dampened_array = diff_array - (diff_array**3) / 6.0
        
        # 5. SIMD Sign Evaluation (Generates the mask for all 100,000 chunks at once)
        encrypted_mask_array = fhe_poly_sign_simd(dampened_array, iterations=12)
        
        return self.chunk_ids, encrypted_mask_array

class ScalableSecureClient:
    def __init__(self):
        self.embedder = SentenceTransformer('BAAI/bge-small-en-v1.5')
        self.local_chunk_store = {}

    def l2_normalize(self, vector):
        norm = np.linalg.norm(vector)
        return vector if norm == 0 else vector / norm

    def ingest_large_document(self, doc_text, doc_name, server):
        """
        Chunks by larger paragraphs (simulated here by splitting on double newlines)
        to heavily reduce 'N' while maintaining deep context.
        """
        # Split into substantial paragraphs, not tiny sentences
        paragraphs = [p.strip() for p in doc_text.split('\n\n') if p.strip()]
        
        vectors = []
        ids = []
        
        for i, para in enumerate(paragraphs):
            chunk_id = f"{doc_name}_para_{i}"
            self.local_chunk_store[chunk_id] = para
            
            vec = self.l2_normalize(self.embedder.encode(para))
            vectors.append(vec)
            ids.append(chunk_id)
            
        # Send the entire batch to the server to simulate packed ingestion
        server.store_batch(ids, vectors)
