import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# ==========================================
# 1. FHE MATHEMATICAL OPERATIONS (UNTRUSTED)
# ==========================================
# These functions strictly use addition, subtraction, multiplication, 
# and integer exponentiation to mimic homomorphic limits.

def fhe_squash(x):
    """
    FHE-compatible way to keep values from exploding.
    This polynomial maps values toward the [-1, 1] range without branching.
    """
    # S(x) = 1.5x - 0.5x^3
    return 1.5 * x - 0.5 * (x**3)

def fhe_step_function(diff):
    """
    Approximates 'is diff > 0'.
    We scale the difference by a constant factor (gain) to sharpen the slope,
    then use iterative polynomials to force it toward -1 or 1.
    """
    # Instead of clipping, we scale by 0.5 to ensure stability
    # since dot product differences are in range [-2, 2].
    x = diff * 0.5

    # Iterative sign evaluation (the core of FHE comparison)
    for _ in range(8):
        x = 1.5 * x - 0.5 * (x**3)

    # Map from [-1, 1] to [0, 1]
    return (x + 1.0) / 2.0

def fhe_dot_product(vec_a, vec_b):
    """Encrypted Inner Product (Similarity). Both vectors must be L2 normalized."""
    return np.sum(vec_a * vec_b)

def fhe_sign_eval(x, iterations=7):
    """
    Iterative polynomial approximation of the sign function.
    FHE cannot do `if x > 0`. We use the Goldschmidt/Newton method:
    x_{k+1} = 1.5 * x_k - 0.5 * x_k^3
    Input x MUST be bounded in [-1, 1].
    """
    for _ in range(iterations):
        x = 1.5 * x - 0.5 * (x**3)
    return x

def fhe_poly_sign(x, iterations=10):
    """Core FHE comparison: Maps x > 0 to ~1 and x < 0 to ~0."""
    # We use a scaled version of the sign function
    # x_{i+1} = 1.5*x_i - 0.5*x_i^3
    res = x
    for _ in range(iterations):
        res = 1.5 * res - 0.5 * (res**3)
    return (res + 1.0) / 2.0

def fhe_step_function(diff):
    """
    Approximates `1 if diff > 0 else 0`.
    Diff is the difference between two dot products (range [-2, 2]).
    We scale it to [-1, 1] to keep the polynomial stable.
    """
    x = diff / 2.0
    sign_x = fhe_sign_eval(x, iterations=7)
    # Shift from [-1, 1] to [0, 1]
    return (sign_x + 1.0) / 2.0
"""
def fhe_compute_top_k_mask(scores, k, n):
    # Computes a binary mask for Top-K using ONLY mathematics.
    # No sorted(), max(), or > operators.
    
    mask = []
    for i in range(n):
        # 1. Compare score[i] against all other scores mathematically
        count_less_than_i = 0
        for j in range(n):
            diff = scores[i] - scores[j]
            # If scores[i] > scores[j], this adds ~1. If equal, adds 0.5.
            count_less_than_i += fhe_step_function(diff)
        
        # 2. Thresholding: To be in Top-K, count_less_than_i must be >= (N - K + 0.4)
        # We use 0.4 instead of 0.5 to mathematically push exact boundary ties to 1.
        threshold = n - k + 0.4
        t_i = count_less_than_i - threshold
        
        # 3. Scale t_i to [-1, 1] bounds. Max possible t_i is ~K, min is ~-N. 
        # Dividing by N safely bounds it.
        t_scaled = t_i / n
        
        # 4. Evaluate sign to create the final [0, 1] mask for this document
        sign_val = fhe_sign_eval(t_scaled, iterations=7)
        is_top_k = (sign_val + 1.0) / 2.0
        mask.append(is_top_k)
        
    return mask
"""
def fhe_compute_top_k_mask(scores, k, n):
    """
    Top-K Mask using ONLY +, -, *
    No clip, no if, no max.
    """
    mask = []

    # Theoretically, the max 'count' is N.
    # To keep the polynomial from exploding, we must scale by 1/N.
    inv_n = 1.0 / n

    for i in range(n):
        count_less_than_i = 0
        for j in range(n):
            # Mathematical comparison: result is ~1 if scores[i] > scores[j]
            diff = scores[i] - scores[j]
            count_less_than_i += fhe_step_function(diff)

        # We want to check if (count_less_than_i) > (n - k - 0.5)
        # Shift the value so that the 'Top-K boundary' sits at Zero.
        t_i = count_less_than_i - (n - k - 0.5)

        # Scale into the [-1, 1] convergence range using inv_n
        t_scaled = t_i * inv_n

        # Final Sign Evaluation to produce the binary mask
        res = t_scaled
        for _ in range(10):
            res = 1.5 * res - 0.5 * (res**3)

        # Map to [0, 1] range
        is_top_k = (res + 1.0) / 2.0
        mask.append(is_top_k)

    return mask

def fhe_statistical_retrieval(scores, sensitivity=1.5):
    """
    FHE-Compatible O(N) Retrieval.
    Instead of Top-K (Sorting), we use Statistical Outlier Detection.
    Keeps chunks where Score > Mean + (sensitivity * Variance).
    """
    n = len(scores)
    if n == 0: return []

    # 1. Compute Mean (Sum / N) - Purely additive/multiplicative
    inv_n = 1.0 / n
    mean_score = sum(scores) * inv_n

    # 2. Compute Variance (Sum((x - mean)^2) / N)
    # This identifies the 'spread' of relevance
    variance = sum([(s - mean_score)**2 for s in scores]) * inv_n

    # 3. Apply Polynomial Threshold
    # We want to keep scores that are significantly better than average.
    # Logic: Score - (Mean + Sensitivity * Variance) > 0
    mask = []
    for s in scores:
        # Distance from the 'threshold'
        # We divide by a normalization constant (e.g. 2.0)
        # to stay within the [-1, 1] polynomial convergence range.
        diff = (s - (mean_score + sensitivity * variance)) * 0.5

        # Stability: Ensure we don't explode if diff > 1 (FHE-style squashing)
        # S(x) = x - (x^3)/6 is a safe FHE-compatible way to dampen high values
        dampened_diff = diff - (diff**3) / 6.0

        mask_val = fhe_poly_sign(dampened_diff, iterations=12)
        mask.append(mask_val)

    return mask

# ==========================================
# 2. UNTRUSTED SERVER (VECTOR DB & SEARCH)
# ==========================================

class FHEServer:
    def __init__(self):
        # In reality, this holds Ciphertexts
        self.encrypted_db_vectors = []
        self.db_chunk_ids = []
    
    def store(self, chunk_id, encrypted_vector):
        self.db_chunk_ids.append(chunk_id)
        self.encrypted_db_vectors.append(encrypted_vector)

    def search(self, encrypted_query_vector, k=2):
        """
        Executes search over encrypted data. 
        Returns an encrypted continuous mask, NOT the chunks themselves.
        """
        n = len(self.encrypted_db_vectors)
        
        # Step 1: Encrypted Dot Products
        scores = []
        for db_vec in self.encrypted_db_vectors:
            score = fhe_dot_product(encrypted_query_vector, db_vec)
            scores.append(score)
            
        # Step 2: Encrypted Top-K Masking via Polynomials
        # Returns an array of floats that are mathematically forced close to 1.0 or 0.0
        
        # encrypted_mask = fhe_compute_top_k_mask(scores, k, n)
        encrypted_mask = fhe_statistical_retrieval(scores, sensitivity=1.2)

        return self.db_chunk_ids, encrypted_mask


# ==========================================
# 3. SECURE CLIENT (TRUST ZONE)
# ==========================================

class SecureClient:
    def __init__(self):
        # Embedding Model (BGE)
        self.embedder = SentenceTransformer('BAAI/bge-small-en-v1.5')
        # SLM (Granite) - Using a smaller pipeline for demonstration
        self.llm = pipeline('text-generation', model='ibm-granite/granite-3.0-2b-instruct', max_new_tokens=100)
        
        # Plaintext database map
        self.local_chunk_store = {}

    def l2_normalize(self, vector):
        norm = np.linalg.norm(vector)
        if norm == 0: 
            return vector
        return vector / norm
    
    def ingest_document_advanced(self, doc_text, server):
        # FIX: Sliding Window Chunking
        # We group 3 sentences at a time with a 2-sentence overlap
        sentences = [s.strip() + "." for s in doc_text.split('.') if s.strip()]

        for i in range(len(sentences)):
            # Window of 3 sentences to maintain context (Mr X -> He)
            window = " ".join(sentences[max(0, i-2) : i+1])

            chunk_id = f"chunk_{i}"
            self.local_chunk_store[chunk_id] = sentences[i] # Store actual sentence

            # Embed the window (Contextual Embedding)
            vector = self.embedder.encode(window)
            normalized_vector = self.l2_normalize(vector)
            server.store(chunk_id, normalized_vector)

    def ingest_document(self, doc_text, server: FHEServer):
        # 1. Chunking (Simulated simple split for demo)
        # chunks = [doc_text[i:i+256] for i in range(0, len(doc_text), 256)]
        chunks = [s.strip() + "." for s in doc_text.split('.') if s.strip()]

        for i, chunk in enumerate(chunks):
            chunk_id = f"chunk_{i}"
            self.local_chunk_store[chunk_id] = chunk
            
            # 2. Embed & Normalize
            vector = self.embedder.encode(chunk)
            normalized_vector = self.l2_normalize(vector)
            
            # 3. Encrypt (Mock: pass as numpy array) & Send to Server
            server.store(chunk_id, normalized_vector)

    def ask_question(self, query, server: FHEServer, top_k=2):
        # 1. Embed, Normalize, and Encrypt query
        q_vec = self.embedder.encode(query)
        norm_q_vec = self.l2_normalize(q_vec)
        
        # 2. Send to Server to get the encrypted mask
        chunk_ids, encrypted_mask = server.search(norm_q_vec, k=top_k)
        
        # 3. Client Decrypts the Mask and rounds the noisy FHE floats
        decrypted_mask = [round(m) for m in encrypted_mask]
        
        # 4. Client reconstructs the context based on the valid mask
        retrieved_contexts = []
        for i, mask_val in enumerate(decrypted_mask):
            if mask_val == 1.0:
                chunk_id = chunk_ids[i]
                retrieved_contexts.append(self.local_chunk_store[chunk_id])
                
        context_str = "\n".join(retrieved_contexts)
        
        # 5. Plaintext LLM Generation
        #prompt = f"Context: {context_str}\n\nQuestion: {query}\nAnswer:"
        messages = [
            {"role": "system", "content": "You are a precise assistant. Answer the user's question using ONLY the provided context. Keep it brief and answer only based on context."},
            {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {query}"}
        ]

        #response = self.llm(prompt)[0]['generated_text']
        response = self.llm(messages, max_new_tokens=50, return_full_text=False)[0]['generated_text']
        
        return response, decrypted_mask, context_str

# ==========================================
# 4. EXECUTION
# ==========================================
if __name__ == "__main__":
    # Initialize zones
    untrusted_fhe_server = FHEServer()
    secure_client = SecureClient()

    # Sample Data
    doc = (
        "Homomorphic encryption allows computation on ciphertexts. "
        "The BGE model is great for creating semantic embeddings. "
        "Python is a versatile programming language. "
        "Granite is a scalable language model developed by IBM. "
        "Quantum computing uses qubits to perform operations."
        "Homomorphic encryption allows computation on ciphertexts. "
        "The BGE model is great for creating semantic embeddings. "
        "Python is a versatile programming language. "
        "Granite is a scalable language model developed by IBM. "
        "Quantum computing uses qubits to perform operations. "
        "Vector databases store embeddings for similarity search. "
        "L2 normalization is required for dot product similarity."
        "Divyesh Saglani is a researcher at TCS Research. He develops algorithms in FHE. Divyesh has an experience of more than 5 years. He also have several papers and patents, which are more than 20."
        "Imtiyaz develops algorithms in MPC."
    )
    
    print("Ingesting document...")
    secure_client.ingest_document_advanced(doc, untrusted_fhe_server)
    
    #query = "What is the BGE model used for?"
    query = "Who developed Granite?"
    print(f"\nQuerying: {query}")
    
    response, decrypted_mask, context = secure_client.ask_question(query, untrusted_fhe_server, top_k=2)
    
    print(f"\nServer Returned Mask (Decrypted): {decrypted_mask}")
    print(f"Retrieved Context: {context}")
    print(f"\nLLM Response:\n{response}")

    query = "What does divyesh develop algorithms in?"
    print(f"\nQuerying: {query}")

    response, decrypted_mask, context = secure_client.ask_question(query, untrusted_fhe_server, top_k=2)

    print(f"\nServer Returned Mask (Decrypted): {decrypted_mask}")
    print(f"Retrieved Context: {context}")
    print(f"\nLLM Response:\n{response}")

    query = "What does Imtiyaz do?"
    print(f"\nQuerying: {query}")

    response, decrypted_mask, context = secure_client.ask_question(query, untrusted_fhe_server, top_k=2)

    print(f"\nServer Returned Mask (Decrypted): {decrypted_mask}")
    print(f"Retrieved Context: {context}")
    print(f"\nLLM Response:\n{response}")
    
    query = "How many papers and patents has divyesh filed?"
    response, decrypted_mask, context = secure_client.ask_question(query, untrusted_fhe_server, top_k=2)

    print(f"\nServer Returned Mask (Decrypted): {decrypted_mask}")
    print(f"Retrieved Context: {context}")
    print(f"\nLLM Response:\n{response}")
