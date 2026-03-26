import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import warnings

# Suppress HuggingFace pipeline warnings for clean output
warnings.filterwarnings("ignore")

# ==========================================
# 1. FHE MATHEMATICAL OPERATIONS (SIMD)
# ==========================================

def fhe_poly_sign_simd(x_array, iterations=12):
    """
    SIMD version of FHE sign evaluation. 
    Processes the entire database matrix simultaneously.
    S(x) = 1.5x - 0.5x^3
    """
    res = x_array
    for _ in range(iterations):
        res = 1.5 * res - 0.5 * (res**3)
    # Shift from [-1, 1] to [0, 1]
    return (res + 1.0) / 2.0

# ==========================================
# 2. UNTRUSTED SERVER (HIERARCHICAL VECTOR DB)
# ==========================================

class ScalableFHEServer:
    def __init__(self):
        # Phase 1: Document Level Storage
        self.doc_ids = []
        self.doc_matrix = None
        
        # Phase 2: Chunk Level Storage
        self.chunk_ids = []
        self.chunk_matrix = None
        
        # Mapping: Which chunk belongs to which document index
        # In FHE, this mapping is handled by how data is packed into the CKKS slots
        self.chunk_to_doc_idx = []

    def store_document(self, doc_id, doc_summary_vector, chunk_id_list, chunk_vector_list):
        """Simulates storing encrypted vectors into the server's matrices."""
        # 1. Store Document Summary
        doc_idx = len(self.doc_ids)
        self.doc_ids.append(doc_id)
        
        if self.doc_matrix is None:
            self.doc_matrix = np.array([doc_summary_vector])
        else:
            self.doc_matrix = np.vstack((self.doc_matrix, doc_summary_vector))
            
        # 2. Store Chunks
        self.chunk_ids.extend(chunk_id_list)
        self.chunk_to_doc_idx.extend([doc_idx] * len(chunk_id_list))
        
        if self.chunk_matrix is None:
            self.chunk_matrix = np.array(chunk_vector_list)
        else:
            self.chunk_matrix = np.vstack((self.chunk_matrix, chunk_vector_list))

    def search_hierarchical(self, query_vector, doc_sensitivity=1.0, chunk_sensitivity=1.2):
        """
        Executes the Two-Phase search purely mathematically.
        Zero branching. Zero sorting.
        """
        if self.doc_matrix is None: return [], []

        # ---------------------------------------------------------
        # PHASE 1: DOCUMENT LEVEL FILTERING
        # ---------------------------------------------------------
        # 1. SIMD Dot Product across all Document Summaries
        doc_scores = np.dot(self.doc_matrix, query_vector)
        
        # 2. Statistical Thresholding for Documents
        doc_mean = np.mean(doc_scores)
        doc_var = np.var(doc_scores)
        
        doc_diff = (doc_scores - (doc_mean + doc_sensitivity * doc_var)) * 0.5
        doc_dampened = doc_diff - (doc_diff**3) / 6.0
        
        # Yields ~1.0 for relevant docs, ~0.0 for irrelevant
        doc_mask = fhe_poly_sign_simd(doc_dampened, iterations=10)
        
        # ---------------------------------------------------------
        # PHASE 2: CHUNK LEVEL FILTERING
        # ---------------------------------------------------------
        # 3. Expand the Document Mask to align with the Chunk slots
        # In FHE, this is a ciphertext multiplication using a pre-computed rotation mask
        expanded_doc_mask = np.array([doc_mask[idx] for idx in self.chunk_to_doc_idx])
        
        # 4. SIMD Dot Product across ALL chunks
        raw_chunk_scores = np.dot(self.chunk_matrix, query_vector)
        
        # 5. MATHEMATICAL ERADICATION: Multiply chunk scores by their Document's mask.
        # If the doc was rejected, these chunk scores instantly become ~0.0
        masked_chunk_scores = raw_chunk_scores * expanded_doc_mask
        
        # 6. Statistical Thresholding for remaining valid chunks
        chunk_mean = np.mean(masked_chunk_scores)
        chunk_var = np.var(masked_chunk_scores)
        
        chunk_diff = (masked_chunk_scores - (chunk_mean + chunk_sensitivity * chunk_var)) * 0.5
        chunk_dampened = chunk_diff - (chunk_diff**3) / 6.0
        
        final_chunk_mask = fhe_poly_sign_simd(chunk_dampened, iterations=12)
        
        # 7. Final Safety Multiplication (Ensures zeroes stay zeroed despite polynomial noise)
        secure_final_mask = final_chunk_mask * expanded_doc_mask

        return self.chunk_ids, secure_final_mask, doc_mask


# ==========================================
# 3. SECURE CLIENT (TRUST ZONE)
# ==========================================

class ScalableSecureClient:
    def __init__(self):
        self.embedder = SentenceTransformer('BAAI/bge-small-en-v1.5')
        self.llm = pipeline('text-generation', model='ibm-granite/granite-3.0-2b-instruct')
        self.local_chunk_store = {}

    def l2_normalize(self, vector):
        norm = np.linalg.norm(vector)
        return vector if norm == 0 else vector / norm

    def ingest_document(self, doc_id, doc_text, server: ScalableFHEServer):
        # 1. Chunk by paragraphs to maintain high semantic density
        paragraphs = [p.strip() for p in doc_text.split('\n\n') if p.strip()]
        
        chunk_ids = []
        chunk_vectors = []
        
        for i, para in enumerate(paragraphs):
            c_id = f"{doc_id}_p{i}"
            self.local_chunk_store[c_id] = para
            
            vec = self.l2_normalize(self.embedder.encode(para))
            chunk_vectors.append(vec)
            chunk_ids.append(c_id)
            
        # 2. Create Document Summary Embedding (Using the mean of its normalized chunks)
        # We re-normalize the mean vector to maintain the [-1, 1] dot product scale
        doc_summary_vector = self.l2_normalize(np.mean(chunk_vectors, axis=0))
        
        # 3. Send batch to Server
        server.store_document(doc_id, doc_summary_vector, chunk_ids, chunk_vectors)

    def ask_question(self, query, server: ScalableFHEServer):
        # 1. Embed and Normalize Query
        q_vec = self.l2_normalize(self.embedder.encode(query))
        
        # 2. Server computes search purely mathematically
        chunk_ids, encrypted_chunk_mask, encrypted_doc_mask = server.search_hierarchical(q_vec)
        
        # 3. Client Decrypts and Rounds the Masks
        decrypted_doc_mask = [round(float(m), 2) for m in encrypted_doc_mask]
        decrypted_chunk_mask = [round(float(m)) for m in encrypted_chunk_mask]
        
        # 4. Context Reconstruction
        retrieved_contexts = []
        for i, mask_val in enumerate(decrypted_chunk_mask):
            if mask_val == 1.0:
                retrieved_contexts.append(self.local_chunk_store[chunk_ids[i]])
                
        context_str = "\n\n".join(retrieved_contexts)
        
        # 5. LLM Generation
        if not context_str:
            return "No relevant context found.", decrypted_doc_mask, decrypted_chunk_mask, ""
            
        messages = [
            {"role": "system", "content": "Answer the question using ONLY the provided context. Be concise."},
            {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {query}"}
        ]
        
        response = self.llm(messages, max_new_tokens=50, return_full_text=False)[0]['generated_text']
        
        return response, decrypted_doc_mask, decrypted_chunk_mask, context_str


# ==========================================
# 4. EXECUTION
# ==========================================
if __name__ == "__main__":
    server = ScalableFHEServer()
    client = ScalableSecureClient()

    # Document 1: Cryptography
    doc_crypto = (
        "Homomorphic encryption allows computation on ciphertexts without decrypting them first. "
        "It is highly secure but computationally intensive.\n\n"
        "Mr X is a security researcher at Google. He develops algorithms in FHE. "
        "His work focuses on reducing the noise in CKKS polynomials."
    )
    
    # Document 2: Biology (Distractor Document)
    doc_biology = (
        "Mitochondria are known as the powerhouses of the cell. They generate most of the cell's supply of ATP.\n\n"
        "Photosynthesis is a process used by plants to convert light energy into chemical energy. "
        "Cellular respiration is the counterpart in animals."
    )
    
    # Document 3: AI Models
    doc_ai = (
        "Granite is a highly efficient language model developed by IBM for enterprise use cases.\n\n"
        "Vector embeddings are numerical representations of text. "
        "The BGE model is currently one of the best open-source embedding models available."
    )

    print("Ingesting Documents...")
    client.ingest_document("Doc_Crypto", doc_crypto, server)
    client.ingest_document("Doc_Bio", doc_biology, server)
    client.ingest_document("Doc_AI", doc_ai, server)

    # We will ask a query specifically targeting the second paragraph of the first document.
    query = "What does Mr X develop algorithms in?"
    print(f"\nQuerying: {query}")
    
    response, doc_mask, chunk_mask, context = client.ask_question(query, server)
    
    print("\n--- FHE MATHEMATICAL MASKS ---")
    print(f"Document Mask (Crypto, Bio, AI): {doc_mask}")
    print(f"Chunk Mask (Across all 6 chunks): {chunk_mask}")
    
    print("\n--- RESULT ---")
    print(f"Retrieved Context:\n{context}")
    print(f"\nLLM Response:\n{response}")
