import openfhe
import os
import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# 1. AES ENCRYPTION MODULE
# ==========================================
class AESManager:
    """Handles standard AES-GCM encryption for the text payloads."""
    def __init__(self, key=None):
        self.key = key if key else AESGCM.generate_key(bit_length=256)
        self.aesgcm = AESGCM(self.key)

    def encrypt(self, plaintext_string):
        nonce = os.urandom(12) 
        ciphertext = self.aesgcm.encrypt(nonce, plaintext_string.encode('utf-8'), None)
        return nonce + ciphertext

    def decrypt(self, encrypted_payload):
        nonce = encrypted_payload[:12]
        ciphertext = encrypted_payload[12:]
        plaintext_bytes = self.aesgcm.decrypt(nonce, ciphertext, None)
        return plaintext_bytes.decode('utf-8')

# ==========================================
# 2. OPENFHE PARAMETER GENERATION
# ==========================================
def generate_fhe_parameters():
    """Sets up the CKKS CryptoContext with Ring Dimension 2^16."""
    parameters = openfhe.CCParamsCKKSRNS()
    parameters.SetMultiplicativeDepth(14)
    parameters.SetScalingModSize(50)
    parameters.SetFirstModSize(60)
    parameters.SetRingDim(65536)
    parameters.SetBatchSize(8192) 
    
    cc = openfhe.GenCryptoContext(parameters)
    cc.Enable(openfhe.PKE)
    cc.Enable(openfhe.KEYSWITCH)
    cc.Enable(openfhe.LEVELEDSHE)
    cc.Enable(openfhe.ADVANCEDSHE) 
    
    key_pair = cc.KeyGen()
    cc.EvalMultKeyGen(key_pair.secretKey)
    cc.EvalSumKeyGen(key_pair.secretKey)
    
    return cc, key_pair

# ==========================================
# 3. FHE POLYNOMIAL EVALUATION
# ==========================================
def fhe_poly_sign(cc, ctx_x, iterations=3):
    """
    Evaluates S(x) = 1.5x - 0.5x^3 using explicit EvalMult and EvalAdd.
    """
    res = ctx_x
    for _ in range(iterations):
        x_sq = cc.EvalMult(res, res)
        x_cube = cc.EvalMult(x_sq, res)
        term1 = cc.EvalMult(res, 1.5)
        term2 = cc.EvalMult(x_cube, 0.5)
        res = cc.EvalSub(term1, term2)
        
    res_plus_1 = cc.EvalAdd(res, 1.0)
    final_res = cc.EvalMult(res_plus_1, 0.5)
    return final_res

# ==========================================
# 4. SERVER ARCHITECTURE
# ==========================================
class FHEVectorDB:
    def __init__(self, cc):
        self.cc = cc
        self.aes_encrypted_chunks = []
        self.fhe_plaintext_vectors = []

    def store_chunk(self, encrypted_text, plaintext_vector):
        self.aes_encrypted_chunks.append(encrypted_text)
        self.fhe_plaintext_vectors.append(plaintext_vector)

    def secure_search(self, ctx_query_vector, vector_length):
        """
        Executes the search across ALL chunks. 
        Returns the AES chunks and their mathematically evaluated FHE masks.
        """
        encrypted_scores = []
        
        # 1. Compute all dot products
        for ptx_db_vector in self.fhe_plaintext_vectors:
            # Fast ctx * ptx multiplication
            ctx_mult = self.cc.EvalMult(ctx_query_vector, ptx_db_vector)
            ctx_dot_product = self.cc.EvalSum(ctx_mult, vector_length)
            encrypted_scores.append(ctx_dot_product)
            
        encrypted_masks = []
        
        # 2. Evaluate the polynomial mask for every chunk
        for score in encrypted_scores:
            # Shift by a threshold of 0.60. 
            # If the dot product is > 0.60, the polynomial pushes it to 1.0
            threshold_shifted = self.cc.EvalSub(score, 0.60)
            ctx_mask = fhe_poly_sign(self.cc, threshold_shifted, iterations=3)
            encrypted_masks.append(ctx_mask)
            
        return self.aes_encrypted_chunks, encrypted_masks

# ==========================================
# 5. EXECUTION & TESTING
# ==========================================
if __name__ == "__main__":
    print("Loading local models (Embedder & LLM)...")
    embedder = SentenceTransformer('BAAI/bge-small-en-v1.5')
    llm = pipeline('text-generation', model='ibm-granite/granite-3.0-2b-instruct')

    print("Initializing FHE Context (Ring Dim: 2^16)...")
    cc, key_pair = generate_fhe_parameters()
    aes_manager = AESManager()
    server = FHEVectorDB(cc)

    def l2_normalize(vec):
        norm = np.linalg.norm(vec)
        return vec if norm == 0 else vec / norm

    # Define the documents
    doc1 = (
        "Homomorphic encryption allows computation on ciphertexts without decrypting them first. "
        "It is highly secure but computationally intensive.\n\n"
        "Mr X is a security researcher at Google. He develops algorithms in FHE. "
        "His work focuses on reducing the noise in CKKS polynomials."
    )
    
    doc2 = (
        "Mitochondria are known as the powerhouses of the cell. They generate most of the cell's supply of ATP.\n\n"
        "Photosynthesis is a process used by plants to convert light energy into chemical energy. "
        "Cellular respiration is the counterpart in animals."
    )
    
    doc3 = (
        "Granite is a highly efficient language model developed by IBM for enterprise use cases.\n\n"
        "Vector embeddings are numerical representations of text. "
        "The BGE model is currently one of the best open-source embedding models available."
    )

    print("\nChunking and Encrypting Documents...")
    documents = {"doc1": doc1, "doc2": doc2, "doc3": doc3}
    
    for doc_name, text in documents.items():
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        for para in paragraphs:
            vector = l2_normalize(embedder.encode(para)).tolist()
            aes_ciphertext = aes_manager.encrypt(para)
            ptx_vector = cc.MakeCKKSPackedPlaintext(vector)
            server.store_chunk(aes_ciphertext, ptx_vector)

    query = "What does Mr X develop algorithms in?"
    print(f"\nQuerying: '{query}'")
    
    query_vector = l2_normalize(embedder.encode(query)).tolist()
    vector_len = len(query_vector)
    
    ptx_query = cc.MakeCKKSPackedPlaintext(query_vector)
    ctx_query = cc.Encrypt(key_pair.publicKey, ptx_query)

    print("Server computing Ciphertext-Plaintext dot products and polynomials...")
    retrieved_aes_chunks, encrypted_masks = server.secure_search(ctx_query, vector_len)

    print("Client decrypting search results...")
    retrieved_contexts = []
    
    for i, ctx_mask in enumerate(encrypted_masks):
        ptx_mask = cc.Decrypt(key_pair.secretKey, ctx_mask)
        mask_value = ptx_mask.GetRealPackedValue()[0]
        
        # Check against our expected binary range
        if mask_value > 0.5:
            decrypted_text = aes_manager.decrypt(retrieved_aes_chunks[i])
            retrieved_contexts.append(decrypted_text)
            print(f" -> Found match at chunk {i} (Mask Value: {mask_value:.4f})")

    context_str = "\n\n".join(retrieved_contexts)
    
    print(f"\n--- Decrypted Context ---\n{context_str}\n")
    
    if context_str:
        print("Generating final answer with Granite...")
        messages = [
            {"role": "system", "content": "Answer the question using ONLY the provided context. Be concise."},
            {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {query}"}
        ]
        
        response = llm(messages, max_new_tokens=50, return_full_text=False)[0]['generated_text']
        print(f"\nLLM Response:\n{response}")
    else:
        print("No relevant context found to answer the query.")
