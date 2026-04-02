import os
import glob
import random
import warnings
import numpy as np
import openfhe
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import PyPDF2

warnings.filterwarnings("ignore")

# ==========================================
# 1. DOCUMENT INGESTION & PROCESSING
# ==========================================
class DocumentProcessor:
    """Reads .txt and .pdf files from a specified directory and chunks them."""
    def __init__(self, docs_dir="docs"):
        self.docs_dir = docs_dir
        if not os.path.exists(self.docs_dir):
            os.makedirs(self.docs_dir)
            print(f"[*] Created directory '{self.docs_dir}'. Please place .txt or .pdf files here.")

    def read_pdf(self, file_path):
        text = ""
        try:
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            print(f"[!] Error reading PDF {file_path}: {e}")
        return text

    def read_txt(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
   '''
    def process_documents(self):
        chunks = []
        files = glob.glob(os.path.join(self.docs_dir, "*.*"))

        for file_path in files:
            ext = file_path.lower().split('.')[-1]
            if ext == 'txt':
                content = self.read_txt(file_path)
            elif ext == 'pdf':
                content = self.read_pdf(file_path)
            else:
                continue

            # Chunking strategy: split by double newline for paragraphs
            paragraphs = [p.strip() for p in content.split('\n\n') if len(p.strip()) > 50]
            chunks.extend(paragraphs)

        return chunks
    '''
    # --- UPDATE IN DocumentProcessor ---
    def process_documents(self):
        chunks = []
        files = glob.glob(os.path.join(self.docs_dir, "*.*"))

        for file_path in files:
            filename = os.path.basename(file_path)
            # Identify the topic from filename (e.g., "document_5.docx" -> "Topic 5")
            topic_label = filename.replace("document_", "Topic ").split('.')[0]

            # ... (reading logic for txt/pdf) ...
            ext = file_path.lower().split('.')[-1]
            if ext == 'txt':
                content = self.read_txt(file_path)
            elif ext == 'pdf':
                content = self.read_pdf(file_path)
            else:
                continue

            # IMPROVED CHUNKING: Prepend the Topic Label to every chunk
            paragraphs = [p.strip() for p in content.split('\n\n') if len(p.strip()) > 50]
            for p in paragraphs:
                # We "inject" the subject so the LLM always knows the context
                enriched_chunk = f"Source: {topic_label}\nContent: {p}"
                chunks.append(enriched_chunk)

        return chunks

# ==========================================
# 2. CRYPTOGRAPHY MODULES (FHE & AES)
# ==========================================
class AESManager:
    """Handles local AES-256-GCM encryption for text payloads."""
    def __init__(self):
        self.key = AESGCM.generate_key(bit_length=256)
        self.aesgcm = AESGCM(self.key)

    def encrypt(self, plaintext_string):
        nonce = os.urandom(12)
        ciphertext = self.aesgcm.encrypt(nonce, plaintext_string.encode('utf-8'), None)
        return nonce + ciphertext

    def decrypt(self, encrypted_payload):
        nonce = encrypted_payload[:12]
        ciphertext = encrypted_payload[12:]
        return self.aesgcm.decrypt(nonce, ciphertext, None).decode('utf-8')

def generate_fhe_parameters():
    """Sets up the CKKS CryptoContext for Ctx-Ctx operations."""
    parameters = openfhe.CCParamsCKKSRNS()
    parameters.SetMultiplicativeDepth(16) # Increased depth for Ctx-Ctx + Poly
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
    cc.EvalMultKeyGen(key_pair.secretKey) # Crucial for Ctx-Ctx math
    cc.EvalSumKeyGen(key_pair.secretKey)

    return cc, key_pair

def fhe_poly_sign(cc, ctx_x, iterations=3):
    """Evaluates S(x) = 1.5x - 0.5x^3 to push values to 0.0 or 1.0"""
    res = ctx_x
    for _ in range(iterations):
        x_sq = cc.EvalMult(res, res)
        x_cube = cc.EvalMult(x_sq, res)
        term1 = cc.EvalMult(res, 1.5)
        term2 = cc.EvalMult(x_cube, 0.5)
        res = cc.EvalSub(term1, term2)

    res_plus_1 = cc.EvalAdd(res, 1.0)
    return cc.EvalMult(res_plus_1, 0.5)

# ==========================================
# 3. UNTRUSTED SERVER (Vector DB)
# ==========================================
class UntrustedServerDB:
    def __init__(self, cc):
        self.cc = cc
        self.database = {} # Format: { chunk_id: {"aes_text": bytes, "ctx_vector": Ciphertext} }

    def store_chunk(self, chunk_id, encrypted_text, ctx_vector):
        self.database[chunk_id] = {
            "aes_text": encrypted_text,
            "ctx_vector": ctx_vector
        }

   def compute_search_masks(self, ctx_query, vector_length):
        """
        Step 1 of Search: Double-Blind Ctx-Ctx dot product + masking.
        Returns ONLY the chunk IDs and their encrypted masks.
        """
        chunk_ids = list(self.database.keys())
        encrypted_masks = []

        for cid in chunk_ids:
            ctx_db_vec = self.database[cid]["ctx_vector"]

            # Ctx * Ctx Multiplication
            ctx_mult = self.cc.EvalMult(ctx_query, ctx_db_vec)
            ctx_dot = self.cc.EvalSum(ctx_mult, vector_length)

            # Shift by 0.60 threshold and apply polynomial mask
            ctx_shifted = self.cc.EvalSub(ctx_dot, 0.60)
            ctx_mask = fhe_poly_sign(self.cc, ctx_shifted, iterations=3)

            encrypted_masks.append(ctx_mask)

        return chunk_ids, encrypted_masks

    def fetch_aes_bucket(self, requested_ids):
        """
        Step 2 of Search: Returns AES blobs for the requested bucket.
        The server does not know which IDs are real and which are chaff.
        """
        return {cid: self.database[cid]["aes_text"] for cid in requested_ids if cid in self.database}

# ==========================================
# 4. SECURE CLIENT ENCLAVE
# ==========================================
class PrivacyRAGClient:
    def __init__(self, cc, key_pair, embedder, llm, server):
        self.cc = cc
        self.key_pair = key_pair
        self.aes_manager = AESManager()
        self.embedder = embedder
        self.llm = llm
        self.server = server
        self.vector_length = 384 # BGE-small dimension

    def l2_normalize(self, vec):
        norm = np.linalg.norm(vec)
        return vec if norm == 0 else vec / norm
    def ingest_documents(self, chunks):
        print(f"\n[Client] Encrypting {len(chunks)} chunks and uploading to Server...")
        for i, text in enumerate(chunks):
            chunk_id = f"chunk_{i}"

            # 1. AES Encrypt Text
            aes_ciphertext = self.aes_manager.encrypt(text)

            # 2. Fully Encrypt Vector (Ctx)
            vector = self.l2_normalize(self.embedder.encode(text)).tolist()
            ptx_vector = self.cc.MakeCKKSPackedPlaintext(vector)
            ctx_vector = self.cc.Encrypt(self.key_pair.publicKey, ptx_vector)

            # 3. Upload to Server
            self.server.store_chunk(chunk_id, aes_ciphertext, ctx_vector)
        print("[Client] Upload complete.")

    def query(self, user_query, bucket_size=10):
        print(f"\n[Client] Querying: '{user_query}'")

        # 1. Encrypt Query
        query_vec = self.l2_normalize(self.embedder.encode(user_query)).tolist()
        ptx_query = self.cc.MakeCKKSPackedPlaintext(query_vec)
        ctx_query = self.cc.Encrypt(self.key_pair.publicKey, ptx_query)

        # 2. Server computes blind masks
        print("[Server] Computing Ctx-Ctx Dot Products and Masks...")
        all_ids, encrypted_masks = self.server.compute_search_masks(ctx_query, self.vector_length)

        # 3. Client decrypts masks locally
        print("[Client] Decrypting masks to identify relevant IDs...")
        real_match_ids = []
        for cid, ctx_mask in zip(all_ids, encrypted_masks):
            ptx_mask = self.cc.Decrypt(self.key_pair.secretKey, ctx_mask)
            mask_val = ptx_mask.GetRealPackedValue()[0]
            if mask_val > 0.5:
                real_match_ids.append(cid)
                print(f"  -> Found match: {cid} (Mask: {mask_val:.4f})")

        # 4. Construct K-Anonymous Bucket (Real + Chaff)
        print(f"[Client] Constructing K-Anonymous bucket of size {bucket_size}...")
        bucket_request = list(real_match_ids)
        available_chaff = [cid for cid in all_ids if cid not in real_match_ids]
        
        while len(bucket_request) < bucket_size and available_chaff:
            chaff_id = random.choice(available_chaff)
            bucket_request.append(chaff_id)
            available_chaff.remove(chaff_id)

        # Shuffle so the server can't guess based on order
        random.shuffle(bucket_request)

        # 5. Fetch AES chunks from Server
        print("[Server] Delivering requested AES bucket...")
        aes_bucket = self.server.fetch_aes_bucket(bucket_request)

        # 6. Decrypt ONLY the real matches
        retrieved_contexts = []
        for cid in real_match_ids:
            decrypted_text = self.aes_manager.decrypt(aes_bucket[cid])
            retrieved_contexts.append(decrypted_text)

        context_str = "\n\n".join(retrieved_contexts)

        # 7. LLM Generation
        if context_str:
            print("\n[Client] Generating response using local SLM...")
            messages = [
                {"role": "system", "content": "Answer the question using ONLY the provided context. Be concise."},
                {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {user_query}"}
            ]
            response = self.llm(messages, max_new_tokens=100, return_full_text=False)[0]['generated_text']
            print(f"\n=== FINAL LLM RESPONSE ===\n{response}\n==========================")
        else:
            print("\n=== FINAL LLM RESPONSE ===\nNo relevant context found to answer the query.\n==========================")


# ==========================================
# 5. MAIN EXECUTION
# ==========================================
f __name__ == "__main__":
    # Setup dummy docs if the folder is empty for immediate testing
    os.makedirs("docs", exist_ok=True)
    if not os.listdir("docs"):
        with open("docs/dummy1.txt", "w") as f:
            f.write("Homomorphic encryption allows computation on ciphertexts without decrypting them first. It is highly secure but computationally intensive.\n\nMr X is a security researcher at Google. He develops algorithms in FHE. His work focuses on reducing the noise in CKKS polynomials.")
        with open("docs/dummy2.txt", "w") as f:
            f.write("Mitochondria are known as the powerhouses of the cell. They generate most of the cell's supply of ATP.\n\nPhotosynthesis is a process used by plants to convert light energy into chemical energy. Cellular respiration is the counterpart in animals.")
        print("[*] Generated dummy text files in './docs' for testing.")

    print("Loading AI Models...")
    embedder_model = SentenceTransformer('BAAI/bge-small-en-v1.5')
    llm_model = pipeline('text-generation', model='ibm-granite/granite-3.0-2b-instruct')

    print("Initializing FHE Environment...")
    cc, key_pair = generate_fhe_parameters()

    # Initialize Architecture
    server = UntrustedServerDB(cc)
    client = PrivacyRAGClient(cc, key_pair, embedder_model, llm_model, server)
    # Ingestion Pipeline
    processor = DocumentProcessor("docs")
    document_chunks = processor.process_documents()

    if document_chunks:
        client.ingest_documents(document_chunks)

        # Ensure the bucket size isn't larger than our total dummy database for the test run
        test_bucket_size = min(10, len(document_chunks))
        #client.query("What does Mr X develop algorithms in?", bucket_size=test_bucket_size)
        #client.query("What is the main focus of topic 3", bucket_size=test_bucket_size)
        client.query("When was topic 5 conceptualized", bucket_size=test_bucket_size)
    else:
        print("[!] No documents found in 'docs/' directory to process.")
