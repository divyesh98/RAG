import os
import glob
import random
import re
import numpy as np
import openfhe
import PyPDF2
import warnings
import concurrent.futures
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import time

warnings.filterwarnings("ignore")

# ==========================================
# 1. DOCUMENT INGESTION & PROCESSING
# ==========================================
class DocumentProcessor:
    def __init__(self, docs_dir="docs"):
        self.docs_dir = docs_dir
        if not os.path.exists(self.docs_dir):
            os.makedirs(self.docs_dir)
            print(f"[*] Created directory '{self.docs_dir}'. Please place .txt or .pdf files here.")

    def clean_text(self, text):
        return re.sub(r'\s+', ' ', text).strip()

    def read_pdf(self, file_path):
        text_blocks = []
        try:
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text_blocks.append(extracted)
        except Exception as e:
            print(f"[!] Error reading PDF {file_path}: {e}")
        return "\n".join(text_blocks)

    def read_txt(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    def process_documents(self):
        chunks = []
        files = glob.glob(os.path.join(self.docs_dir, "*.*"))
        
        for file_path in files:
            filename = os.path.basename(file_path)
            topic_label = filename.replace("document_", "Topic ").split('.')[0]
            
            ext = filename.lower().split('.')[-1]
            if ext == 'txt':
                content = self.read_txt(file_path)
            elif ext == 'pdf':
                content = self.read_pdf(file_path)
            else:
                continue
                
            paragraphs = [p for p in content.split('\n\n') if len(p.strip()) > 50]
            
            for p in paragraphs:
                cleaned_p = self.clean_text(p)
                enriched_chunk = f"Source: {topic_label} | Content: {cleaned_p}"
                chunks.append(enriched_chunk)
            
        return chunks

# ==========================================
# 2. SECURE TEE ENCLAVE (The "Black Box")
# ==========================================
class TrustedEnclave:
    def __init__(self, cc, secret_key, aes_key):
        self.cc = cc
        self.secret_key = secret_key
        self.aes_manager = AESGCM(aes_key)
        self.llm = pipeline('text-generation', model='ibm-granite/granite-3.0-2b-instruct')
        print("[TEE] Enclave Initialized: Keys Provisioned & LLM Loaded.")

    def process_and_generate(self, user_query, encrypted_masks, aes_bucket):
        print("[TEE] Decrypting FHE masks and AES payloads...")
        verified_contexts = []
        query_terms = user_query.lower().replace('?', '').split()
        digits_in_query = [term for term in query_terms if term.isdigit()]
        
        for cid, ctx_mask in encrypted_masks.items():
            ptx_mask = self.cc.Decrypt(self.secret_key, ctx_mask)
            mask_val = ptx_mask.GetRealPackedValue()[0]
            
            if mask_val > 0.4:  
                encrypted_blob = aes_bucket[cid]
                nonce = encrypted_blob[:12]
                ciphertext = encrypted_blob[12:]
                decrypted_text = self.aes_manager.decrypt(nonce, ciphertext, None).decode('utf-8')
                
                text_lower = decrypted_text.lower()
                
                # Fast-fail verification
                is_valid = True
                for digit in digits_in_query:
                    if digit not in text_lower:
                        is_valid = False
                        break
                        
                if is_valid:
                    verified_contexts.append(decrypted_text)

        if not verified_contexts:
            return "No specific information found for your request based on the provided exact parameters."
            
        context_str = "\n".join(verified_contexts)
        prompt = f"Facts:\n{context_str}\n\nQuestion: {user_query}\nAnswer briefly and directly based on the facts:"
        
        print("[TEE] Generating answer based on verified facts...")
        response = self.llm(prompt, max_new_tokens=50, return_full_text=False)[0]['generated_text']
        return response.strip()

# ==========================================
# 3. UNTRUSTED PARALLEL CLOUD SERVER
# ==========================================
class CloudServer:
    def __init__(self, cc):
        self.cc = cc
        self.db = {} 
        self.max_workers = os.cpu_count()

    def _process_single_vector(self, cid, db_vec, ctx_query):
        """Worker thread function for heavy Ctx-Ctx FHE math."""
        ctx_mult = self.cc.EvalMult(ctx_query, db_vec)
        ctx_dot = self.cc.EvalSum(ctx_mult, 384) 
        
        ctx_shifted = self.cc.EvalSub(ctx_dot, 0.50) 
        ctx_shifted_plus_1 = self.cc.EvalAdd(ctx_shifted, 1.0)
        ctx_mask = self.cc.EvalMult(ctx_shifted_plus_1, 0.5)
        return cid, ctx_mask

    def search(self, ctx_query):
        print(f"[Server] Executing Parallel Ctx-Ctx Search across {len(self.db)} records using {self.max_workers} threads...")
        masks = {}
        
        # Dispatch FHE workload across all available CPU cores
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_cid = {
                executor.submit(self._process_single_vector, cid, data["vec"], ctx_query): cid 
                for cid, data in self.db.items()
            }
            
            for future in concurrent.futures.as_completed(future_to_cid):
                cid, ctx_mask = future.result()
                masks[cid] = ctx_mask
            
        selected_ids = list(self.db.keys())
        bucket_size = min(20, len(selected_ids))
        if len(selected_ids) > bucket_size:
            selected_ids = random.sample(selected_ids, bucket_size)
            
        return {cid: masks[cid] for cid in selected_ids}, \
               {cid: self.db[cid]["text"] for cid in selected_ids}

# ==========================================
# 4. THIN CLIENT
# ==========================================
class ThinClient:
    def __init__(self, public_key, embedder):
        self.public_key = public_key
        self.embedder = embedder

    def create_query(self, cc, text):
        vec = self.embedder.encode(text)
        vec = vec / np.linalg.norm(vec)
        ptx = cc.MakeCKKSPackedPlaintext(vec.tolist())
        return cc.Encrypt(self.public_key, ptx)

# ==========================================
# 5. ORCHESTRATION (The Workflow)
# ==========================================
def run_production_pipeline():
    print("Initializing FHE Parameters...")
    params = openfhe.CCParamsCKKSRNS()
    params.SetMultiplicativeDepth(12)
    cc = openfhe.GenCryptoContext(params)
    cc.Enable(openfhe.PKE); cc.Enable(openfhe.KEYSWITCH); cc.Enable(openfhe.LEVELEDSHE); cc.Enable(openfhe.ADVANCEDSHE)
    kp = cc.KeyGen()
    cc.EvalMultKeyGen(kp.secretKey)
    cc.EvalSumKeyGen(kp.secretKey)

    aes_key = AESGCM.generate_key(bit_length=256)
    embedder = SentenceTransformer('BAAI/bge-small-en-v1.5')
    
    server = CloudServer(cc)
    enclave = TrustedEnclave(cc, kp.secretKey, aes_key)
    client = ThinClient(kp.publicKey, embedder)

    processor = DocumentProcessor("docs")
    document_chunks = processor.process_documents()
    
    if not document_chunks:
        print("[!] No text found in 'docs/' folder. Please add files and run again.")
        return

    print(f"\n[*] Encrypting and Ingesting {len(document_chunks)} chunks from 'docs/'...")
    for i, chunk_text in enumerate(document_chunks):
        chunk_id = f"chunk_{i}"
        
        nonce = os.urandom(12)
        aes_blob = nonce + AESGCM(aes_key).encrypt(nonce, chunk_text.encode('utf-8'), None)
        
        vec = embedder.encode(chunk_text)
        vec = vec / np.linalg.norm(vec)
        ctx_vec = cc.Encrypt(kp.publicKey, cc.MakeCKKSPackedPlaintext(vec.tolist()))
        
        server.db[chunk_id] = {"vec": ctx_vec, "text": aes_blob}

    query_text = "When was topic 5 conceptualized?"
    print(f"\n[Client] Sending Query: '{query_text}'")
    ctx_query = client.create_query(cc, query_text)

    s =  time.time()    
    encrypted_masks, aes_bucket = server.search(ctx_query)
    e  = time.time()
    print(f"Time taken for database search is: {e-s} s")

    s = time.time()
    final_answer = enclave.process_and_generate(query_text, encrypted_masks, aes_bucket)
    e = time.time()

    print(f"Time taken for response generation is {e-s} s")
    print(f"\n=== FINAL ANSWER DIRECT TO CLIENT ===\n{final_answer}\n=====================================")

if __name__ == "__main__":
    run_production_pipeline()
