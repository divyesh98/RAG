import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from openfhe import *

def generate_params():
    mult_depth = 20
    scale_mod_size = 50
    batch_size = 512
    
    parameters = CCParamsCKKSRNS()
    parameters.SetMultiplicativeDepth(mult_depth)
    parameters.SetScalingModSize(scale_mod_size)
    parameters.SetBatchSize(batch_size)

    cc = GenCryptoContext(parameters)
    cc.Enable(PKESchemeFeature.PKE)
    cc.Enable(PKESchemeFeature.KEYSWITCH)
    cc.Enable(PKESchemeFeature.LEVELEDSHE)

    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)
    cc.EvalRotateKeyGen(keys.secretKey, [1, 2, 4, 8, 16, 32, 64, 128, 256])

    return cc, keys.publicKey, keys.secretKey

class SecureClient:
    def __init__(self, cc, secKey):
        self.embedder = SentenceTransformer('BAAI/bge-small-en-v1.5')
        self.llm = pipeline('text-generation', model = 'ibm-granite/granite-3.0-2b-instruct', max_new_tokens=100)
        self.local_chunk_store = {}
        self.cc = cc
        self.secKey = secKey

    def l2_normalize(self, vector):
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm

    def ingest_document_advanced(self, doc_text, server):
        sentences = [s.strip() + "."  for s in doc_text.split('.') if s.strip()]

        for i in range(len(sentences)):
            window = " ".join(sentence[max(0, i-2): i+1])

            chunk_id = f"chunk_{i}"
            self.local_chunk_store[chunk_id] = sentence[i]

            vector = self.embedder.encode(window)
            normalized_vector = self.l2_normalize(vector)
            
            ctx = self.cc.Encrypt(self.pubKey, self.cc.MakeCKKSPackedPlaintext(normalized_vector))
            server.store(chunk_id, ctx, sentence[i])

    def ask_question(self, query, server: FHEServer, top_k=2):
        q_vec = self.embedder.encode(query)
        norm_q_vec = self.l2_normalize(q_vec)
        q_vec_enc = self.cc.Encrypt(self.pubKey, self.cc.MakeCKKSPackedPlaintext(norm_q_vec))

        chunk_ids, encrypted_masks, sentences = server.search(q_vec_enc, k=top_k)
        
        enc_masks = [cc.Decrypt(mask, self.secKey) for mask in encrypted_masks)]

        decrypted_mask = [round(m) for m in (cc.Decrypt(encrypted_masks, self.secKey).)]

if __name__ == "__main__":
    
    cc, pubKey, secKey = generate_params()
    
    trusted_fhe_server = FHEServer()
    secure_client = SecureClient()

    doc = (        
           "Homomorphic encryption allows computation on ciphertexts. "                                                            "The BGE model is great for creating semantic embeddings. "                                                             "Python is a versatile programming language. "                                                                          "Granite is a scalable language model developed by IBM. "                                                               "Quantum computing uses qubits to perform operations."                                                                  "Homomorphic encryption allows computation on ciphertexts. "                                                            "The BGE model is great for creating semantic embeddings. "                                                             "Python is a versatile programming language. "                                                                          "Granite is a scalable language model developed by IBM. "                                                               "Quantum computing uses qubits to perform operations. "                                                                 "Vector databases store embeddings for similarity search. "                                                             "L2 normalization is required for dot product similarity."                                                              "Divyesh Saglani is a researcher at TCS Research. He develops algorithms in FHE. Divyesh has an experience of more than 5 years. He also have several papers and patents, which are more than 20."
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
