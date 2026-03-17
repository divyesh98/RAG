import numpy as np
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
import asyncio
from Searcher import *
from prepareData import *
from engine import *
from PackedSimilarity import *
from EncryptedVector import *

def load_documents(path):

    with open(path, "r", encoding="utf-8") as f:
        docs = [line.strip() for line in f.readlines() if line.strip()]

    return docs


def build_system(doc_path):

    docs = load_documents(doc_path)

    chunker = Chunker()
    embedder = BGEEmbedding()

    all_chunks = []

    for doc in docs:
        all_chunks.extend(chunker.chunk(doc))

    print(f"Total chunks: {len(all_chunks)}")

    embeddings = embedder.embed(all_chunks)

    index = IVFIndex(n_clusters=4)
    index.build(embeddings, all_chunks)

    searcher = IVFSearcher(index)

    engine = QueryEngine(embedder, searcher)

    return engine

if __name__ == "__main__":

    engine = build_system("data/documents.txt")

    # -------- SINGLE QUERY --------
    query = "What is homomorphic encryption?"

    print("\nSingle Query Result:\n")

    result = asyncio.run(run_queries(engine, [query]))

    for r in result[0]:
        print("-", r)

    # -------- MULTI QUERY (1000) --------
    queries = ["What is AI?"] * 1000

    print("\nRunning 1000 queries...\n")

    results = asyncio.run(run_queries(engine, queries))

    print(f"Completed {len(results)} queries.")
