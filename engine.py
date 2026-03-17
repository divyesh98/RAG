import asyncio
import numpy as np
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
import asyncio

class QueryEngine:

    def __init__(self, embedder, searcher):
        self.embedder = embedder
        self.searcher = searcher

    async def process(self, query):

        emb = self.embedder.embed([query])[0]

        results = self.searcher.search(emb)

        return results


async def run_queries(engine, queries):

    tasks = [engine.process(q) for q in queries]

    return await asyncio.gather(*tasks)
