import numpy as np
class IVFSearcher:

    def __init__(self, index):
        self.index = index

    def search(self, query_embedding, top_k=5):

        # Step 1: find nearest centroid
        sims = np.dot(self.index.centroids, query_embedding)

        cluster_id = np.argmax(sims)

        # Step 2: search inside cluster
        cluster = self.index.get_cluster(cluster_id)

        embeddings = [x[0] for x in cluster]
        docs = [x[1] for x in cluster]

        sims = np.dot(embeddings, query_embedding)

        top_idx = np.argsort(sims)[-top_k:][::-1]

        return [docs[i] for i in top_idx]

class HybridANN:

    def __init__(self, index, projection_dim=64):
        self.index = index
        self.projection = np.random.randn(512, projection_dim)

    def hash(self, vec):
        proj = np.dot(vec, self.projection)
        return (proj > 0).astype(int)

    def search(self, query, top_k=5):

        q_hash = self.hash(query)

        # filter clusters (approx)
        centroid_hashes = [self.hash(c) for c in self.index.centroids]

        scores = [np.sum(q_hash == h) for h in centroid_hashes]

        cluster_id = np.argmax(scores)

        cluster = self.index.get_cluster(cluster_id)

        embeddings = [x[0] for x in cluster]
        docs = [x[1] for x in cluster]

        sims = np.dot(embeddings, query)

        top_idx = np.argsort(sims)[-top_k:][::-1]

        return [docs[i] for i in top_idx]

