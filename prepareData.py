from sentence_transformers import SentenceTransformer

class BGEEmbedding:

    def __init__(self):
        self.model = SentenceTransformer("BAAI/bge-base-en-v1.5")

    def embed(self, texts):
        return self.model.encode(texts, normalize_embeddings=True)

class Chunker:

    def __init__(self, size=350, overlap=60):
        self.size = size
        self.overlap = overlap

    def chunk(self, text):
        words = text.split()
        chunks = []

        i = 0
        while i < len(words):
            chunk = words[i:i+self.size]
            chunks.append(" ".join(chunk))
            i += self.size - self.overlap

        return chunks

from sklearn.cluster import KMeans

class IVFIndex:

    def __init__(self, n_clusters=64):
        self.n_clusters = n_clusters
        self.kmeans = None
        self.clusters = {}
        self.centroids = None

    def build(self, embeddings, documents):

        self.kmeans = KMeans(n_clusters=self.n_clusters)
        labels = self.kmeans.fit_predict(embeddings)

        self.centroids = self.kmeans.cluster_centers_

        for i in range(self.n_clusters):
            self.clusters[i] = []

        for idx, label in enumerate(labels):
            self.clusters[label].append((embeddings[idx], documents[idx]))

    def get_cluster(self, cluster_id):
        return self.clusters[cluster_id]
