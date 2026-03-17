class PackedSimilarity:

    def __init__(self, dim=512, pack_size=16):
        self.dim = dim
        self.pack_size = pack_size

    def pack_vectors(self, vectors):

        packed = []

        for i in range(0, len(vectors), self.pack_size):

            block = vectors[i:i+self.pack_size]

            if len(block) < self.pack_size:
                break

            concat = np.concatenate(block)
            packed.append(EncryptedVector(concat))

        return packed

    def pack_query(self, query):

        repeated = np.tile(query, self.pack_size)

        return EncryptedVector(repeated)

    def compute_similarity(self, packed_query, packed_vectors):

        scores = []

        for pv in packed_vectors:

            prod = pv.multiply(packed_query)

            data = prod.decrypt()

            # split back
            sims = data.reshape(self.pack_size, self.dim).sum(axis=1)

            scores.extend(sims)

        return scores
