import numpy as np


class Retriever:
    def __init__(self, embeddings, chunks):
        self.embeddings = embeddings
        self.chunks = chunks

    def search(
        self,
        query_embedding,
        top_k: int = 3,
    ) -> list[dict]:
        """
        Ищет наиболее похожие chunks.
        """

        scores = self.embeddings @ query_embedding

        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []

        for index in top_indices:
            result = self.chunks[index].copy()
            result["score"] = float(scores[index])
            results.append(result)

        return results