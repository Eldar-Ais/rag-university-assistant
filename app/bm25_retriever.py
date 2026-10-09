import re
from rank_bm25 import BM25Okapi


class BM25Retriever:
    def __init__(self, chunks):
        self.chunks = chunks

        tokenized_chunks = [
            self.tokenize(chunk["text"])
            for chunk in chunks
        ]

        self.bm25 = BM25Okapi(tokenized_chunks)

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        tokenized_query = self.tokenize(query)

        scores = self.bm25.get_scores(tokenized_query)

        top_indices = scores.argsort()[::-1][:top_k]

        results = []

        for index in top_indices:
            result = self.chunks[index].copy()
            result["score"] = float(scores[index])
            results.append(result)

        return results