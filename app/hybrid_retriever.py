class HybridRetriever:
    def __init__(self, semantic_retriever, bm25_retriever, rrf_k=60):
        self.semantic_retriever = semantic_retriever
        self.bm25_retriever = bm25_retriever
        self.rrf_k = rrf_k

    def search(self, query, query_embedding, top_k=5, candidate_k=10):
        semantic_results = self.semantic_retriever.search(
            query_embedding=query_embedding,
            top_k=candidate_k,
        )

        bm25_results = self.bm25_retriever.search(
            query=query,
            top_k=candidate_k,
        )

        fused = {}

        for rank, result in enumerate(semantic_results, start=1):
            key = (result["source"], result["page"], result["text"])
            fused.setdefault(key, {
                "chunk": result,
                "rrf_score": 0.0,
            })
            fused[key]["rrf_score"] += 1 / (self.rrf_k + rank)

        for rank, result in enumerate(bm25_results, start=1):
            key = (result["source"], result["page"], result["text"])
            fused.setdefault(key, {
                "chunk": result,
                "rrf_score": 0.0,
            })
            fused[key]["rrf_score"] += 1 / (self.rrf_k + rank)

        ranked = sorted(
            fused.values(),
            key=lambda item: item["rrf_score"],
            reverse=True,
        )

        results = []

        for item in ranked[:top_k]:
            result = item["chunk"].copy()
            result["rrf_score"] = item["rrf_score"]
            results.append(result)

        return results