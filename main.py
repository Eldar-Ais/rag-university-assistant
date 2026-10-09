from app.load_documents import load_pdf
from app.chunking import chunk_documents
from app.embeddings import EmbeddingModel
from app.retriever import Retriever
from app.bm25_retriever import BM25Retriever
from app.hybrid_retriever import HybridRetriever


def main():
    # 1. Загружаем PDF
    documents = load_pdf("data/students_guide_enu.pdf")
    print(f"Загружено страниц: {len(documents)}")

    # 2. Создаём chunks
    chunks = chunk_documents(documents)
    print(f"Создано chunks: {len(chunks)}")

    # 3. Инициализируем embedding model
    embedding_model = EmbeddingModel()

    # 4. Создаём embeddings документов
    texts = [chunk["text"] for chunk in chunks]
    embeddings = embedding_model.encode(texts)

    print(f"Размер embeddings: {embeddings.shape}")

    # 5. Создаём поисковые компоненты
    retriever = Retriever(
        embeddings=embeddings,
        chunks=chunks,
    )

    bm25_retriever = BM25Retriever(chunks)

    # 6. Создаём hybrid retriever
    hybrid_retriever = HybridRetriever(
        semantic_retriever=retriever,
        bm25_retriever=bm25_retriever,
    )

    # 7. Получаем вопрос пользователя
    query = input("\nВаш вопрос: ").strip()

    if not query:
        print("Вопрос не должен быть пустым.")
        return

    # 8. Создаём embedding вопроса
    query_embedding = embedding_model.encode([query])[0]

    # 9. Выполняем semantic search
    semantic_results = retriever.search(
        query_embedding=query_embedding,
        top_k=5,
    )

    # 10. Выполняем BM25 search
    bm25_results = bm25_retriever.search(
        query=query,
        top_k=5,
    )

    # 11. Выполняем hybrid search
    hybrid_results = hybrid_retriever.search(
        query=query,
        query_embedding=query_embedding,
        top_k=5,
    )

    # 12. Выводим semantic search
    print("\n" + "=" * 70)
    print("SEMANTIC SEARCH RESULTS")
    print("=" * 70)

    for i, result in enumerate(semantic_results, start=1):
        print("-" * 70)
        print(f"Результат #{i}")
        print(f"Источник: {result['source']}")
        print(f"Страница: {result['page']}")
        print(f"Score: {result['score']:.4f}")
        print(result["text"][:500])
        print()

    # 13. Выводим BM25
    print("\n" + "=" * 70)
    print("BM25 RESULTS")
    print("=" * 70)

    for i, result in enumerate(bm25_results, start=1):
        print("-" * 70)
        print(f"Результат #{i}")
        print(f"Источник: {result['source']}")
        print(f"Страница: {result['page']}")
        print(f"Score: {result['score']:.4f}")
        print(result["text"][:500])
        print()

    # 14. Выводим hybrid search
    print("\n" + "=" * 70)
    print("HYBRID SEARCH RESULTS")
    print("=" * 70)

    for i, result in enumerate(hybrid_results, start=1):
        print("-" * 70)
        print(f"Результат #{i}")
        print(f"Источник: {result['source']}")
        print(f"Страница: {result['page']}")
        print(f"RRF Score: {result['rrf_score']:.4f}")
        print(result["text"][:500])
        print()


if __name__ == "__main__":
    main()