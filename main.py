from app.load_documents import load_pdf
from app.chunking import chunk_documents
from app.embeddings import EmbeddingModel
from app.retriever import Retriever


def main():
    # 1. Загружаем PDF
    documents = load_pdf("data/students_guide_enu.pdf")

    print(f"Загружено страниц: {len(documents)}")

    # 2. Делим документы на chunks
    chunks = chunk_documents(documents)

    print(f"Создано chunks: {len(chunks)}")

    # 3. Создаём embedding model
    embedding_model = EmbeddingModel()

    # 4. Получаем embeddings для всех chunks
    texts = [chunk["text"] for chunk in chunks]

    embeddings = embedding_model.encode(texts)

    print(f"Размер embeddings: {embeddings.shape}")

    # 5. Создаём retriever
    retriever = Retriever(
        embeddings=embeddings,
        chunks=chunks,
    )

    # 6. Вопрос пользователя
    query = input("\nВаш вопрос: ")

    # 7. Получаем embedding вопроса
    query_embedding = embedding_model.encode([query])[0]

    # 8. Ищем похожие chunks
    results = retriever.search(
        query_embedding=query_embedding,
        top_k=5,
    )

    # 9. Показываем результат
    print("\nНаиболее релевантные фрагменты:\n")

    for i, result in enumerate(results, start=1):
        print("-" * 70)
        print(f"Результат #{i}")
        print(f"Источник: {result['source']}")
        print(f"Страница: {result['page']}")
        print(f"Score: {result['score']:.4f}")
        print()
        print(result["text"][:500])
        print()


if __name__ == "__main__":
    main()