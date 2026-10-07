import re


def split_into_paragraphs(text: str) -> list[str]:
    """
    Разбивает текст на абзацы.
    """

    paragraphs = re.split(r"\n\s*\n", text)

    return [
        paragraph.strip()
        for paragraph in paragraphs
        if paragraph.strip()
    ]


def chunk_documents(
    documents: list[dict],
    chunk_size: int = 1000,
    overlap: int = 200,
) -> list[dict]:

    chunks = []

    for document in documents:
        paragraphs = split_into_paragraphs(document["text"])

        current_chunk = ""

        for paragraph in paragraphs:

            # Если добавление абзаца не превышает размер chunk
            if len(current_chunk) + len(paragraph) <= chunk_size:
                current_chunk += "\n\n" + paragraph

            else:
                if current_chunk.strip():
                    chunks.append(
                        {
                            "text": current_chunk.strip(),
                            "source": document["source"],
                            "page": document["page"],
                        }
                    )

                # overlap
                overlap_text = current_chunk[-overlap:]

                current_chunk = overlap_text + "\n\n" + paragraph

        # Добавляем последний chunk
        if current_chunk.strip():
            chunks.append(
                {
                    "text": current_chunk.strip(),
                    "source": document["source"],
                    "page": document["page"],
                }
            )

    return chunks