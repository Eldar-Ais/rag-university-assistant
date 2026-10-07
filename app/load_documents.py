from pathlib import Path

from pypdf import PdfReader


def load_pdf(pdf_path: str) -> list[dict]:
    """
    Читает PDF и возвращает список страниц
    с текстом и метаданными.
    """

    reader = PdfReader(pdf_path)

    documents = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        text = text.strip()

        if not text:
            continue

        documents.append(
            {
                "text": text,
                "source": Path(pdf_path).name,
                "page": page_number,
            }
        )

    return documents