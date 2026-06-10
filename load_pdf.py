"""Load a PDF into a list of LangChain Documents (one per page).

Uses PyMuPDF4LLM, which renders each page to Markdown — preserving reading order,
table structure (as Markdown tables), and headings far better than a plain text
dump. Charts/figures embedded as images are not recovered (their data lives in
pixels). Each Document carries metadata["page"] (1-indexed, straight from
PyMuPDF4LLM's `page_number`) so the page flows through chunking into the `page`
column for citations.
"""

from pathlib import Path

import pymupdf4llm
from langchain_core.documents import Document


def load_pdf(path: Path) -> list[Document]:
    pages = pymupdf4llm.to_markdown(str(path), page_chunks=True, show_progress=False)
    return [
        Document(
            page_content=page["text"],
            metadata={"page": page["metadata"]["page_number"]},
        )
        for page in pages
    ]
