"""Split loaded Documents into overlapping chunks on natural boundaries.

RecursiveCharacterTextSplitter tries paragraph → line → word → char separators
in order, so chunks break on whitespace rather than mid-word. Each chunk inherits
its source page's metadata, so the `page` number survives into the index.
"""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(docs: list[Document], size: int, overlap: int) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
    return splitter.split_documents(docs)
