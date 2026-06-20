"""Orchestrator: PDFs → chunks → embeddings → Postgres (pgvector).

Run:  python build_index.py     (from this repo root)
Deps: sentence-transformers, pymupdf4llm, langchain-text-splitters, numpy, psycopg, pgvector
Env:  DATABASE_URL  (this repo's .env first, then the umbrella .env as fallback)
Note: needs the Postgres+pgvector container running
      (cd ../vector-db-rag-context-pipeline && docker compose up -d).
      First run also downloads the embedding model (~130MB) to ~/.cache/huggingface/.

The corpus is the three EPBC Act 1999 volumes. Each volume restarts its page
numbers at 1, so pages alone are ambiguous across the set — every chunk carries
a `volume` label alongside its `page`, and the two together form the citation.
"""

import os
import re
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

from chunker import chunk_documents
from embed import embed
from load_pdf import load_pdf

REPO_ROOT = Path(__file__).resolve().parent     # indexing-rag-context-pipeline/
UMBRELLA = REPO_ROOT.parent                      # rag-context-pipeline/ (transitional fallback)
PDF_DIR = REPO_ROOT / "data" / "EPBC_Act_1999" / "pdf"   # the three Act volumes
CHUNK_SIZE = 1200          # characters
CHUNK_OVERLAP = 150
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


def volume_label(stem: str) -> str:
    # C2026C00116VOL01 → "Volume 1"; fall back to the raw stem if it doesn't match.
    m = re.search(r"VOL0*(\d+)", stem)
    return f"Volume {int(m.group(1))}" if m else stem


def main() -> None:
    # Precedence: real environment > repo-local .env > umbrella .env — an
    # explicitly set var (e.g. inline DATABASE_URL for a remote build) always wins.
    load_dotenv(REPO_ROOT / ".env")
    load_dotenv(UMBRELLA / ".env")
    dsn = os.environ["DATABASE_URL"]

    print(f"Loading embedding model {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    pdf_paths = sorted(PDF_DIR.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDFs found in {PDF_DIR}")
    docs = []
    for path in pdf_paths:
        vol = volume_label(path.stem)
        print(f"Loading {path.name} ({vol})...")
        pages = load_pdf(path)
        for d in pages:
            d.metadata["volume"] = vol     # rides through chunking into the `volume` column
        docs.extend(pages)
        print(f"  {len(pages)} pages")
    chunks = chunk_documents(docs, CHUNK_SIZE, CHUNK_OVERLAP)
    print(f"  {len(docs)} pages total, {len(chunks)} chunks")

    print("Embedding chunks...")
    embeddings = embed([c.page_content for c in chunks], model)
    dim = embeddings.shape[1]

    print(f"Writing {len(chunks)} rows to Postgres...")
    with psycopg.connect(dsn) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            # Drop + recreate so a rebuild is idempotent and the VECTOR(dim) always
            # matches the current model (changing EMBEDDING_MODEL just works on rebuild).
            cur.execute("DROP TABLE IF EXISTS chunks")
            cur.execute(
                f"""
                CREATE TABLE chunks (
                    id              SERIAL PRIMARY KEY,
                    content         TEXT NOT NULL,
                    embedding       VECTOR({dim}) NOT NULL,
                    embedding_model TEXT NOT NULL,
                    page            INTEGER,
                    volume          TEXT
                )
                """
            )
            cur.executemany(
                "INSERT INTO chunks (content, embedding, embedding_model, page, volume) VALUES (%s, %s, %s, %s, %s)",
                [
                    # metadata["page"] is already 1-indexed (PyMuPDF4LLM page_number);
                    # `volume` disambiguates pages that repeat across the three volumes.
                    (c.page_content, e, EMBEDDING_MODEL, c.metadata.get("page"), c.metadata.get("volume"))
                    for c, e in zip(chunks, embeddings)
                ],
            )
        conn.commit()
    print("Done.")


if __name__ == "__main__":
    main()
