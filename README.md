# Indexing RAG Context Pipeline

The indexing pipeline for the [RAG Context Pipeline](../): loads the source PDFs,
splits them into overlapping chunks, embeds each chunk with a local
sentence-transformers model, and writes the rows into the Postgres `chunks` table
(pgvector). Run once, or whenever the source PDFs change.

The corpus is the three-volume Environment Protection and Biodiversity Conservation
Act 1999 (`data/EPBC_Act_1999/pdf/`). `build_index.py` globs the volume PDFs and tags
each chunk with its `volume` ("Volume 1"/`2`/`3`); since pages restart per volume,
volume + page together identify a chunk's location.

## Contents

```
build_index.py    # orchestrator: PDFs → chunks → embeddings → Postgres (drop/recreate chunks)
load_pdf.py       # PDF → per-page Markdown Documents (PyMuPDF4LLM, metadata["page"])
chunker.py        # Documents → overlapping chunks (RecursiveCharacterTextSplitter)
embed.py          # chunks → numpy array of embeddings (normalized)
data/
└── EPBC_Act_1999/
    └── pdf/
        ├── C2026C00116VOL01.pdf    # Volume 1
        ├── C2026C00116VOL02.pdf    # Volume 2
        └── C2026C00116VOL03.pdf    # Volume 3
```

The `chunks` table schema is defined here, in `build_index.py` (it drops and
recreates the table each run, so a rebuild is idempotent and the `VECTOR(dim)`
always matches the current embedding model). It records the `embedding_model` name
so the query side can't embed with a mismatched model, and a `page` + `volume` per
chunk for citations. A full build is ~7 min and yields ~2,700 chunks.

## Run

Prerequisites: Postgres + pgvector running first —
`cd ../vector-db-rag-context-pipeline && docker compose up -d`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env        # DATABASE_URL is preset to match the vector-db compose

python build_index.py       # first run downloads the ~130MB embedding model
```

`build_index.py` reads this repo's `.env` first, falling back to the umbrella
`.env`, so an existing umbrella `.env` keeps it running with no extra setup.

## Tuning

| Constant | Default | Notes |
|---|---|---|
| `CHUNK_SIZE` | 1200 chars | |
| `CHUNK_OVERLAP` | 150 chars | |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | changing it just works on rebuild — the table is recreated with the new `VECTOR(dim)` and model name |

To index a different set of PDFs, drop them in a folder and point `PDF_DIR` in
`build_index.py` at it (it globs `*.pdf` and derives each `volume` label from the
filename via `volume_label`).

## Required environment variables

- `DATABASE_URL` — e.g. `postgresql://rag:rag@localhost:5432/rag`, matching
  `vector-db-rag-context-pipeline/docker-compose.yml`. No OpenAI key is needed —
  embedding runs locally.

## Note on extraction

PyMuPDF4LLM extracts text, tables, and headings well — a good fit for a legal Act,
which has no chart data to lose. It also keeps the Act's
`Chapter / Part / Division / Section` hierarchy in a per-page running header, so the
structural context rides along inside each chunk. (PyMuPDF4LLM does drop data labels
embedded in charts/figures, but this corpus has none.)
