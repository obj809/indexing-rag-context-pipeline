# Indexing RAG Context Pipeline

The indexing pipeline for the [RAG Context Pipeline](../): loads the source PDF,
splits it into overlapping chunks, embeds each chunk with a local
sentence-transformers model, and writes the rows into the Postgres `chunks` table
(pgvector). Run once, or whenever the source PDF changes.

## Contents

```
build_index.py    # orchestrator: PDF → chunks → embeddings → Postgres (drop/recreate chunks)
load_pdf.py       # PDF → per-page Markdown Documents (PyMuPDF4LLM, metadata["page"])
chunker.py        # Documents → overlapping chunks (RecursiveCharacterTextSplitter)
embed.py          # chunks → numpy array of embeddings (normalized)
data/
└── net-zero-report.pdf    # source document
```

The `chunks` table schema is defined here, in `build_index.py` (it drops and
recreates the table each run, so a rebuild is idempotent and the `VECTOR(dim)`
always matches the current embedding model). The table records the
`embedding_model` name so the query side can't embed with a mismatched model.

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

To index a different PDF, drop it in `data/` and update `PDF_PATH` in
`build_index.py`.

## Required environment variables

- `DATABASE_URL` — e.g. `postgresql://rag:rag@localhost:5432/rag`, matching
  `vector-db-rag-context-pipeline/docker-compose.yml`. No OpenAI key is needed —
  embedding runs locally.

## Note on chart/figure extraction

PyMuPDF4LLM extracts text, tables, and headings well, but discards data labels
embedded in **charts/figures** (they live in pixels). The net-zero report is
chart-heavy, so some figures' numbers aren't in the index — recovering them would
need vision-based extraction. The affected eval questions are flagged
`"chart_dependent": true` in `engine-rag-context-pipeline/eval/dataset.jsonl`.
