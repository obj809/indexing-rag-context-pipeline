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
Dockerfile        # run-on-demand image for the one-shot index build (CPU torch + baked model)
docker-compose.yml         # `docker compose run --rm indexer` against the vector-db network
Dockerfile.dockerignore    # keeps .venv/.git/.env out of the build context (data/ is kept on purpose)
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

## Run in Docker

Same prerequisite as above (the database container up). The indexer is a
**run-on-demand one-shot** — it builds the index to completion and exits, so
there's no `up`:

```bash
docker compose run --rm indexer    # drop + rebuild the chunks table, then exit
```

Notes on how the image works:

- This is an **ops tool, not a service**: nothing depends on this image, and it
  runs `build_index.py` to completion and exits. Its only runtime peer is Postgres.
- It joins the vector-db repo's Compose network
  (`vector-db-rag-context-pipeline_default`, an `external` network) to reach
  Postgres as `db:5432`; `DATABASE_URL` is set in `docker-compose.yml`. It does
  **not** join `webnet` — the indexer makes no LLM call, so it needs only the
  database (no OpenAI key either).
- The source PDFs (`data/EPBC_Act_1999/pdf/`) are copied into the image, so a
  build is self-contained.
- The embedding model (`BAAI/bge-small-en-v1.5`) is baked into the image at build
  time, so the build needs no HuggingFace download at run time. If you change
  `EMBEDDING_MODEL`, rebuild with `--build-arg EMBEDDING_MODEL=...` (keep it in
  sync with the constant in `build_index.py`). The torch-install layer is written
  to match the engine/backend Dockerfiles so the multi-GB layer is shared.
- `.env` files are deliberately excluded from the image
  (`Dockerfile.dockerignore`), which also keeps the 1.1 GB `.venv` and `.git` out
  of the build context; `data/` is kept **in** on purpose so the PDFs ship in the
  image.

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
