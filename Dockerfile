# Run-on-demand container for the one-shot index build:
#   docker compose run --rm indexer        # drop + rebuild the `chunks` table
#
# This is an ops tool, not a service: it runs build_index.py to completion and
# exits. Nothing depends on this image. Its only runtime peer is Postgres on the
# vector-db compose network (reached as db:5432). The source PDFs are committed
# to this repo (data/EPBC_Act_1999/pdf/) and copied into the image.
FROM python:3.12-slim

# Install torch from the CPU-only index first; PyPI's default wheels (amd64 AND
# aarch64) bundle the multi-GB CUDA/nvidia libraries, which this never uses.
# Pinned for a reproducible index. Keep this line byte-identical to the
# backend/engine Dockerfiles (same pin) so the multi-GB layer stays shared.
RUN pip install --no-cache-dir torch==2.12.0 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Bakes the embedding model into the image (~130 MB) so the build needs no
# HuggingFace download at run time. Must match EMBEDDING_MODEL in build_index.py 
ARG EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('$EMBEDDING_MODEL')"

COPY . /app/indexing-rag-context-pipeline/

WORKDIR /app/indexing-rag-context-pipeline
CMD ["python", "build_index.py"]
