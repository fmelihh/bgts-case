"""Qdrant hybrid indexer for the ingestion pipeline.

Dense  : Fireworks `accounts/fireworks/models/qwen3-embedding-8b` via OpenAI SDK.
Sparse : Qdrant server-side BM25 inference (`Qdrant/bm25`).

Idempotency:
- Point IDs are deterministic uuid5 from (source_pdf_name, sha256(content)),
  so the same chunk always lands on the same point and re-upserts overwrite
  in place (Qdrant `upsert` is idempotent on identical IDs).

Functional API: `upsert_chunks` constructs its own Qdrant + OpenAI clients
from the shared `secrets` object on each call. No persistent state is held
at the module level.
"""

from __future__ import annotations

import time

from langchain_core.documents import Document
from loguru import logger
from openai import OpenAI
from qdrant_client import QdrantClient, models

from bgts_case.agents.utils import batched, content_hash, point_id
from bgts_case.secret import secrets

FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"
DENSE_MODEL = "accounts/fireworks/models/qwen3-embedding-8b"
DENSE_DIM = 1024

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "bm25"
SPARSE_MODEL = "Qdrant/bm25"

COLLECTION_NAME = "bgts_kb"

# Fireworks /embeddings batch size. Conservative — bump if your tier allows.
EMBED_BATCH_SIZE = 32

# Qdrant upsert batch size.
UPSERT_BATCH_SIZE = 64


def _make_qdrant_client() -> QdrantClient:
    return QdrantClient(
        url=secrets.qdrant_url,
        api_key=(
            secrets.qdrant_api_key.get_secret_value()
            if secrets.qdrant_api_key
            else None
        ),
    )


def _make_openai_client() -> OpenAI:
    return OpenAI(
        base_url=FIREWORKS_BASE_URL,
        api_key=secrets.fireworks_api_key.get_secret_value(),
    )


def _ensure_collection(
    *,
    qdrant: QdrantClient,
    dense_dim: int = DENSE_DIM,
) -> None:
    """Create the collection if it doesn't exist. Idempotent."""
    if qdrant.collection_exists(COLLECTION_NAME):
        logger.debug(f"_ensure_collection: '{COLLECTION_NAME}' already exists")
        return

    logger.info(
        f"_ensure_collection: creating '{COLLECTION_NAME}' (dense_dim={dense_dim})"
    )
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            DENSE_VECTOR_NAME: models.VectorParams(
                size=dense_dim,
                distance=models.Distance.COSINE,
            ),
        },
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: models.SparseVectorParams(
                modifier=models.Modifier.IDF,
            ),
        },
    )
    qdrant.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="source_pdf_name",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )


def _embed_batch(*, openai: OpenAI, texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts via Fireworks (OpenAI-compatible)."""
    t0 = time.perf_counter()
    resp = openai.embeddings.create(
        model=DENSE_MODEL,
        input=texts,
    )
    elapsed = time.perf_counter() - t0
    logger.debug(
        f"_embed_batch: {len(texts)} texts in {elapsed:.2f}s "
        f"({len(texts) / elapsed:.1f} texts/s)"
    )
    return [d.embedding for d in resp.data]


def _run_upsert(
    *,
    qdrant: QdrantClient,
    openai: OpenAI,
    chunks: list[Document],
    dense_dim: int,
    embed_batch_size: int,
    upsert_batch_size: int,
) -> dict:
    """Workhorse: ensure collection, embed missing, upsert. Takes pre-built clients."""
    _ensure_collection(qdrant=qdrant, dense_dim=dense_dim)

    if not chunks:
        return {"total": 0, "upserted": 0}

    prepared = []
    for c in chunks:
        source = c.metadata.get("source_pdf_name") or c.metadata.get("source_pdf")
        if not source:
            logger.warning(
                "_run_upsert: chunk missing source_pdf_name; skipping. "
                "metadata keys: {}",
                list(c.metadata.keys()),
            )
            continue
        h = content_hash(text=c.page_content)
        pid = point_id(source_pdf_name=source, content_hash=h)
        prepared.append((pid, h, c))

    logger.info(f"_run_upsert: embedding+upserting {len(prepared)} chunks")

    upserted = 0
    for batch in batched(seq=prepared, size=embed_batch_size):
        texts = [c.page_content for _, _, c in batch]
        try:
            vectors = _embed_batch(openai=openai, texts=texts)
        except Exception as e:
            logger.exception(
                f"_run_upsert: embedding batch failed ({len(texts)} texts): {e}"
            )
            raise

        points = []
        for (pid, h, c), vec in zip(batch, vectors):
            payload = dict(c.metadata)
            payload["text"] = c.page_content
            payload["content_hash"] = h
            points.append(
                models.PointStruct(
                    id=pid,
                    vector={
                        DENSE_VECTOR_NAME: vec,
                        SPARSE_VECTOR_NAME: models.Document(
                            text=c.page_content,
                            model=SPARSE_MODEL,
                        ),
                    },
                    payload=payload,
                )
            )

        for upsert_batch in batched(seq=points, size=upsert_batch_size):
            qdrant.upsert(
                collection_name=COLLECTION_NAME,
                points=list(upsert_batch),
                wait=True,
            )
            upserted += len(upsert_batch)

    result = {
        "total": len(prepared),
        "upserted": upserted,
    }
    logger.success(f"_run_upsert: {result}")
    return result


def upsert_chunks(
    *,
    chunks: list[Document],
    dense_dim: int = DENSE_DIM,
    embed_batch_size: int = EMBED_BATCH_SIZE,
    upsert_batch_size: int = UPSERT_BATCH_SIZE,
) -> dict:
    """Upsert a batch of chunks. Builds clients fresh on each call from ``secrets``.

    Returns a dict with counters: ``total``, ``upserted``.
    """
    qdrant = _make_qdrant_client()
    openai = _make_openai_client()
    return _run_upsert(
        qdrant=qdrant,
        openai=openai,
        chunks=chunks,
        dense_dim=dense_dim,
        embed_batch_size=embed_batch_size,
        upsert_batch_size=upsert_batch_size,
    )


__all__ = ["upsert_chunks"]
