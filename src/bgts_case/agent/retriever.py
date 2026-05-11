"""Hybrid retriever over the Qdrant collection populated by the RAG pipeline.

Runs dense + BM25-sparse prefetches and fuses with RRF server-side. Reranking
is intentionally out of scope for this iteration — the fused list is returned
as-is.
"""

from __future__ import annotations

from typing import Any

from loguru import logger
from openai import OpenAI
from pydantic import BaseModel, ConfigDict
from qdrant_client import models

from bgts_case.agents.interfaces import make_openai_client, make_qdrant_client
from bgts_case.agents.rag_pipeline.step2_qdrant_index import (
    COLLECTION_NAME,
    DENSE_DIM,
    DENSE_MODEL,
    DENSE_VECTOR_NAME,
    SPARSE_MODEL,
    SPARSE_VECTOR_NAME,
)

DEFAULT_LIMIT = 5
PREFETCH_LIMIT = 20

TEXT_CHUNK = "text_chunk"
TABLE_CHUNK = "table_chunk"
ALLOWED_CHUNK_TYPES = (TEXT_CHUNK, TABLE_CHUNK)


class RetrievedChunk(BaseModel):
    model_config = ConfigDict(frozen=True)

    score: float
    text: str
    doc_title: str | None = None
    page_number: int | None = None
    source_pdf_name: str | None = None
    chunk_type: str | None = None
    payload: dict[str, Any]


def _embed_query(*, openai: OpenAI, text: str) -> list[float]:
    resp = openai.embeddings.create(
        model=DENSE_MODEL,
        input=[text],
        dimensions=DENSE_DIM,
    )
    return resp.data[0].embedding


def _build_filter(
    *,
    source_pdf_name: str | None,
    chunk_type: str | None,
    page_number: int | None,
    doc_title: str | None,
) -> models.Filter | None:
    """Compose an ANDed Qdrant filter from the supplied metadata constraints."""
    conditions: list[models.Condition] = []
    if source_pdf_name:
        conditions.append(
            models.FieldCondition(
                key="source_pdf_name",
                match=models.MatchValue(value=source_pdf_name),
            )
        )
    if chunk_type:
        if chunk_type not in ALLOWED_CHUNK_TYPES:
            raise ValueError(
                f"chunk_type must be one of {ALLOWED_CHUNK_TYPES}, got {chunk_type!r}"
            )
        conditions.append(
            models.FieldCondition(
                key="type",
                match=models.MatchValue(value=chunk_type),
            )
        )
    if page_number is not None:
        conditions.append(
            models.FieldCondition(
                key="page_number",
                match=models.MatchValue(value=page_number),
            )
        )
    if doc_title:
        conditions.append(
            models.FieldCondition(
                key="doc_title",
                match=models.MatchValue(value=doc_title),
            )
        )
    if not conditions:
        return None
    return models.Filter(must=conditions)


def search(
    *,
    query: str,
    limit: int = DEFAULT_LIMIT,
    source_pdf_name: str | None = None,
    chunk_type: str | None = None,
    page_number: int | None = None,
    doc_title: str | None = None,
) -> list[RetrievedChunk]:
    """Hybrid search: dense + BM25-sparse, fused server-side with RRF.

    Optional metadata filters are ANDed together and applied to both
    prefetches so the dense/sparse candidate pools stay aligned.
    """
    query = query.strip()
    if not query:
        return []

    qdrant = make_qdrant_client()
    openai = make_openai_client()

    dense_vec = _embed_query(openai=openai, text=query)

    query_filter = _build_filter(
        source_pdf_name=source_pdf_name,
        chunk_type=chunk_type,
        page_number=page_number,
        doc_title=doc_title,
    )

    response = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            models.Prefetch(
                query=dense_vec,
                using=DENSE_VECTOR_NAME,
                limit=PREFETCH_LIMIT,
                filter=query_filter,
            ),
            models.Prefetch(
                query=models.Document(text=query, model=SPARSE_MODEL),
                using=SPARSE_VECTOR_NAME,
                limit=PREFETCH_LIMIT,
                filter=query_filter,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=limit,
        with_payload=True,
    )

    results: list[RetrievedChunk] = []
    for point in response.points:
        payload = point.payload or {}
        results.append(
            RetrievedChunk(
                score=point.score,
                text=payload.get("text", ""),
                doc_title=payload.get("doc_title"),
                page_number=payload.get("page_number"),
                source_pdf_name=payload.get("source_pdf_name"),
                chunk_type=payload.get("type"),
                payload=payload,
            )
        )
    logger.debug(
        f"retriever.search: query={query!r} -> {len(results)} hits "
        f"(limit={limit}, filters="
        f"source_pdf_name={source_pdf_name!r}, chunk_type={chunk_type!r}, "
        f"page_number={page_number!r}, doc_title={doc_title!r})"
    )
    return results
