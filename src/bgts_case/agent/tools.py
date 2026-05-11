"""LangChain tools exposed to the agent.

Right now this only wires the Qdrant hybrid retriever (RAG knowledge base).
MCP tools are still loaded separately via :class:`MultiServerMCPClient`.
"""

from __future__ import annotations

from langchain_core.tools import tool
from loguru import logger

from bgts_case.agent.alerts import mock_slack_alert
from bgts_case.agent.retriever import (
    ALLOWED_CHUNK_TYPES,
    DEFAULT_LIMIT,
    RetrievedChunk,
    search,
)

MAX_LIMIT = 10


def _format_chunk(idx: int, chunk: RetrievedChunk) -> str:
    header_bits = []
    if chunk.doc_title:
        header_bits.append(f"doc={chunk.doc_title}")
    if chunk.source_pdf_name:
        header_bits.append(f"source={chunk.source_pdf_name}")
    if chunk.page_number is not None:
        header_bits.append(f"page={chunk.page_number}")
    if chunk.chunk_type:
        header_bits.append(f"type={chunk.chunk_type}")
    header_bits.append(f"score={chunk.score:.4f}")
    header = " | ".join(header_bits)
    return f"[{idx}] {header}\n{chunk.text}"


@tool(parse_docstring=True)
def retrieve_knowledge_base(
    query: str,
    limit: int = DEFAULT_LIMIT,
    source_pdf_name: str | None = None,
    chunk_type: str | None = None,
    page_number: int | None = None,
    doc_title: str | None = None,
) -> str:
    """Search the Turkish-language knowledge base and return relevant passages.

    Queries must be in Turkish (the corpus is Turkish). Translate the user's
    information need into Turkish before calling; keep acronyms like BGP,
    VPN, DNS, STP as-is. Start with no filters, then narrow only if needed.
    Each hit is prefixed with source PDF and page so you can cite it.

    Args:
        query: Short focused Turkish query (a phrase or one sentence).
        limit: Max passages to return (1-10, default 5).
        source_pdf_name: Restrict to one PDF by exact filename.
        chunk_type: Either "text_chunk" (prose) or "table_chunk" (table rows).
        page_number: Restrict to a single 1-based page.
        doc_title: Restrict to chunks under this exact heading.
    """
    limit = max(1, min(limit, MAX_LIMIT))
    if chunk_type is not None and chunk_type not in ALLOWED_CHUNK_TYPES:
        return (
            f"Invalid chunk_type {chunk_type!r}. "
            f"Allowed values: {list(ALLOWED_CHUNK_TYPES)}."
        )
    try:
        hits = search(
            query=query,
            limit=limit,
            source_pdf_name=source_pdf_name,
            chunk_type=chunk_type,
            page_number=page_number,
            doc_title=doc_title,
        )
    except Exception as e:
        logger.exception(f"retrieve_knowledge_base failed: {e}")
        mock_slack_alert(
            "Knowledge base retrieval failed.",
            channel="alerts",
            level="error",
            error_type=type(e).__name__,
            query=query,
            filters={
                "source_pdf_name": source_pdf_name,
                "chunk_type": chunk_type,
                "page_number": page_number,
                "doc_title": doc_title,
            },
        )
        return (
            "Knowledge base search is temporarily unavailable. "
            "Do not retry the same call immediately. Either answer from the "
            "conversation context if you can, or tell the user the KB lookup "
            "failed and ask them to rephrase or try again shortly."
        )

    if not hits:
        return (
            f"No results for query={query!r} with filters "
            f"source_pdf_name={source_pdf_name!r}, chunk_type={chunk_type!r}, "
            f"page_number={page_number!r}, doc_title={doc_title!r}. "
            "Try relaxing or removing filters."
        )
    return "\n\n".join(_format_chunk(i + 1, h) for i, h in enumerate(hits))
