"""LangChain tools exposed to the agent.

Right now this only wires the Qdrant hybrid retriever (RAG knowledge base).
MCP tools are still loaded separately via :class:`MultiServerMCPClient`.
"""

from __future__ import annotations

from langchain_core.tools import tool
from loguru import logger

from bgts_case.agents.alerts import mock_slack_alert
from bgts_case.agents.retriever import (
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
    """Search the internal knowledge base (Qdrant hybrid index) for passages
    relevant to ``query`` and return them as a numbered, citation-ready list.

    The index combines dense embeddings (Fireworks qwen3-embedding-8b) with
    BM25 sparse vectors and fuses results with RRF. Use this whenever the
    user's question should be answered from indexed KB documents (runbooks,
    architecture notes, vendor PDFs).

    Each returned hit has a header line of the form
    ``[N] doc=<title> | source=<pdf> | page=<n> | type=<text|table> | score=<f>``
    followed by the passage text. Cite results by source + page in your
    final answer.

    Strategy tips:
    - Start with no filters and a short focused query; only add filters if
      the first pass returns irrelevant or too-broad results.
    - Filters are ANDed. Combining too many filters can return zero hits —
      drop the narrowest filter first when that happens.
    - For tabular/spec questions ("what's the power rating", "list ports"),
      pass ``chunk_type="table_chunk"``.
    - For prose explanations or procedures, pass ``chunk_type="text_chunk"``
      or leave it unset.

    Args:
        query: Natural-language search query. A focused phrase or single
            sentence works best (the index uses both dense and BM25, so
            keywords still help). Do not pass the user's entire turn —
            extract the actual information need.
        limit: Maximum number of passages to return. Defaults to 5 and is
            capped at 10. Use 3-5 for focused questions and 8-10 for broad
            overviews.
        source_pdf_name: Restrict results to one PDF by exact filename, for
            example ``"KB-10_Wireless_Altyapi.pdf"``. Use only when you
            already know which document the answer lives in; otherwise leave
            unset so all documents are searched.
        chunk_type: Restrict results to a chunk category. Must be either
            ``"text_chunk"`` (narrative/markdown sections) or
            ``"table_chunk"`` (rows extracted from PDF tables, rendered as
            comma-separated column-value pairs per row). Unset means both.
        page_number: Restrict results to a single 1-based page in the source
            PDF. Useful when the user references a specific page; otherwise
            leave unset.
        doc_title: Restrict results to chunks whose nearest heading equals
            this string exactly. Case- and whitespace-sensitive — use
            sparingly, since slight title variations will return no hits.
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
