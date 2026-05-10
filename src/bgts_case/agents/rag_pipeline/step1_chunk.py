"""Step 1 of the RAG ingestion pipeline: PDF -> chunked Documents.

Turns PDF knowledge base files into indexable LangChain Documents:

    PDF -> markdown (pymupdf4llm, per-page)
        -> tables  : UnstructuredMarkdownLoader(mode='elements')
                     -> KV-lines + RecursiveCharacterTextSplitter
        -> text    : MarkdownHeaderTextSplitter
                     -> RecursiveCharacterTextSplitter

The public entry point is :func:`process_pdf`. The orchestrator drives this
step and will compose later steps (embedding, indexing) around it.
"""

from __future__ import annotations

import re
import tempfile
import time
from io import StringIO
from pathlib import Path
from typing import cast

import pandas as pd
import pymupdf4llm
from bs4 import BeautifulSoup
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from loguru import logger

from bgts_case.agents.alerts import AlertFn, mock_slack_alert

CHUNK_SIZE = 1024
TEXT_CHUNK_OVERLAP = 100
TABLE_CHUNK_OVERLAP = 0
MIN_TABLE_CHUNK_SIZE = 256


_HEADER_SPLITTER = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
    strip_headers=True,
)

_TEXT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=TEXT_CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
    length_function=len,
)

_MD_TABLE_RE = re.compile(r"^\|.*\n\|[\s\-:|]+\|\n(?:\|.*\n?)+", flags=re.MULTILINE)


def pdf_to_page_markdowns(pdf_path: str | Path) -> list[dict]:
    """Render a PDF to per-page markdown dicts using pymupdf4llm.

    Calls ``pymupdf4llm.to_markdown(..., page_chunks=True)``, which returns
    one dict per page with the page's markdown under ``"text"`` plus pymupdf
    metadata. We process pages individually so downstream Documents can carry
    1-based ``page_number`` and ``total_pages`` in their metadata.
    """
    pdf_path = Path(pdf_path)
    pdf_size_kb = pdf_path.stat().st_size / 1024 if pdf_path.exists() else 0
    logger.debug(
        f"pdf_to_page_markdowns: rendering {pdf_path.name} ({pdf_size_kb:.1f} KB)"
    )
    t0 = time.perf_counter()
    pages = cast(list[dict], pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True))
    elapsed = time.perf_counter() - t0
    total_chars = sum(len(p.get("text", "")) for p in pages)
    logger.info(
        f"pdf_to_page_markdowns: {pdf_path.name} -> {len(pages)} pages, "
        f"{total_chars} chars markdown in {elapsed:.2f}s"
    )
    return pages


def _clean_heading(text: str) -> str:
    return text.strip().strip("*").strip()


def _derive_doc_title(meta: dict, fallback: str = "Unknown Document") -> str:
    """Pick the most specific heading available as the doc_title."""
    for key in ("h3", "h2", "h1"):
        if meta.get(key):
            return _clean_heading(meta[key])
    return fallback


def load_split_documents(
    markdown_path: str | Path,
    *,
    initial_doc_title: str = "Unknown Document",
) -> tuple[list[Document], list[Document], str]:
    """Split a markdown file into table and narrative Documents.

    - Tables come from UnstructuredMarkdownLoader(mode='elements') so we keep
      the rendered HTML in metadata['text_as_html'].
    - Narrative comes from MarkdownHeaderTextSplitter with embedded markdown
      tables stripped (they are already covered by the elements loader).

    ``initial_doc_title`` seeds the running section title and is also used as
    the fallback when a header chunk has no heading metadata — this lets
    page-by-page callers carry the running heading across page boundaries.
    The final running section title is returned as the third tuple element so
    the caller can feed it into the next page's call.
    """
    markdown_path = str(markdown_path)
    logger.debug(f"load_split_documents: loading elements from {markdown_path}")

    t0 = time.perf_counter()
    el_loader = UnstructuredMarkdownLoader(file_path=markdown_path, mode="elements")
    el_docs = el_loader.load()
    logger.debug(
        f"load_split_documents: Unstructured produced {len(el_docs)} elements "
        f"in {time.perf_counter() - t0:.2f}s"
    )

    category_counts: dict[str, int] = {}
    for d in el_docs:
        cat = d.metadata.get("category", "Unknown")
        category_counts[cat] = category_counts.get(cat, 0) + 1
    logger.debug(f"load_split_documents: element categories = {category_counts}")

    table_docs: list[Document] = []
    current_section_title = initial_doc_title
    title_count = 0

    for doc in el_docs:
        category = doc.metadata.get("category")

        if category == "Title":
            current_section_title = _clean_heading(doc.page_content)
            title_count += 1
            logger.trace(f"load_split_documents: section -> '{current_section_title}'")
            continue

        if category == "Table":
            doc.metadata["doc_title"] = current_section_title
            table_docs.append(doc)
            html = doc.metadata.get("text_as_html") or ""
            logger.debug(
                f"load_split_documents: table under '{current_section_title}' "
                f"({len(html)} chars HTML)"
            )

    raw_md = Path(markdown_path).read_text(encoding="utf-8")
    logger.debug(
        f"load_split_documents: read {len(raw_md)} chars of raw markdown for header split"
    )
    header_chunks = _HEADER_SPLITTER.split_text(raw_md)
    logger.debug(
        f"load_split_documents: header splitter produced {len(header_chunks)} chunks"
    )

    text_docs: list[Document] = []
    skipped_empty = 0
    for hc in header_chunks:
        body = hc.page_content.strip()
        if not body:
            skipped_empty += 1
            continue

        body = _MD_TABLE_RE.sub("\n", body)
        body = re.sub(r"\n{3,}", "\n\n", body).strip()
        if not body:
            skipped_empty += 1
            continue

        text_docs.append(
            Document(
                page_content=body,
                metadata={
                    **hc.metadata,
                    "doc_title": _derive_doc_title(
                        hc.metadata, fallback=initial_doc_title
                    ),
                    "source": markdown_path,
                },
            )
        )

    logger.info(
        f"load_split_documents: {markdown_path} -> {len(table_docs)} table docs, "
        f"{len(text_docs)} narrative sections "
        f"({title_count} titles seen, {skipped_empty} empty sections skipped, "
        f"final_section='{current_section_title}')"
    )
    return table_docs, text_docs, current_section_title


def is_simple_table(html_table: str) -> tuple[bool, int]:
    """A simple table has no other ``<table>`` nested inside it.

    Returns ``(is_simple, nested_table_count)`` so callers can include the
    count in alerts without re-parsing the HTML.
    """
    soup = BeautifulSoup(html_table, "html.parser")
    nested_count = len(soup.find_all("table"))

    if nested_count > 1:
        logger.warning(
            f"is_simple_table: nested table detected ({nested_count} <table> elements, "
            f"{len(html_table)} chars HTML)"
        )
        return False, nested_count

    logger.debug(f"is_simple_table: simple table ({len(html_table)} chars HTML)")
    return True, nested_count


def table_to_kv_lines(html_table: str) -> tuple[list[str], list[str]]:
    """Convert a simple 2D table into ``col1: v1, col2: v2, ...`` rows."""
    df = pd.read_html(StringIO(html_table))[0]

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            " / ".join(str(p) for p in col if str(p) != "nan").strip()
            for col in df.columns
        ]
    else:
        df.columns = [str(c) for c in df.columns]

    df = df.fillna("")
    cols = [str(c) for c in df.columns]
    lines = [", ".join(f"{c}: {row[c]}" for c in cols) for _, row in df.iterrows()]
    logger.debug(
        f"table_to_kv_lines: parsed {len(lines)} rows x {len(cols)} cols (cols={cols})"
    )
    return lines, cols


def split_kv_table(
    lines: list[str],
    cols: list[str],
    doc_title: str,
    extra_meta: dict | None = None,
) -> list[Document]:
    """Split KV table rows, prepending header context to every chunk."""
    extra_meta = extra_meta or {}

    header_ctx = f"[Document: {doc_title}]\n[Columns: {', '.join(cols)}]\n"
    body = "\n".join(lines)
    effective_size = max(CHUNK_SIZE - len(header_ctx), MIN_TABLE_CHUNK_SIZE)
    logger.debug(
        f"split_kv_table: doc_title='{doc_title}' rows={len(lines)} "
        f"body_chars={len(body)} header_ctx_chars={len(header_ctx)} "
        f"effective_chunk_size={effective_size}"
    )

    table_splitter = RecursiveCharacterTextSplitter(
        chunk_size=effective_size,
        chunk_overlap=TABLE_CHUNK_OVERLAP,
        separators=["\n", ", ", " ", ""],
        length_function=len,
    )

    raw_chunks = table_splitter.split_text(body)
    logger.debug(
        f"split_kv_table: doc_title='{doc_title}' produced {len(raw_chunks)} chunks"
    )

    return [
        Document(
            page_content=header_ctx + ch,
            metadata={
                "doc_title": doc_title,
                "type": "table_chunk",
                **extra_meta,
            },
        )
        for ch in raw_chunks
    ]


def chunk_documents(
    table_docs: list[Document],
    text_docs: list[Document],
    *,
    alert_fn: AlertFn = mock_slack_alert,
) -> list[Document]:
    """Turn raw table/text Documents into indexable chunks.

    - Simple tables -> KV chunks with header context, no overlap.
    - Tables with nested tables -> skipped (no chunk emitted); ``alert_fn`` is
      invoked with a human-readable message + structured context kwargs.
    - Narrative -> RecursiveCharacterTextSplitter with overlap.

    Each output has ``metadata['type']`` in {'table_chunk', 'text_chunk'}.
    """
    logger.debug(
        f"chunk_documents: chunking {len(text_docs)} text docs, {len(table_docs)} table docs"
    )
    chunks: list[Document] = []
    text_chunks_added = 0
    skipped_empty_text = 0

    ## Processing TEXT DOCUMENTS ##
    for tdoc in text_docs:
        if not tdoc.page_content.strip():
            skipped_empty_text += 1
            continue

        doc_title = tdoc.metadata.get("doc_title", "Unknown Document")
        pieces = _TEXT_SPLITTER.split_text(tdoc.page_content)
        logger.debug(
            f"chunk_documents: text doc_title='{doc_title}' "
            f"{len(tdoc.page_content)} chars -> {len(pieces)} chunks"
        )
        for piece in pieces:
            chunks.append(
                Document(
                    page_content=piece,
                    metadata={
                        **tdoc.metadata,
                        "doc_title": doc_title,
                        "type": "text_chunk",
                    },
                )
            )
            text_chunks_added += 1

    table_chunks_added = 0
    nested_tables = 0
    simple_tables = 0
    skipped_empty_tables = 0

    ## Processing TABLE DOCUMENTS ##
    for tdoc in table_docs:
        html_table = tdoc.metadata.get("text_as_html") or ""
        doc_title = tdoc.metadata.get("doc_title", "Unknown Document")
        source = tdoc.metadata.get("source")

        simple, nested_count = is_simple_table(html_table)
        if not simple:
            nested_tables += 1
            alert_fn(
                "Nested table detected; skipping (not chunked).",
                channel="rag-ingest",
                level="warning",
                doc_title=doc_title,
                source_pdf=tdoc.metadata.get("source_pdf"),
                source_pdf_name=tdoc.metadata.get("source_pdf_name"),
                nested_table_count=nested_count,
                html_chars=len(html_table),
            )
            continue

        simple_tables += 1
        lines, cols = table_to_kv_lines(html_table)
        if not lines:
            skipped_empty_tables += 1
            logger.warning(
                f"chunk_documents: empty simple table, skipping. doc_title='{doc_title}'"
            )
            continue

        extra_meta: dict = {"source_category": "Table"}
        if source:
            extra_meta["source"] = source
        for key in ("page_number", "total_pages", "source_pdf", "source_pdf_name"):
            val = tdoc.metadata.get(key)
            if val is not None:
                extra_meta[key] = val

        new_chunks = split_kv_table(
            lines=lines,
            cols=cols,
            doc_title=doc_title,
            extra_meta=extra_meta,
        )
        chunks.extend(new_chunks)
        table_chunks_added += len(new_chunks)

    total_chars = sum(len(c.page_content) for c in chunks)
    avg_chars = (total_chars // len(chunks)) if chunks else 0
    logger.info(
        f"chunk_documents: produced {len(chunks)} chunks "
        f"(text={text_chunks_added} from {len(text_docs)} docs, "
        f"table={table_chunks_added} from {simple_tables} simple+{nested_tables} nested, "
        f"skipped {skipped_empty_text} empty text, {skipped_empty_tables} empty tables) "
        f"| total_chars={total_chars} avg={avg_chars}"
    )
    return chunks


def process_pdf(
    pdf_path: str | Path,
    *,
    alert_fn: AlertFn = mock_slack_alert,
) -> list[Document]:
    """Full pipeline for a single PDF: PDF -> per-page markdown -> chunks.

    Pages are processed one at a time so every emitted Document carries
    ``page_number`` (1-based) and ``total_pages`` in its metadata. The running
    section title from :func:`load_split_documents` is threaded across page
    boundaries via ``initial_doc_title`` so content on a page that opens
    without its own heading inherits the heading from the previous page.

    Each page's markdown is written into a shared ``TemporaryDirectory`` so
    everything is removed when the ``with`` block exits — no manual unlink.

    ``alert`` is forwarded to :func:`chunk_documents` and fires for nested
    tables. Default is :func:`mock_slack_alert`.
    """
    pdf_path = Path(pdf_path)
    logger.info(f"process_pdf: START {pdf_path}")
    t_total = time.perf_counter()

    pages = pdf_to_page_markdowns(pdf_path)
    total_pages = len(pages)

    table_docs: list[Document] = []
    text_docs: list[Document] = []
    running_title = "Unknown Document"

    with tempfile.TemporaryDirectory(prefix="bgts_rag_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        for idx, page in enumerate(pages):
            page_number = idx + 1
            page_md = page.get("text", "")
            tmp_path = tmpdir_path / f"{pdf_path.stem}_p{page_number}.md"
            tmp_path.write_text(page_md, encoding="utf-8")
            logger.debug(
                f"process_pdf: wrote page {page_number}/{total_pages} markdown "
                f"to {tmp_path} ({len(page_md)} chars)"
            )

            page_table_docs, page_text_docs, running_title = load_split_documents(
                tmp_path, initial_doc_title=running_title
            )
            for doc in page_table_docs + page_text_docs:
                doc.metadata["page_number"] = page_number
                doc.metadata["total_pages"] = total_pages
                doc.metadata["source_pdf"] = str(pdf_path)
                doc.metadata["source_pdf_name"] = pdf_path.name

            table_docs.extend(page_table_docs)
            text_docs.extend(page_text_docs)

    chunks = chunk_documents(
        table_docs=table_docs, text_docs=text_docs, alert_fn=alert_fn
    )

    elapsed = time.perf_counter() - t_total
    logger.success(
        f"process_pdf: DONE {pdf_path.name} -> {len(chunks)} chunks "
        f"across {total_pages} pages in {elapsed:.2f}s"
    )
    return chunks


__all__ = ["process_pdf"]
