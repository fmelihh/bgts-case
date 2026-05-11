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

PAGE_SENTINEL_RE = re.compile(r"\[\[PAGE_(\d+)\]\]")
_PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n")

# Matches a metadata header line with at least two pipe-separated "key: value" pairs.
# Generic across languages and field names; optionally wrapped in _..._ or *...* emphasis.
HEADER_META_LINE_RE = re.compile(
    r"^[_*]?\s*"
    r"([^:\n|]+?:\s*[^|\n]+?)"  # first key: value
    r"(?:\s*\|\s*[^:\n|]+?:\s*[^|\n]+?){1,}"  # at least one more key: value
    r"\s*[_*]?\s*$",
    flags=re.MULTILINE,
)

KV_PAIR_RE = re.compile(r"\s*([^:|]+?)\s*:\s*([^|]+?)\s*(?:\||$)")

MIN_TEXT_CHUNK_CHARS = 150

# A fenced code block: ``` ... ``` (non-greedy, multiline).
FENCED_CODE_BLOCK_RE = re.compile(r"```[^\n]*\n(.*?)```", flags=re.DOTALL)

# A run of lines that explain the preceding code block — lines starting with --> or #.
TRAILING_EXPLANATION_RE = re.compile(
    r"^\s*(?:-->|#)[^\n]*(?:\n\s*(?:-->|#)[^\n]*)*",
    flags=re.MULTILINE,
)

# Treat a text doc as "code-listing-dominated" when fenced blocks make up
# at least this fraction of its characters.
CODE_LISTING_RATIO_THRESHOLD = 0.4

# Cisco-style syslog entry boundary: ``%FACILITY-SEVERITY-MNEMONIC:`` at line
# start. Used to sub-split a single fenced log block that contains multiple
# independent log entries into one fenced chunk per entry.
LOG_LINE_BOUNDARY_RE = re.compile(r"(?=^%[A-Z_]+-\d+-[A-Z_]+:)", flags=re.MULTILINE)


def _extract_page_numbers(text: str) -> list[int]:
    """Return all 1-based page numbers referenced by [[PAGE_N]] sentinels in text."""
    return sorted({int(m.group(1)) for m in PAGE_SENTINEL_RE.finditer(text)})


def _interleave_page_sentinels(page_text: str, page_number: int) -> str:
    """Insert ``[[PAGE_N]]`` markers between every paragraph of a page's markdown.

    Plain-text markers survive ``MarkdownHeaderTextSplitter`` and
    ``UnstructuredMarkdownLoader`` (unlike HTML comments). Inserting between
    paragraphs — rather than only at page boundaries — guarantees every
    section body chunk contains at least one marker, so ``page_number`` can
    be recovered after splitting.
    """
    sentinel = f"[[PAGE_{page_number}]]"
    paragraphs = _PARAGRAPH_SPLIT_RE.split(page_text)
    parts: list[str] = []
    for p in paragraphs:
        stripped = p.strip()
        if not stripped:
            continue
        parts.append(stripped)
        parts.append(sentinel)
    return "\n\n".join(parts)


def _slugify_key(key: str) -> str:
    """Normalize a metadata key into snake_case for use as a dict key."""
    k = key.strip().lower()
    k = re.sub(r"[^a-z0-9]+", "_", k, flags=re.UNICODE)
    return k.strip("_") or "field"


def _extract_header_meta(combined_md: str) -> dict:
    """Extract document-level metadata from the first ``k: v | k: v | ...`` line found.

    Returns a dict of slugified keys to string values. Empty if no such line exists.
    Generic across documents: any pipe-delimited `key: value` line near the top works.
    """
    match = HEADER_META_LINE_RE.search(combined_md)
    if not match:
        return {}
    line = match.group(0).strip().strip("_*").strip()
    pairs = KV_PAIR_RE.findall(line)
    if not pairs:
        return {}
    meta = {_slugify_key(k): v.strip() for k, v in pairs}
    logger.debug(f"_extract_header_meta: parsed {len(meta)} fields: {meta}")
    return meta


def _is_header_meta_only(text: str) -> bool:
    """True if the chunk text is essentially just the document header metadata line."""
    stripped = text.strip().strip("_*").strip()
    if not stripped:
        return True
    if not HEADER_META_LINE_RE.search(stripped):
        return False
    leftover = HEADER_META_LINE_RE.sub("", stripped).strip()
    return len(leftover) < 20


def _is_code_listing_section(text: str) -> bool:
    """True if the section is dominated by fenced code blocks."""
    if "```" not in text:
        return False
    code_chars = sum(len(m.group(0)) for m in FENCED_CODE_BLOCK_RE.finditer(text))
    return code_chars / max(len(text), 1) >= CODE_LISTING_RATIO_THRESHOLD


def _further_split_log_block(block: str) -> list[str]:
    """If a fenced block holds multiple log entries separated by
    ``LOG_LINE_BOUNDARY_RE``, return one fenced block per entry.

    Otherwise return ``[block]`` unchanged.
    """
    if not LOG_LINE_BOUNDARY_RE.search(block):
        return [block]
    inner = re.sub(r"^```[^\n]*\n|\n```\s*$", "", block).strip()
    parts = LOG_LINE_BOUNDARY_RE.split(inner)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) <= 1:
        return [block]
    logger.debug(
        f"_further_split_log_block: split fenced block into {len(parts)} log entries"
    )
    return [f"```\n{p}\n```" for p in parts]


def _split_code_listing(text: str) -> list[str]:
    """Split a code-listing-dominated section into one chunk per fenced block.

    Each chunk includes the fenced block plus any immediately-following
    explanation lines (``-->`` or ``#`` style). Prose between blocks that
    isn't an explanation is not pulled in. Each fenced block is then run
    through :func:`_further_split_log_block` to break up multi-entry logs.
    """
    blocks: list[tuple[int, int]] = [
        (m.start(), m.end()) for m in FENCED_CODE_BLOCK_RE.finditer(text)
    ]
    if not blocks:
        return [text]

    pieces: list[str] = []
    for i, (start, end) in enumerate(blocks):
        next_start = blocks[i + 1][0] if i + 1 < len(blocks) else len(text)
        tail = text[end:next_start]

        # Capture explanation lines (--> or #) that immediately follow the block.
        explanation_match = TRAILING_EXPLANATION_RE.match(tail.lstrip("\n"))
        if explanation_match:
            offset = len(tail) - len(tail.lstrip("\n"))
            tail_kept = tail[: offset + explanation_match.end()]
        else:
            tail_kept = ""

        block_with_tail = (text[start:end] + tail_kept).strip()
        if not block_with_tail:
            continue

        for sub in _further_split_log_block(block_with_tail):
            sub = sub.strip()
            if sub:
                pieces.append(sub)

    return pieces or [text]


def _find_intro_text_doc(
    doc_title: str,
    text_docs: list[Document],
    max_chars: int = 300,
) -> Document | None:
    """Find a short text doc under the same doc_title to use as table intro context."""
    candidates = [
        td
        for td in text_docs
        if td.metadata.get("doc_title") == doc_title
        and 0 < len(td.page_content.strip()) <= max_chars
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda d: len(d.page_content.strip()))


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


def _looks_like_generic_header(columns) -> bool:
    """True if pandas assigned generic 0,1,2,... or '0','1','2',... column names."""
    for c in columns:
        if isinstance(c, int):
            continue
        if isinstance(c, str) and c.strip().isdigit():
            continue
        return False
    return True


def table_to_kv_lines(html_table: str) -> tuple[list[str], list[str]]:
    """Convert a simple 2D table into ``col1: v1, col2: v2, ...`` rows."""
    html_table = PAGE_SENTINEL_RE.sub("", html_table)
    df = pd.read_html(StringIO(html_table))[0]

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            " / ".join(str(p) for p in col if str(p) != "nan").strip()
            for col in df.columns
        ]
    else:
        df.columns = [str(c) for c in df.columns]

    # If pandas couldn't find <th>, the real headers ended up as row 0.
    # Promote row 0 to the column header, but only if every promoted value is non-empty.
    if _looks_like_generic_header(df.columns) and len(df) > 0:
        new_header = [str(v).strip() for v in df.iloc[0].tolist()]
        if all(h and h.lower() != "nan" for h in new_header):
            logger.debug(
                f"table_to_kv_lines: promoting first row to header. "
                f"Old: {list(df.columns)} -> New: {new_header}"
            )
            df.columns = new_header
            df = df.iloc[1:].reset_index(drop=True)

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
    intro: str | None = None,
) -> list[Document]:
    """Split KV table rows, prepending header context to every chunk."""
    extra_meta = extra_meta or {}

    intro_line = f"[Context: {intro.strip()}]\n" if intro else ""
    header_ctx = f"[Document: {doc_title}]\n{intro_line}[Columns: {', '.join(cols)}]\n"
    body = "\n".join(lines)
    effective_size = max(CHUNK_SIZE - len(header_ctx), MIN_TABLE_CHUNK_SIZE)
    logger.debug(
        f"split_kv_table: doc_title='{doc_title}' rows={len(lines)} "
        f"body_chars={len(body)} header_ctx_chars={len(header_ctx)} "
        f"effective_chunk_size={effective_size} has_intro={intro is not None}"
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

    ## Processing TABLE DOCUMENTS FIRST so we can mark intro text docs as consumed. ##
    consumed_text_doc_ids: set[int] = set()
    table_chunks_added = 0
    nested_tables = 0
    simple_tables = 0
    skipped_empty_tables = 0

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

        intro_doc = _find_intro_text_doc(doc_title, text_docs)
        intro_text = None
        if intro_doc is not None:
            intro_text = intro_doc.page_content.strip()
            consumed_text_doc_ids.add(id(intro_doc))
            logger.debug(
                f"chunk_documents: using intro for table '{doc_title}': "
                f"{intro_text[:80]}..."
            )

        extra_meta: dict = {"source_category": "Table"}
        if source:
            extra_meta["source"] = source
        for key in (
            "page_number",
            "total_pages",
            "source_pdf",
            "source_pdf_name",
            "header_meta",
        ):
            val = tdoc.metadata.get(key)
            if val is not None:
                extra_meta[key] = val

        new_chunks = split_kv_table(
            lines=lines,
            cols=cols,
            doc_title=doc_title,
            extra_meta=extra_meta,
            intro=intro_text,
        )
        chunks.extend(new_chunks)
        table_chunks_added += len(new_chunks)

    ## Processing TEXT DOCUMENTS, skipping the ones already used as intros. ##
    text_chunks_added = 0
    skipped_empty_text = 0
    skipped_short_text = 0
    skipped_header_meta_text = 0
    skipped_consumed_text = 0

    for tdoc in text_docs:
        if id(tdoc) in consumed_text_doc_ids:
            skipped_consumed_text += 1
            continue

        body = tdoc.page_content.strip()
        if not body:
            skipped_empty_text += 1
            continue
        if _is_header_meta_only(body):
            skipped_header_meta_text += 1
            logger.debug("chunk_documents: dropping header-meta-only chunk")
            continue

        doc_title = tdoc.metadata.get("doc_title", "Unknown Document")

        is_code_listing = _is_code_listing_section(tdoc.page_content)
        if is_code_listing:
            pieces = _split_code_listing(tdoc.page_content)
            logger.debug(
                f"chunk_documents: code-listing section '{doc_title}' "
                f"-> {len(pieces)} fenced-block chunks"
            )
        else:
            pieces = _TEXT_SPLITTER.split_text(tdoc.page_content)
            logger.debug(
                f"chunk_documents: text doc_title='{doc_title}' "
                f"{len(tdoc.page_content)} chars -> {len(pieces)} chunks"
            )

        for piece in pieces:
            piece_stripped = piece.strip()
            if not piece_stripped:
                continue
            if not is_code_listing and len(piece_stripped) < MIN_TEXT_CHUNK_CHARS:
                skipped_short_text += 1
                logger.debug(
                    f"chunk_documents: dropping short text chunk "
                    f"({len(piece_stripped)} chars) under doc_title='{doc_title}'"
                )
                continue
            page_content = (
                f"[Document: {doc_title}]\n{piece_stripped}"
                if is_code_listing
                else piece
            )
            chunks.append(
                Document(
                    page_content=page_content,
                    metadata={
                        **tdoc.metadata,
                        "doc_title": doc_title,
                        "type": "text_chunk",
                    },
                )
            )
            text_chunks_added += 1

    total_chars = sum(len(c.page_content) for c in chunks)
    avg_chars = (total_chars // len(chunks)) if chunks else 0
    logger.info(
        f"chunk_documents: produced {len(chunks)} chunks "
        f"(text={text_chunks_added} from {len(text_docs)} docs, "
        f"table={table_chunks_added} from {simple_tables} simple+{nested_tables} nested, "
        f"skipped {skipped_empty_text} empty + {skipped_short_text} short + "
        f"{skipped_header_meta_text} header-meta + {skipped_consumed_text} intro-consumed text, "
        f"{skipped_empty_tables} empty tables) "
        f"| total_chars={total_chars} avg={avg_chars}"
    )
    return chunks


def process_pdf(
    pdf_path: str | Path,
    *,
    alert_fn: AlertFn = mock_slack_alert,
) -> list[Document]:
    """Full pipeline for a single PDF: PDF -> per-page markdown -> chunks.

    All pages are merged into a single markdown document before splitting, so
    sections that span page boundaries are kept together. ``<!-- page: N -->``
    sentinels mark page boundaries; after splitting they are used to recover
    the ``page_number`` metadata and then stripped from the chunk content.

    ``alert`` is forwarded to :func:`chunk_documents` and fires for nested
    tables. Default is :func:`mock_slack_alert`.
    """
    pdf_path = Path(pdf_path)
    logger.info(f"process_pdf: START {pdf_path}")
    t_total = time.perf_counter()

    pages = pdf_to_page_markdowns(pdf_path)
    total_pages = len(pages)

    # Merge all pages into a single markdown document with [[PAGE_N]] sentinels
    # interleaved between every paragraph (not just at page boundaries) so that
    # every chunk recovers a page_number after splitting.
    parts = []
    for idx, page in enumerate(pages):
        page_number = idx + 1
        parts.append(_interleave_page_sentinels(page.get("text", ""), page_number))
    combined_md = "\n\n".join(parts)

    sentinel_count = len(PAGE_SENTINEL_RE.findall(combined_md))
    logger.debug(
        f"process_pdf: combined_md {len(combined_md)} chars, "
        f"{sentinel_count} page sentinels"
    )

    header_meta = _extract_header_meta(combined_md)

    running_title = "Unknown Document"
    with tempfile.TemporaryDirectory(prefix="bgts_rag_") as tmpdir:
        tmp_path = Path(tmpdir) / f"{pdf_path.stem}_combined.md"
        tmp_path.write_text(combined_md, encoding="utf-8")
        logger.debug(
            f"process_pdf: wrote combined markdown for {total_pages} pages "
            f"to {tmp_path} ({len(combined_md)} chars)"
        )

        table_docs, text_docs, running_title = load_split_documents(
            tmp_path, initial_doc_title=running_title
        )
        logger.debug(f"process_pdf: final running section title = '{running_title}'")

    pre_chunk_docs = table_docs + text_docs
    with_sentinel = sum(
        1 for d in pre_chunk_docs if PAGE_SENTINEL_RE.search(d.page_content)
    )
    logger.debug(
        f"process_pdf: {with_sentinel}/{len(pre_chunk_docs)} pre-chunk docs "
        f"contain at least one page sentinel"
    )

    # Recover page_number(s) per chunk, then strip sentinels from page_content.
    for doc in pre_chunk_docs:
        page_nums = _extract_page_numbers(doc.page_content)
        doc.page_content = PAGE_SENTINEL_RE.sub("", doc.page_content).strip()
        # Collapse blank lines left behind by sentinel removal.
        doc.page_content = re.sub(r"\n{3,}", "\n\n", doc.page_content).strip()
        doc.metadata["page_number"] = page_nums[0] if page_nums else None
        doc.metadata["total_pages"] = total_pages
        doc.metadata["source_pdf"] = str(pdf_path)
        doc.metadata["source_pdf_name"] = pdf_path.name
        if header_meta:
            doc.metadata["header_meta"] = header_meta

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
