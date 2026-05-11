"""Step 1 of the RAG ingestion pipeline: PDF -> chunked Documents.

Pipeline:
    pdf_to_page_markdowns  -> per-page markdown via pymupdf4llm
    _prepare_combined_md   -> normalise + inject page sentinels + merge
    load_split_documents   -> tables (Unstructured) + sections (Markdown headers)
    chunk_documents        -> emit indexable text_chunk / table_chunk Documents

Public entry point: :func:`process_pdf`.
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

from bgts_case.agent.alerts import AlertFn, mock_slack_alert
from bgts_case.agent.rag_pipeline.patterns import (
    FENCED_CODE_BLOCK_RE,
    HEADER_META_LINE_RE,
    HEADING_LINE_RE,
    LOG_LINE_BOUNDARY_RE,
    MD_TABLE_RE,
    NUMBERED_LIST_PREFIX_RE,
    PAGE_SENTINEL_RE,
    PROMOTED_LIST_ITEM_RE,
    parse_kv_pairs,
)

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

CHUNK_SIZE = 1024
TEXT_CHUNK_OVERLAP = 100
MIN_TABLE_CHUNK_SIZE = 256
MIN_TEXT_CHUNK_CHARS = 150
# Insert a [[PAGE_N]] sentinel at the next paragraph boundary once this many
# characters have passed without one. Below CHUNK_SIZE so every emitted chunk
# is guaranteed to contain at least one sentinel.
SENTINEL_STRIDE_CHARS = 800
# A section is treated as "code-listing-dominated" when fenced blocks make
# up at least this fraction of its characters.
CODE_LISTING_RATIO = 0.4


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


# ---------------------------------------------------------------------------
# Small text helpers
# ---------------------------------------------------------------------------


def _normalise_page(page_text: str) -> str:
    """Clean up a page's raw pymupdf4llm output before sentinel injection.

    Two surgical fixes for known quirks, in order:
      1. Demote 'promoted' list items ('## **1.** body' -> '**1.** body').
      2. Strip numbered-list prefixes once they're at column 0.
    Then collapse runs of blank lines.
    """
    text = PROMOTED_LIST_ITEM_RE.sub(r"\1", page_text)
    text = NUMBERED_LIST_PREFIX_RE.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _clean_heading(text: str) -> str:
    """Strip markdown emphasis (*..*, **..**) and collapse whitespace."""
    text = re.sub(r"\*+(.+?)\*+", r"\1", text)
    return " ".join(text.split()).strip()


def _inject_page_sentinels(page_text: str, page_number: int) -> str:
    """Inject ``[[PAGE_N]]`` sentinels sparsely so chunks can recover page_number.

    Strategy: one sentinel before every heading line (outside fenced code
    blocks), plus one at the very top of the page, plus one inside long
    heading-less prose runs every SENTINEL_STRIDE_CHARS. Tables, code
    blocks, and list items are never broken up — sentinels live only on
    heading or blank-line boundaries outside fenced code.
    """
    sentinel = f"[[PAGE_{page_number}]]"
    if not page_text.strip():
        return sentinel

    def _code_ranges(s: str) -> list[tuple[int, int]]:
        return [(m.start(), m.end()) for m in FENCED_CODE_BLOCK_RE.finditer(s)]

    def _in_any(pos: int, ranges: list[tuple[int, int]]) -> bool:
        return any(s <= pos < e for s, e in ranges)

    # Pass 1: sentinel before each heading, skipping '#' lines inside
    # fenced code blocks (shell comments, log markers, etc.).
    ranges = _code_ranges(page_text)
    pieces: list[str] = []
    last_end = 0
    for m in HEADING_LINE_RE.finditer(page_text):
        if _in_any(m.start(), ranges):
            continue
        pieces.append(page_text[last_end : m.start()])
        pieces.append(f"{sentinel}\n\n")
        last_end = m.start()
    pieces.append(page_text[last_end:])
    text = "".join(pieces)

    # Top-of-page sentinel — but only if not already there from the
    # heading pass (page starts with a heading at column 0).
    if not text.lstrip().startswith(sentinel):
        text = f"{sentinel}\n\n{text}"

    # Pass 2: stride-based top-up on blank-line boundaries, also skipping
    # boundaries that fall inside fenced code blocks.
    ranges = _code_ranges(text)
    pieces = []
    last_emitted = 0
    last_sentinel_pos = 0
    for m in re.finditer(r"\n\s*\n", text):
        boundary = m.end()
        if _in_any(boundary, ranges):
            continue
        if sentinel in text[last_sentinel_pos:boundary]:
            last_sentinel_pos = boundary
            continue
        if boundary - last_sentinel_pos >= SENTINEL_STRIDE_CHARS:
            pieces.append(text[last_emitted:boundary])
            pieces.append(f"{sentinel}\n\n")
            last_emitted = boundary
            last_sentinel_pos = boundary
    pieces.append(text[last_emitted:])
    return "".join(pieces)


def _strip_sentinels(text: str) -> str:
    """Remove all [[PAGE_N]] markers and collapse the leftover blank lines."""
    text = PAGE_SENTINEL_RE.sub("", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _page_numbers_in(text: str) -> list[int]:
    """Return sorted unique 1-based page numbers referenced in text."""
    return sorted({int(m.group(1)) for m in PAGE_SENTINEL_RE.finditer(text)})


# ---------------------------------------------------------------------------
# Header-meta (document-level: version / owner / date / etc.)
# ---------------------------------------------------------------------------


def _extract_header_meta(combined_md: str) -> dict:
    """Return the parsed `k: v | k: v | ...` header line, or {} if absent."""
    m = HEADER_META_LINE_RE.search(combined_md)
    if not m:
        return {}
    meta = parse_kv_pairs(m.group(1))
    if meta:
        logger.debug(f"header_meta: {meta}")
    return meta


def _is_header_meta_only(text: str) -> bool:
    """True if a chunk is essentially just the header-meta line."""
    stripped = text.strip().strip("_*").strip()
    if not stripped:
        return True
    if not HEADER_META_LINE_RE.search(stripped):
        return False
    return len(HEADER_META_LINE_RE.sub("", stripped).strip()) < 20


# ---------------------------------------------------------------------------
# Code-listing splitting (logs, command examples, etc.)
# ---------------------------------------------------------------------------


def _is_code_listing(text: str) -> bool:
    """True when fenced blocks dominate the section."""
    if "```" not in text:
        return False
    code_chars = sum(len(m.group(0)) for m in FENCED_CODE_BLOCK_RE.finditer(text))
    return code_chars / max(len(text), 1) >= CODE_LISTING_RATIO


def _split_code_listing(text: str) -> list[str]:
    """One chunk per fenced block, plus immediate `-->`/`#` explanation lines.

    Multi-entry blocks (Cisco-style logs) are further split per entry.
    """
    blocks = [(m.start(), m.end()) for m in FENCED_CODE_BLOCK_RE.finditer(text)]
    if not blocks:
        return [text]

    pieces: list[str] = []
    for i, (start, end) in enumerate(blocks):
        next_start = blocks[i + 1][0] if i + 1 < len(blocks) else len(text)
        tail = text[end:next_start]
        explanation = re.match(
            r"^\s*(?:-->|#)[^\n]*(?:\n\s*(?:-->|#)[^\n]*)*",
            tail.lstrip("\n"),
            flags=re.MULTILINE,
        )
        if explanation:
            offset = len(tail) - len(tail.lstrip("\n"))
            tail_kept = tail[: offset + explanation.end()]
        else:
            tail_kept = ""

        block = (text[start:end] + tail_kept).strip()
        if not block:
            continue
        pieces.extend(_split_multi_entry_log_block(block))
    return pieces or [text]


def _split_multi_entry_log_block(block: str) -> list[str]:
    """If a fenced block holds multiple `%XYZ-N-FOO:` log entries, split them."""
    if not LOG_LINE_BOUNDARY_RE.search(block):
        return [block]
    inner = re.sub(r"^```[^\n]*\n|\n```\s*$", "", block).strip()
    parts = [p.strip() for p in LOG_LINE_BOUNDARY_RE.split(inner) if p.strip()]
    if len(parts) <= 1:
        return [block]
    return [f"```\n{p}\n```" for p in parts]


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def _is_simple_table(html_table: str) -> tuple[bool, int]:
    """True iff the table contains no nested <table>."""
    nested = len(BeautifulSoup(html_table, "html.parser").find_all("table"))
    return nested <= 1, nested


def _table_to_kv_lines(html_table: str) -> tuple[list[str], list[str]]:
    """Parse an HTML table into ['col: v, col: v, ...'] rows + column list.

    If pandas couldn't find <th> and the column names look like 0,1,2,...,
    we promote the first data row to header (only if every promoted value
    is non-empty).
    """
    html_table = PAGE_SENTINEL_RE.sub("", html_table)
    df = pd.read_html(StringIO(html_table))[0]

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            " / ".join(str(p) for p in col if str(p) != "nan").strip()
            for col in df.columns
        ]
    else:
        df.columns = [str(c) for c in df.columns]

    looks_generic = all(str(c).strip().isdigit() for c in df.columns)
    if looks_generic and len(df) > 0:
        new_header = [str(v).strip() for v in df.iloc[0].tolist()]
        if all(h and h.lower() != "nan" for h in new_header):
            logger.debug(f"promote first row to header: {new_header}")
            df.columns = new_header
            df = df.iloc[1:].reset_index(drop=True)

    df = df.fillna("")
    cols = [str(c) for c in df.columns]
    lines = [", ".join(f"{c}: {row[c]}" for c in cols) for _, row in df.iterrows()]
    return lines, cols


def _split_kv_table(
    lines: list[str],
    cols: list[str],
    doc_title: str,
    intro: str | None,
    extra_meta: dict,
) -> list[Document]:
    """Emit table chunks with a [Document]/[Context]/[Columns] header context."""
    intro_line = f"[Context: {intro.strip()}]\n" if intro else ""
    header_ctx = f"[Document: {doc_title}]\n{intro_line}[Columns: {', '.join(cols)}]\n"
    effective_size = max(CHUNK_SIZE - len(header_ctx), MIN_TABLE_CHUNK_SIZE)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=effective_size,
        chunk_overlap=0,
        separators=["\n", ", ", " ", ""],
        length_function=len,
    )
    return [
        Document(
            page_content=header_ctx + chunk,
            metadata={"doc_title": doc_title, "type": "table_chunk", **extra_meta},
        )
        for chunk in splitter.split_text("\n".join(lines))
    ]


# ---------------------------------------------------------------------------
# Stage 1: PDF -> combined markdown
# ---------------------------------------------------------------------------


def _pdf_to_pages(pdf_path: Path) -> list[dict]:
    """Render PDF to per-page markdown dicts (pymupdf4llm)."""
    t0 = time.perf_counter()
    pages = cast(list[dict], pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True))
    logger.info(
        f"pdf_to_pages: {pdf_path.name} -> {len(pages)} pages "
        f"in {time.perf_counter() - t0:.2f}s"
    )
    return pages


def _prepare_combined_md(pages: list[dict]) -> str:
    """Normalise each page and merge into one document with page sentinels."""
    parts: list[str] = []
    for idx, page in enumerate(pages):
        page_text = _normalise_page(page.get("text", ""))
        parts.append(_inject_page_sentinels(page_text, idx + 1))
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Stage 2: combined markdown -> raw table/text Documents
# ---------------------------------------------------------------------------


def _section_title_from(meta: dict, fallback: str) -> str:
    """Pick the most specific h1/h2/h3 heading available as the section title."""
    for key in ("h3", "h2", "h1"):
        if meta.get(key):
            return _clean_heading(meta[key])
    return fallback


def load_split_documents(
    markdown_path: str | Path,
) -> tuple[list[Document], list[Document]]:
    """Split a markdown file into table Documents and narrative Documents."""
    markdown_path = str(markdown_path)
    raw_md = Path(markdown_path).read_text(encoding="utf-8")

    # Tables: use Unstructured so we keep the rendered HTML for pandas.
    elements = UnstructuredMarkdownLoader(
        file_path=markdown_path, mode="elements"
    ).load()
    logger.debug(
        f"unstructured: {len(elements)} elements "
        f"({sum(1 for d in elements if d.metadata.get('category') == 'Table')} tables)"
    )

    # Unstructured doesn't reliably surface [[PAGE_N]] sentinels inside
    # Table elements' page_content, so we resolve each Table's page by
    # matching its source position in raw_md against sentinel offsets.
    sentinel_index = [
        (m.start(), int(m.group(1))) for m in PAGE_SENTINEL_RE.finditer(raw_md)
    ]
    table_offsets = [m.start() for m in MD_TABLE_RE.finditer(raw_md)]

    def _page_for_offset(off: int) -> int | None:
        page = None
        for pos, num in sentinel_index:
            if pos > off:
                break
            page = num
        return page

    table_docs: list[Document] = []
    current_title = "Unknown Document"
    table_seen = 0
    for el in elements:
        cat = el.metadata.get("category")
        if cat == "Title":
            current_title = _clean_heading(el.page_content)
        elif cat == "Table":
            el.metadata["doc_title"] = current_title
            if table_seen < len(table_offsets):
                el.metadata["page_number"] = _page_for_offset(table_offsets[table_seen])
            else:
                logger.warning(
                    f"unstructured table #{table_seen + 1} has no matching "
                    f"raw_md table offset; page_number left to recovery"
                )
            table_seen += 1
            table_docs.append(el)

    # Narrative: header-aware split, then strip any embedded markdown tables
    # (Unstructured already captured those).
    text_docs: list[Document] = []
    for hc in _HEADER_SPLITTER.split_text(raw_md):
        body = MD_TABLE_RE.sub("\n", hc.page_content).strip()
        if not body:
            continue
        text_docs.append(
            Document(
                page_content=body,
                metadata={
                    **hc.metadata,
                    "doc_title": _section_title_from(hc.metadata, current_title),
                    "source": markdown_path,
                },
            )
        )

    logger.info(
        f"load_split_documents: {len(table_docs)} tables, {len(text_docs)} sections"
    )
    return table_docs, text_docs


# ---------------------------------------------------------------------------
# Stage 3: raw Documents -> indexable chunks
# ---------------------------------------------------------------------------


def _find_intro_text_doc(
    doc_title: str, text_docs: list[Document], max_chars: int = 300
) -> Document | None:
    """A short text doc under the same doc_title, used as a table's intro."""
    candidates = [
        td
        for td in text_docs
        if td.metadata.get("doc_title") == doc_title
        and 0 < len(td.page_content.strip()) <= max_chars
    ]
    return min(candidates, key=lambda d: len(d.page_content), default=None)


def chunk_documents(
    table_docs: list[Document],
    text_docs: list[Document],
    *,
    alert_fn: AlertFn = mock_slack_alert,
) -> list[Document]:
    """Emit indexable chunks. Tables run first so their intros can be reused."""
    chunks: list[Document] = []
    consumed_intros: set[int] = set()

    # --- Tables ---
    for tdoc in table_docs:
        html = tdoc.metadata.get("text_as_html") or ""
        doc_title = tdoc.metadata.get("doc_title", "Unknown Document")
        simple, nested = _is_simple_table(html)
        if not simple:
            alert_fn(
                "Nested table detected; skipping.",
                channel="rag-ingest",
                level="warning",
                doc_title=doc_title,
                source_pdf=tdoc.metadata.get("source_pdf"),
                nested_table_count=nested,
            )
            continue

        lines, cols = _table_to_kv_lines(html)
        if not lines:
            logger.warning(f"empty table under '{doc_title}', skipping")
            continue

        intro_doc = _find_intro_text_doc(doc_title, text_docs)
        intro = None
        if intro_doc is not None:
            intro = intro_doc.page_content.strip()
            consumed_intros.add(id(intro_doc))

        extra = {"source_category": "Table", **_carry_metadata(tdoc)}
        chunks.extend(
            _split_kv_table(lines, cols, doc_title, intro=intro, extra_meta=extra)
        )

    # --- Text ---
    for tdoc in text_docs:
        if id(tdoc) in consumed_intros:
            continue
        body = tdoc.page_content.strip()
        if not body or _is_header_meta_only(body):
            continue

        doc_title = tdoc.metadata.get("doc_title", "Unknown Document")
        is_code = _is_code_listing(body)
        pieces = (
            _split_code_listing(body) if is_code else _TEXT_SPLITTER.split_text(body)
        )

        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue
            if not is_code and len(piece) < MIN_TEXT_CHUNK_CHARS:
                continue
            content = f"[Document: {doc_title}]\n{piece}" if is_code else piece
            chunks.append(
                Document(
                    page_content=content,
                    metadata={
                        **tdoc.metadata,
                        "doc_title": doc_title,
                        "type": "text_chunk",
                    },
                )
            )

    logger.info(f"chunk_documents: produced {len(chunks)} chunks")
    return chunks


def _carry_metadata(doc: Document) -> dict:
    """Pick the subset of metadata that table chunks need to inherit."""
    return {
        k: v
        for k, v in doc.metadata.items()
        if k
        in {
            "page_number",
            "total_pages",
            "source",
            "source_pdf",
            "source_pdf_name",
            "header_meta",
        }
        and v is not None
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def process_pdf(
    pdf_path: str | Path,
    *,
    alert_fn: AlertFn = mock_slack_alert,
) -> list[Document]:
    """Full pipeline: PDF -> chunks. Pages are merged before splitting so
    sections that span page boundaries stay together; [[PAGE_N]] sentinels
    let us recover page_number per chunk after splitting.
    """
    pdf_path = Path(pdf_path)
    logger.info(f"process_pdf: START {pdf_path}")
    t0 = time.perf_counter()

    pages = _pdf_to_pages(pdf_path)
    combined_md = _prepare_combined_md(pages)
    header_meta = _extract_header_meta(combined_md)
    logger.debug(
        f"combined_md: {len(combined_md)} chars, "
        f"{len(PAGE_SENTINEL_RE.findall(combined_md))} sentinels"
    )

    with tempfile.TemporaryDirectory(prefix="bgts_rag_") as tmpdir:
        tmp_path = Path(tmpdir) / f"{pdf_path.stem}_combined.md"
        tmp_path.write_text(combined_md, encoding="utf-8")
        table_docs, text_docs = load_split_documents(tmp_path)

    # Single pass: recover page_number, strip sentinels, attach common metadata.
    for doc in table_docs + text_docs:
        page_nums = _page_numbers_in(doc.page_content)
        doc.page_content = _strip_sentinels(doc.page_content)
        # Tables get page_number set directly in load_split_documents via raw_md
        # offset lookup. Only fall back to sentinel recovery (text_docs path)
        # when it wasn't already resolved.
        if doc.metadata.get("page_number") is None:
            doc.metadata["page_number"] = page_nums[0] if page_nums else None
        doc.metadata["total_pages"] = len(pages)
        doc.metadata["source_pdf"] = str(pdf_path)
        doc.metadata["source_pdf_name"] = pdf_path.name
        if header_meta:
            doc.metadata["header_meta"] = header_meta

    chunks = chunk_documents(table_docs, text_docs, alert_fn=alert_fn)
    logger.success(
        f"process_pdf: DONE {pdf_path.name} -> {len(chunks)} chunks "
        f"in {time.perf_counter() - t0:.2f}s"
    )
    return chunks


__all__ = ["process_pdf"]
