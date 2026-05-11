"""Regex patterns used across the RAG ingestion pipeline.

Centralised here so the rest of the pipeline holds business logic, not
regex literals. Each pattern carries a docstring with a concrete example so
future edits can be made without re-deriving intent.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Page sentinels
# ---------------------------------------------------------------------------

PAGE_SENTINEL_RE = re.compile(r"\[\[PAGE_(\d+)\]\]")
"""Markers we deliberately inject between paragraphs of every page so each
post-split chunk still references the page it came from. Capture group 1 is
the 1-based page number.

Matches:
    "Some prose...\\n\\n[[PAGE_3]]\\n\\nMore prose..."  ->  group(1) == "3"
"""


# ---------------------------------------------------------------------------
# Paragraph & blank-line handling
# ---------------------------------------------------------------------------

PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n")
"""Split markdown into paragraphs on a blank line (whitespace-only lines count
as blank too).

Example:
    "para one\\n\\npara two\\n   \\npara three"
    -> ["para one", "para two", "para three"]
"""

BLANK_LINE_RUN_RE = re.compile(r"\n{3,}")
"""A run of 3 or more newlines, used to collapse excess blank lines back to a
single blank line (\\n\\n).

Example:
    "a\\n\\n\\n\\nb"  ->  "a\\n\\nb"
"""


# ---------------------------------------------------------------------------
# Markdown tables
# ---------------------------------------------------------------------------

MD_TABLE_RE = re.compile(
    r"^\|.*\n\|[\s\-:|]+\|\n(?:\|.*\n?)+",
    flags=re.MULTILINE,
)
"""A GFM-style pipe table: header row, delimiter row of dashes/colons, then
one or more body rows. Used to *strip* tables out of the narrative pass since
``UnstructuredMarkdownLoader(mode='elements')`` already captures them.

Matches:
    | col1 | col2 |
    |------|------|
    | a    | b    |
"""


# ---------------------------------------------------------------------------
# Numbered-list noise
# ---------------------------------------------------------------------------

NUMBERED_LIST_PREFIX_RE = re.compile(r"^\d+\.\s+", flags=re.MULTILINE)
"""A markdown numbered-list marker ("1. ", "2. ", ...) at the very start of a
line (column 0). Stripped before formatting so the ordinal doesn't pollute
embeddings.

Matches:
    "1. first item"          ->  "first item"
    "2.  second"             ->  "second"
Does NOT match:
    "See step 1. It is..."   (not at line start)
    "   1. indented"         (leading whitespace — likely nested or in code)
"""


# ---------------------------------------------------------------------------
# Header metadata (key: value | key: value | ...)
# ---------------------------------------------------------------------------

HEADER_META_LINE_RE = re.compile(
    r"^[_*]?\s*"
    r"([^:\n|]+?:\s*[^|\n]+?)"  # first key: value
    r"(?:\s*\|\s*[^:\n|]+?:\s*[^|\n]+?){1,}"  # at least one more key: value
    r"\s*[_*]?\s*$",
    flags=re.MULTILINE,
)
"""A metadata header line: two or more pipe-separated ``key: value`` pairs,
optionally wrapped in ``_..._`` or ``*...*`` emphasis. Generic across
languages and field names.

Example match:
    "_Doküman: KB-04 | Versiyon: 1.0 | Tarih: 2024-08-15_"
"""

KV_PAIR_RE = re.compile(r"\s*([^:|]+?)\s*:\s*([^|]+?)\s*(?:\||$)")
"""Parses one ``key: value`` pair from a pipe-delimited line. Use with
``findall`` to extract every pair on the line.

Example:
    "Doküman: KB-04 | Versiyon: 1.0"
    findall -> [("Doküman", "KB-04"), ("Versiyon", "1.0")]
"""


# ---------------------------------------------------------------------------
# Fenced code blocks & their trailing explanations
# ---------------------------------------------------------------------------

FENCED_CODE_BLOCK_RE = re.compile(r"```[^\n]*\n(.*?)```", flags=re.DOTALL)
"""A fenced markdown code block: ``` ... ```. Non-greedy on body so adjacent
blocks are matched independently. Capture group 1 is the inner body.

Example:
    ```python
    def f(): pass
    ```
    -> group(1) == "def f(): pass\\n"
"""

TRAILING_EXPLANATION_RE = re.compile(
    r"^\s*(?:-->|#)[^\n]*(?:\n\s*(?:-->|#)[^\n]*)*",
    flags=re.MULTILINE,
)
"""A run of explanation lines beginning with ``-->`` or ``#`` that immediately
follow a fenced code block. Kept stuck to the preceding block when splitting
code-listing-heavy sections.

Example:
    --> note about the snippet above
    # another note
"""


# ---------------------------------------------------------------------------
# Log entries (Cisco syslog style)
# ---------------------------------------------------------------------------

LOG_LINE_BOUNDARY_RE = re.compile(r"(?=^%[A-Z_]+-\d+-[A-Z_]+:)", flags=re.MULTILINE)
"""Cisco-style syslog entry boundary: ``%FACILITY-SEVERITY-MNEMONIC:`` at
line start. Used as a zero-width lookahead split point so each entry becomes
its own chunk inside a multi-entry fenced log block.

Example boundaries:
    %BGP-5-ADJCHANGE: neighbor 10.0.0.1 Up
    %OSPF-6-RECV: Received Hello from 10.0.0.2
"""
