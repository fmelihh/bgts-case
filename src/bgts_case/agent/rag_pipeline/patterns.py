"""Regex patterns and tiny helpers for the RAG ingestion pipeline.

Kept small on purpose. If you find yourself adding a fourth regex for the
same concept, refactor first.
"""

from __future__ import annotations

import re

# Marker we inject so chunks can recover their originating page number.
# Example: "para\n\n[[PAGE_3]]\n\nmore" -> group(1) == "3"
PAGE_SENTINEL_RE = re.compile(r"\[\[PAGE_(\d+)\]\]")

# ATX heading line at column 0. Used both as a sentinel-injection anchor
# and (indirectly) to anchor section-aware logic.
# Matches: "## 2. DHCP Lease"  Does NOT match: "  ## indented", "#tag"
HEADING_LINE_RE = re.compile(r"^#{1,6}[ \t]+\S.*$", flags=re.MULTILINE)

# A GFM pipe table. Used only to strip embedded tables out of narrative
# text once UnstructuredMarkdownLoader has already captured them.
MD_TABLE_RE = re.compile(
    r"^\|.*\n\|[\s\-:|]+\|\n(?:\|.*\n?)+",
    flags=re.MULTILINE,
)

# Numbered-list marker at column 0: "1. ", "**2.** ", etc. Stripped before
# chunking so the ordinal doesn't pollute embeddings.
NUMBERED_LIST_PREFIX_RE = re.compile(
    r"^(?:\*\*\d+\.\*\*|\d+\.)\s+",
    flags=re.MULTILINE,
)

# An ATX heading whose body starts with a bold-wrapped numbered marker
# ("## **1.** ..."). pymupdf4llm sometimes emits these for list items;
# capture group 1 is the body without the leading "#"s so a substitution
# demotes the line back to a numbered list item.
PROMOTED_LIST_ITEM_RE = re.compile(
    r"^#{1,6}\s+(\*\*\d+\.\*\*\s+.*)$",
    flags=re.MULTILINE,
)

# A document-level metadata line: two or more pipe-separated "key: value"
# pairs, optionally wrapped in _..._ or *...* emphasis. Capture group 1 is
# the inner content without the wrapping emphasis.
# Example: "_Doc: KB-04 | Version: 1.0 | Date: 2024-08-15_"
HEADER_META_LINE_RE = re.compile(
    r"^[_*]?\s*((?:[^:\n|]+?:\s*[^|\n]+?)(?:\s*\|\s*[^:\n|]+?:\s*[^|\n]+?){1,})\s*[_*]?\s*$",
    flags=re.MULTILINE,
)

# A fenced code block (```...```). Non-greedy body so adjacent blocks
# match independently. Capture group 1 is the inner body.
FENCED_CODE_BLOCK_RE = re.compile(r"```[^\n]*\n(.*?)```", flags=re.DOTALL)

# Cisco-style log entry boundary inside a fenced block, used as a
# zero-width split point so multi-entry blocks become per-entry chunks.
# Example boundaries:
#     %BGP-5-ADJCHANGE: ...
#     %OSPF-6-RECV: ...
LOG_LINE_BOUNDARY_RE = re.compile(r"(?=^%[A-Z_]+-\d+-[A-Z_]+:)", flags=re.MULTILINE)


def parse_kv_pairs(line: str) -> dict[str, str]:
    """Parse a 'k1: v1 | k2: v2 | ...' line into a dict of slugified keys.

    Returns an empty dict if no pairs are found. Generic across languages.
    """
    pairs = re.findall(r"\s*([^:|]+?)\s*:\s*([^|]+?)\s*(?:\||$)", line)
    return {_slugify(k): v.strip() for k, v in pairs}


def _slugify(key: str) -> str:
    """snake_case-ify a key for safe use as a dict key."""
    k = re.sub(r"[^a-z0-9]+", "_", key.strip().lower(), flags=re.UNICODE)
    return k.strip("_") or "field"
