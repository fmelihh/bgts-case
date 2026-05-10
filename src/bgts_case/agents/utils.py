"""Shared, general-purpose helpers for the agents package.

Anything in here must be reusable across pipeline steps and agents — keep
step-specific logic out.
"""

from __future__ import annotations

import hashlib
import uuid
from typing import Iterable, Iterator, TypeVar

# Deterministic UUID namespace for content-addressed point IDs.
POINT_ID_NAMESPACE = uuid.UUID("6f1c7a90-3b3e-4d8e-bf2b-7c9b1b3a4f01")


def content_hash(*, text: str) -> str:
    """SHA-256 hex digest of ``text`` (utf-8). Stable across processes."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def point_id(*, source_pdf_name: str, content_hash: str) -> str:
    """Stable uuid5 from ``(source_pdf_name, content_hash)``.

    Same content under the same PDF always yields the same UUID, so re-runs
    overwrite in place instead of creating duplicates.
    """
    return str(uuid.uuid5(POINT_ID_NAMESPACE, f"{source_pdf_name}|{content_hash}"))


T = TypeVar("T")


def batched(*, seq: Iterable[T], size: int) -> Iterator[list[T]]:
    """Yield successive ``size``-sized lists from ``seq``."""
    buf: list[T] = []
    for item in seq:
        buf.append(item)
        if len(buf) >= size:
            yield buf
            buf = []
    if buf:
        yield buf


__all__ = ["POINT_ID_NAMESPACE", "content_hash", "point_id", "batched"]
