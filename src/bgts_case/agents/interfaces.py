"""Shared client factories for external services used by the agents layer.

Builds fresh ``QdrantClient`` and ``OpenAI`` (Fireworks-compatible) clients
from the project-wide ``secrets``. Centralised here so RAG ingestion, the
retriever, and any future agent tooling share the same construction.
"""

from __future__ import annotations

from openai import OpenAI
from qdrant_client import QdrantClient

from bgts_case.secret import secrets

FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"


def make_qdrant_client() -> QdrantClient:
    return QdrantClient(
        url=secrets.qdrant_url,
        api_key=(
            secrets.qdrant_api_key.get_secret_value()
            if secrets.qdrant_api_key
            else None
        ),
    )


def make_openai_client() -> OpenAI:
    return OpenAI(
        base_url=FIREWORKS_BASE_URL,
        api_key=secrets.fireworks_api_key.get_secret_value(),
    )
