"""Shared client factories for external services used by the agents layer.

Builds fresh ``QdrantClient``, ``OpenAI`` (Fireworks-compatible), and
``ChatOpenAI`` instances from the project-wide ``secrets``. Centralised here
so RAG ingestion, the retriever, and the agent itself share the same
construction.
"""

from __future__ import annotations

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from openai import OpenAI
from qdrant_client import QdrantClient

from bgts_case.secret import secrets

MODEL_PROVIDER_BASE_URL = "https://api.fireworks.ai/inference/v1"
DEFAULT_CHAT_MODEL = "accounts/fireworks/models/deepseek-v4-pro"
FALLBACK_CHAT_MODEL = "accounts/fireworks/models/qwen3-vl-30b-a3b-instruct"
EVAL_MODEL = "accounts/fireworks/models/qwen3-8b"
# Dense embedding model used by both the RAG indexer (step2) and the
# eval pipeline. Shared so the retriever and the relevancy judge live
# in the same embedding space.
EVAL_EMBEDDING_MODEL = "accounts/fireworks/models/qwen3-embedding-8b"
EVAL_EMBEDDING_DIM = 1024


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
        base_url=MODEL_PROVIDER_BASE_URL,
        api_key=secrets.fireworks_api_key.get_secret_value(),
    )


def make_chat_model(
    *,
    model: str = DEFAULT_CHAT_MODEL,
    temperature: float = 0,
) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        base_url=MODEL_PROVIDER_BASE_URL,
        api_key=secrets.fireworks_api_key,
        temperature=temperature,
    )


def make_embeddings(
    *,
    model: str = EVAL_EMBEDDING_MODEL,
    dimensions: int = EVAL_EMBEDDING_DIM,
) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=model,
        dimensions=dimensions,
        base_url=MODEL_PROVIDER_BASE_URL,
        api_key=secrets.fireworks_api_key,
        check_embedding_ctx_length=False,
    )
