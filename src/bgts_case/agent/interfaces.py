from __future__ import annotations

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from loguru import logger
from openai import OpenAI
from pydantic import SecretStr
from qdrant_client import QdrantClient

from bgts_case.secret import secrets

# --- Remote (Fireworks) configuration -----------------------------------
FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"
FIREWORKS_CHAT_MODEL = "accounts/fireworks/models/deepseek-v4-pro"
FALLBACK_CHAT_MODEL = "accounts/fireworks/models/qwen3-vl-30b-a3b-instruct"
EVAL_MODEL = "accounts/fireworks/models/qwen3-8b"
EVAL_EMBEDDING_MODEL = "accounts/fireworks/models/qwen3-embedding-8b"
EVAL_EMBEDDING_DIM = 1024


def _chat_model_config() -> tuple[str, str, SecretStr]:
    """Resolve (base_url, model, api_key) for the primary chat model.

    Honours ``secrets.run_as_a_local_model``. Fallback / eval / embeddings
    always stay on Fireworks.
    """
    if secrets.run_as_a_local_model:
        # Local OpenAI-compatible servers don't require auth, but the OpenAI
        # SDK insists on a non-empty key.
        return (
            secrets.local_model_base_url,
            secrets.local_model_name,
            SecretStr("not-needed"),
        )

    return (
        FIREWORKS_BASE_URL,
        FIREWORKS_CHAT_MODEL,
        secrets.fireworks_api_key,
    )


# Backwards-compatible exports.
MODEL_PROVIDER_BASE_URL = FIREWORKS_BASE_URL
DEFAULT_CHAT_MODEL = (
    secrets.local_model_name if secrets.run_as_a_local_model else FIREWORKS_CHAT_MODEL
)


def make_qdrant_client() -> QdrantClient:
    logger.info(f"make_qdrant_client: url={secrets.qdrant_url}")
    return QdrantClient(
        url=secrets.qdrant_url,
        api_key=(
            secrets.qdrant_api_key.get_secret_value()
            if secrets.qdrant_api_key
            else None
        ),
    )


def make_openai_client() -> OpenAI:
    """Raw OpenAI client for the primary chat model (local or Fireworks)."""
    base_url, model, api_key = _chat_model_config()
    logger.info(f"make_openai_client: base_url={base_url} model={model}")
    return OpenAI(base_url=base_url, api_key=api_key.get_secret_value())


def make_chat_model(
    *,
    model: str | None = None,
    temperature: float = 0,
) -> ChatOpenAI:
    """Build a chat model.

    - ``model=None`` → primary model, routed per ``secrets.run_as_a_local_model``.
    - Explicit ``model`` → always Fireworks (fallback / eval models stay remote).
    """
    if model is None:
        base_url, resolved_model, api_key = _chat_model_config()
        role = "local" if secrets.run_as_a_local_model else "fireworks"
    else:
        base_url = FIREWORKS_BASE_URL
        resolved_model = model
        api_key = secrets.fireworks_api_key
        role = "fireworks-explicit"

    logger.info(
        f"make_chat_model: role={role} model={resolved_model} "
        f"base_url={base_url} temperature={temperature}"
    )

    return ChatOpenAI(
        model=resolved_model,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
    )


def make_fallback_chat_model(*, temperature: float = 0) -> ChatOpenAI:
    """Explicit helper for the fallback model. Always Fireworks."""
    return make_chat_model(model=FALLBACK_CHAT_MODEL, temperature=temperature)


def make_embeddings(
    *,
    model: str = EVAL_EMBEDDING_MODEL,
    dimensions: int = EVAL_EMBEDDING_DIM,
) -> OpenAIEmbeddings:
    """Embeddings always come from Fireworks — they must live in the same
    vector space as the indexed corpus regardless of local/remote mode.
    """
    logger.info(
        f"make_embeddings: model={model} dimensions={dimensions} "
        f"base_url={FIREWORKS_BASE_URL}"
    )
    return OpenAIEmbeddings(
        model=model,
        dimensions=dimensions,
        base_url=FIREWORKS_BASE_URL,
        api_key=secrets.fireworks_api_key,
        check_embedding_ctx_length=False,
    )
