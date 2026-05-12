from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Secrets(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    fireworks_api_key: SecretStr
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: SecretStr | None = None
    database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/postgres"
    mcp_server_url: str = "http://localhost:8765/mcp/"
    mlflow_tracking_uri: str = "http://localhost:5000"

    # --- Local model routing -------------------------------------------------
    # When enabled, the primary chat model is routed to a local
    # OpenAI-compatible endpoint (Docker Model Runner with vLLM/vllm-metal
    # on Apple Silicon). Fallback, eval, and embeddings remain on Fireworks.
    run_as_a_local_model: bool = True
    local_model_base_url: str = "http://localhost:12434/engines/v1"
    local_model_name: str = "hf.co/mlx-community/Qwen2.5-7B-Instruct-4bit"


secrets = Secrets()  # type: ignore[call-arg]
