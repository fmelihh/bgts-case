from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Secrets(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    fireworks_api_key: SecretStr
    database_url: str = (
        "postgresql+psycopg://postgres:postgres@localhost:5432/postgres"
    )
    mcp_server_url: str = "http://localhost:8765/mcp/"


secrets = Secrets()  # type: ignore[call-arg]
