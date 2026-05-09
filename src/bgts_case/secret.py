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


secrets = Secrets()  # type: ignore[call-arg]
