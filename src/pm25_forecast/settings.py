"""Application settings, read from environment variables (and an optional local .env file)."""

from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration. Every field is optional so the package imports without secrets."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: SecretStr | None = None
    openaq_api_key: SecretStr | None = None
    alert_webhook_url: SecretStr | None = None
    heartbeat_url: SecretStr | None = None
    # Comma-separated Air4Thai station ids (KAN-9 will choose them); empty keeps all stations.
    air4thai_station_ids: str = ""

    @property
    def station_id_list(self) -> list[str]:
        return [s.strip() for s in self.air4thai_station_ids.split(",") if s.strip()]


@lru_cache
def get_settings() -> Settings:
    """Return the cached settings instance."""
    return Settings()
