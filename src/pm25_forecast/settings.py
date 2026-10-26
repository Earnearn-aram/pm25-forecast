"""Application settings, read from environment variables (and an optional local .env file)."""

from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr, field_validator
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

    # Optional PEM file with extra CA certificates for Air4Thai TLS (KAN-35); unset = defaults.
    air4thai_ca_bundle: Path | None = None

    @field_validator("air4thai_ca_bundle", mode="before")
    @classmethod
    def _blank_bundle_is_unset(cls, v: object) -> object:
        return None if isinstance(v, str) and not v.strip() else v

    @property
    def station_id_list(self) -> list[str]:
        return [s.strip() for s in self.air4thai_station_ids.split(",") if s.strip()]


@lru_cache
def get_settings() -> Settings:
    """Return the cached settings instance."""
    return Settings()
