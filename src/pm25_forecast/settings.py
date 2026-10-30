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
    # Collector stations and the coordinates sent to Open-Meteo, as `ID=lat,lon;ID=lat,lon`
    # (KAN-9 will choose them). Not secret: ids and coordinates of public monitoring stations.
    station_locations: str = ""
    # Weather is fetched only when enabled: the schema has no table for it yet (KAN-18 note).
    collect_weather: bool = False
    # Where the collector writes gzip raw-response archives (git-ignored, see .gitignore).
    raw_archive_dir: str = "data/raw"

    # Optional PEM file with extra CA certificates for Air4Thai TLS (KAN-35); unset = defaults.
    air4thai_ca_bundle: Path | None = None
    # Registered model version scored in shadow mode by the hourly predict step (KAN-29). Unset =
    # no challenger. Not secret: a registry label such as `kan15-retrospective-24h`.
    challenger_model_version: str | None = None

    # KAN-23: weekly retrain may move the production champion only when this is true. Default off;
    # a blank value (unset repo variable) is off; an unparseable value is a config error.
    auto_promote: bool = False

    @field_validator("auto_promote", mode="before")
    @classmethod
    def _auto_promote_strict(cls, v: object) -> bool:
        """Only "true", "false" or blank (any case) is accepted; "1", "yes", "on" are errors."""
        if isinstance(v, bool):
            return v
        text = v.strip().lower() if isinstance(v, str) else None
        if text in ("", "false"):
            return False
        if text == "true":
            return True
        raise ValueError('AUTO_PROMOTE must be "true", "false" or blank')

    @field_validator("challenger_model_version", mode="before")
    @classmethod
    def _blank_challenger_is_unset(cls, v: object) -> object:
        if isinstance(v, str):
            v = v.strip()
            return v or None
        return v

    @field_validator("air4thai_ca_bundle", mode="before")
    @classmethod
    def _blank_bundle_is_unset(cls, v: object) -> object:
        return None if isinstance(v, str) and not v.strip() else v

    @property
    def station_id_list(self) -> list[str]:
        return [s.strip() for s in self.air4thai_station_ids.split(",") if s.strip()]

    @property
    def station_location_list(self) -> list[tuple[str, float, float]]:
        """Parse `station_locations`; a malformed entry raises ValueError (fail loudly)."""
        parsed: list[tuple[str, float, float]] = []
        for entry in (e.strip() for e in self.station_locations.split(";")):
            if not entry:
                continue
            try:
                station_id, coords = entry.split("=")
                lat, lon = coords.split(",")
                parsed.append((station_id.strip(), float(lat), float(lon)))
            except ValueError:
                raise ValueError(f"STATION_LOCATIONS entry {entry!r} is not ID=lat,lon") from None
        return parsed


@lru_cache
def get_settings() -> Settings:
    """Return the cached settings instance."""
    return Settings()
