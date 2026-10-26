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
    # Collector stations and the coordinates sent to Open-Meteo, as `ID=lat,lon;ID=lat,lon`
    # (KAN-9 will choose them). Not secret: ids and coordinates of public monitoring stations.
    station_locations: str = ""
    # Where the collector writes gzip raw-response archives (git-ignored, see .gitignore).
    raw_archive_dir: str = "data/raw"

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
