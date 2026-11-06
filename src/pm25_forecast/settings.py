"""Application settings, read from environment variables (and an optional local .env file)."""

from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# KAN-77: observations after this are affected by the KAN-74 collector bug and are kept out of the
# drift reference. Lift it with DRIFT_KAN74_CUTOFF= (blank) once KAN-74 is repaired.
KAN74_BUG_CUTOFF = datetime(2026, 10, 26, 0, 0, tzinfo=UTC)

# KAN-104: CHALLENGER_MODEL_VERSION may list several shadow challengers, comma-separated.
MAX_CHALLENGERS = 3


def parse_challenger_versions(raw: str | None) -> tuple[str, ...]:
    """Split a comma-separated challenger list: trimmed, in order, at most ``MAX_CHALLENGERS``.

    Blank or unset gives ``()``. An empty entry (``a,,b``), a duplicate or more than the cap is an
    error (``ValueError``; the message names the problem, not the values). A single value gives
    a one-item tuple, exactly the pre-KAN-104 behaviour.
    """
    if raw is None or not raw.strip():
        return ()
    versions = tuple(part.strip() for part in raw.split(","))
    if any(not v for v in versions):
        raise ValueError("CHALLENGER_MODEL_VERSION has an empty entry")
    if len(set(versions)) != len(versions):
        raise ValueError("CHALLENGER_MODEL_VERSION lists a model version twice")
    if len(versions) > MAX_CHALLENGERS:
        raise ValueError(f"CHALLENGER_MODEL_VERSION lists more than {MAX_CHALLENGERS} versions")
    return versions


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
    # Registered model version(s) scored in shadow mode by the hourly predict step (KAN-29). Unset =
    # no challenger. KAN-104: a comma-separated list of up to 3 versions (see
    # `challenger_model_versions`). Not secret: registry labels such as `kan15-retrospective-24h`.
    challenger_model_version: str | None = None

    # KAN-23: weekly retrain may move the production champion only when this is true. Default off;
    # a blank value (unset repo variable) is off; an unparseable value is a config error.
    auto_promote: bool = False

    # KAN-77: DRIFT_KAN74_CUTOFF, an ISO timestamp with a UTC offset; unset = default, blank = none.
    drift_kan74_cutoff: datetime | None = KAN74_BUG_CUTOFF

    @field_validator("drift_kan74_cutoff", mode="before")
    @classmethod
    def _cutoff_blank_disables(cls, v: object) -> object:
        return None if isinstance(v, str) and not v.strip() else v

    @field_validator("drift_kan74_cutoff", mode="after")
    @classmethod
    def _cutoff_is_aware(cls, v: datetime | None) -> datetime | None:
        if v is not None and v.utcoffset() is None:
            raise ValueError(
                "DRIFT_KAN74_CUTOFF needs a UTC offset, e.g. 2026-10-26T00:00:00+00:00"
            )
        return v

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
            versions = parse_challenger_versions(v)  # validates the list (KAN-104)
            return ",".join(versions) or None  # canonical: trimmed, single comma separators
        return v

    @field_validator("air4thai_ca_bundle", mode="before")
    @classmethod
    def _blank_bundle_is_unset(cls, v: object) -> object:
        return None if isinstance(v, str) and not v.strip() else v

    @property
    def challenger_model_versions(self) -> tuple[str, ...]:
        """The configured shadow challengers in the order listed; empty when none (KAN-104)."""
        return parse_challenger_versions(self.challenger_model_version)

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
