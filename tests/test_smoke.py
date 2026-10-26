"""Smoke tests: the package imports and settings load with defaults."""

import importlib

import pytest

import pm25_forecast
from pm25_forecast.settings import Settings

SUBPACKAGES = ["ingest", "features", "models", "storage", "monitoring"]


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackages_import(name: str) -> None:
    module = importlib.import_module(f"pm25_forecast.{name}")
    assert module.__doc__


def test_package_has_version() -> None:
    assert pm25_forecast.__version__


def test_settings_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("DATABASE_URL", "OPENAQ_API_KEY", "ALERT_WEBHOOK_URL", "HEARTBEAT_URL"):
        monkeypatch.delenv(var, raising=False)
    settings = Settings(_env_file=None)
    assert settings.database_url is None
    assert settings.openaq_api_key is None
    assert settings.alert_webhook_url is None
    assert settings.heartbeat_url is None


def test_settings_read_env_and_hide_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAQ_API_KEY", "dummy-value")
    settings = Settings(_env_file=None)
    assert settings.openaq_api_key is not None
    assert settings.openaq_api_key.get_secret_value() == "dummy-value"
    assert "dummy-value" not in repr(settings)


def test_get_settings_is_cached() -> None:
    from pm25_forecast.settings import get_settings

    assert get_settings() is get_settings()
