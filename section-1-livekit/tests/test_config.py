"""Tests for environment-backed application settings."""

import config
import pytest
from pydantic import ValidationError


ENVIRONMENT = {
    "LIVEKIT_URL": "wss://example.livekit.cloud",
    "LIVEKIT_API_KEY": "livekit-key",
    "LIVEKIT_API_SECRET": "livekit-secret",
    "DEEPGRAM_API_KEY": "deepgram-key",
    "GOOGLE_API_KEY": "google-key",
    "CARTESIA_API_KEY": "cartesia-key",
}


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove relevant variables and disable access to a real .env file."""
    for name in ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(config, "load_dotenv", lambda **_: False)


def set_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Populate all required environment variables."""
    for name, value in ENVIRONMENT.items():
        monkeypatch.setenv(name, value)


def test_get_settings_with_all_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    set_environment(monkeypatch)

    settings = config.get_settings()

    assert settings.livekit_url == ENVIRONMENT["LIVEKIT_URL"]
    assert settings.livekit_api_key == ENVIRONMENT["LIVEKIT_API_KEY"]
    assert settings.livekit_api_secret == ENVIRONMENT["LIVEKIT_API_SECRET"]
    assert settings.deepgram_api_key == ENVIRONMENT["DEEPGRAM_API_KEY"]
    assert settings.google_api_key == ENVIRONMENT["GOOGLE_API_KEY"]
    assert settings.cartesia_api_key == ENVIRONMENT["CARTESIA_API_KEY"]


def test_get_settings_rejects_missing_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_environment(monkeypatch)
    monkeypatch.delenv("LIVEKIT_API_SECRET")

    with pytest.raises(ValidationError, match="LIVEKIT_API_SECRET"):
        config.get_settings()


def test_get_settings_rejects_whitespace_only_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_environment(monkeypatch)
    monkeypatch.setenv("GOOGLE_API_KEY", " \t ")

    with pytest.raises(ValidationError, match="whitespace-only"):
        config.get_settings()


def test_get_settings_strips_surrounding_whitespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_environment(monkeypatch)
    monkeypatch.setenv("LIVEKIT_URL", "  wss://example.livekit.cloud \t")

    settings = config.get_settings()

    assert settings.livekit_url == "wss://example.livekit.cloud"
