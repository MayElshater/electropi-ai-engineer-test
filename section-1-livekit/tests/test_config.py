"""Tests for environment-backed application settings."""

import config
import pytest


ENVIRONMENT = {
    "LIVEKIT_URL": "wss://example.livekit.cloud",
    "LIVEKIT_API_KEY": "livekit-key",
    "LIVEKIT_API_SECRET": "livekit-secret",
    "DEEPGRAM_API_KEY": "deepgram-key",
    "ASSEMBLYAI_API_KEY": "assemblyai-key",
    "GOOGLE_API_KEY": "google-key",
    "GROQ_API_KEY": "groq-key",
    "CARTESIA_API_KEY": "cartesia-key",
    "ELEVENLABS_API_KEY": "elevenlabs-key",
    "ELEVENLABS_VOICE_ID": "elevenlabs-voice-id",
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
    assert settings.assemblyai_api_key == ENVIRONMENT["ASSEMBLYAI_API_KEY"]
    assert settings.google_api_key == ENVIRONMENT["GOOGLE_API_KEY"]
    assert settings.groq_api_key == ENVIRONMENT["GROQ_API_KEY"]
    assert settings.cartesia_api_key == ENVIRONMENT["CARTESIA_API_KEY"]
    assert settings.elevenlabs_api_key == ENVIRONMENT["ELEVENLABS_API_KEY"]
    assert settings.elevenlabs_voice_id == ENVIRONMENT["ELEVENLABS_VOICE_ID"]


@pytest.mark.parametrize(
    ("missing_name", "provider"),
    [
        ("ASSEMBLYAI_API_KEY", "AssemblyAI fallback STT"),
        ("GROQ_API_KEY", "Groq fallback LLM"),
        ("ELEVENLABS_API_KEY", "ElevenLabs fallback TTS"),
        ("ELEVENLABS_VOICE_ID", "ElevenLabs fallback TTS"),
    ],
)
def test_get_settings_reports_missing_provider_key(
    monkeypatch: pytest.MonkeyPatch,
    missing_name: str,
    provider: str,
) -> None:
    set_environment(monkeypatch)
    monkeypatch.setenv("LIVEKIT_API_SECRET", "secret-value-not-to-expose")
    monkeypatch.delenv(missing_name)

    with pytest.raises(ValueError) as error:
        config.get_settings()

    assert missing_name in str(error.value)
    assert provider in str(error.value)
    assert "secret-value-not-to-expose" not in str(error.value)


def test_get_settings_rejects_whitespace_only_value(monkeypatch: pytest.MonkeyPatch) -> None:
    set_environment(monkeypatch)
    monkeypatch.setenv("GOOGLE_API_KEY", " \t ")

    with pytest.raises(ValueError, match="GOOGLE_API_KEY.*Google Gemini primary LLM"):
        config.get_settings()


def test_get_settings_strips_surrounding_whitespace(monkeypatch: pytest.MonkeyPatch) -> None:
    set_environment(monkeypatch)
    monkeypatch.setenv("LIVEKIT_URL", "  wss://example.livekit.cloud \t")

    settings = config.get_settings()

    assert settings.livekit_url == "wss://example.livekit.cloud"
