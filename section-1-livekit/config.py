"""Application configuration loaded from environment variables."""

import os
from collections.abc import Mapping
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, field_validator

REQUIRED_CONFIGURATION = {
    "LIVEKIT_URL": "LiveKit",
    "LIVEKIT_API_KEY": "LiveKit",
    "LIVEKIT_API_SECRET": "LiveKit",
    "DEEPGRAM_API_KEY": "Deepgram primary STT",
    "ASSEMBLYAI_API_KEY": "AssemblyAI fallback STT",
    "GOOGLE_API_KEY": "Google Gemini primary LLM",
    "GROQ_API_KEY": "Groq fallback LLM",
    "CARTESIA_API_KEY": "Cartesia primary TTS",
    "ELEVENLABS_API_KEY": "ElevenLabs fallback TTS",
    "ELEVENLABS_VOICE_ID": "ElevenLabs fallback TTS",
}


class Settings(BaseModel):
    """Required service configuration."""

    model_config = ConfigDict(populate_by_name=True)

    livekit_url: str = Field(alias="LIVEKIT_URL")
    livekit_api_key: str = Field(alias="LIVEKIT_API_KEY", repr=False)
    livekit_api_secret: str = Field(alias="LIVEKIT_API_SECRET", repr=False)
    deepgram_api_key: str = Field(alias="DEEPGRAM_API_KEY", repr=False)
    assemblyai_api_key: str = Field(alias="ASSEMBLYAI_API_KEY", repr=False)
    google_api_key: str = Field(alias="GOOGLE_API_KEY", repr=False)
    groq_api_key: str = Field(alias="GROQ_API_KEY", repr=False)
    cartesia_api_key: str = Field(alias="CARTESIA_API_KEY", repr=False)
    elevenlabs_api_key: str = Field(alias="ELEVENLABS_API_KEY", repr=False)
    elevenlabs_voice_id: str = Field(alias="ELEVENLABS_VOICE_ID")

    @field_validator("*", mode="before")
    @classmethod
    def strip_and_reject_blank(cls, value: object) -> object:
        """Strip strings and reject empty configuration values."""
        if isinstance(value, str):
            value = value.strip()
            if not value:
                raise ValueError("value must not be empty or whitespace-only")
        return value


def validate_provider_configuration(environment: Mapping[str, str]) -> None:
    """Raise a provider-specific error for missing startup configuration."""
    missing = [
        f"{name} ({provider})"
        for name, provider in REQUIRED_CONFIGURATION.items()
        if not environment.get(name, "").strip()
    ]
    if missing:
        raise ValueError("Missing required configuration: " + ", ".join(missing))


def get_settings() -> Settings:
    """Load the local .env file and validate required settings."""
    load_dotenv(dotenv_path=Path(__file__).with_name(".env"), override=False)
    validate_provider_configuration(os.environ)
    return Settings.model_validate(os.environ)
