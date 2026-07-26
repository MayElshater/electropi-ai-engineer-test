"""Application configuration loaded from environment variables."""

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, field_validator


class Settings(BaseModel):
    """Required service configuration."""

    model_config = ConfigDict(populate_by_name=True)

    livekit_url: str = Field(alias="LIVEKIT_URL")
    livekit_api_key: str = Field(alias="LIVEKIT_API_KEY", repr=False)
    livekit_api_secret: str = Field(alias="LIVEKIT_API_SECRET", repr=False)
    deepgram_api_key: str = Field(alias="DEEPGRAM_API_KEY", repr=False)
    google_api_key: str = Field(alias="GOOGLE_API_KEY", repr=False)
    cartesia_api_key: str = Field(alias="CARTESIA_API_KEY", repr=False)

    @field_validator("*", mode="before")
    @classmethod
    def strip_and_reject_blank(cls, value: object) -> object:
        """Strip strings and reject empty configuration values."""
        if isinstance(value, str):
            value = value.strip()
            if not value:
                raise ValueError("value must not be empty or whitespace-only")
        return value


def get_settings() -> Settings:
    """Load the local .env file and validate required settings."""
    load_dotenv(dotenv_path=Path(__file__).with_name(".env"), override=False)
    return Settings.model_validate(os.environ)
