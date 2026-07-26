"""Deepgram speech recognition and LiveKit session construction."""

from livekit.agents import AgentSession
from livekit.plugins import deepgram

from config import Settings, get_settings


def create_stt(settings: Settings) -> deepgram.STT:
    """Return multilingual Deepgram speech recognition."""
    return deepgram.STT(
        model="nova-3",
        language="multi",
        api_key=settings.deepgram_api_key,
    )


def create_agent_session(settings: Settings | None = None) -> AgentSession:
    """Return a new LiveKit session configured with Deepgram STT."""
    resolved_settings = settings if settings is not None else get_settings()
    return AgentSession(stt=create_stt(resolved_settings))