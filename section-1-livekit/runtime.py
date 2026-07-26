"""Speech, language-model, and LiveKit session construction."""

from livekit.agents import AgentSession
from livekit.plugins import cartesia, deepgram, google

from config import Settings, get_settings


def create_stt(settings: Settings) -> deepgram.STT:
    """Return multilingual Deepgram speech recognition."""
    return deepgram.STT(
        model="nova-3",
        language="multi",
        api_key=settings.deepgram_api_key,
    )


def create_llm(settings: Settings) -> google.LLM:
    """Return a Gemini Flash text language model."""
    return google.LLM(
        model="gemini-2.5-flash",
        api_key=settings.google_api_key,
    )


def create_tts(settings: Settings) -> cartesia.TTS:
    """Return multilingual Cartesia text-to-speech synthesis."""
    return cartesia.TTS(
        model="sonic-3",
        language=None,
        api_key=settings.cartesia_api_key,
    )


def create_agent_session(settings: Settings | None = None) -> AgentSession:
    """Return a new LiveKit session configured with STT, LLM, and TTS."""
    resolved_settings = settings if settings is not None else get_settings()
    return AgentSession(
        stt=create_stt(resolved_settings),
        llm=create_llm(resolved_settings),
        tts=create_tts(resolved_settings),
    )