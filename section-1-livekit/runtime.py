"""Provider fallback chains and LiveKit session construction."""

import logging
from typing import Any

from livekit.agents import AgentSession, llm, stt, tts
from livekit.plugins import assemblyai, cartesia, deepgram, elevenlabs, google, groq

from config import Settings, get_settings

logger = logging.getLogger(__name__)


def _log_availability(provider_type: str, event: Any) -> None:
    """Log provider availability without credentials or conversation data."""
    provider = getattr(event, provider_type)
    logger.warning(
        "%s provider %s is %s",
        provider_type.upper(),
        provider.label,
        "available" if event.available else "unavailable",
    )


def create_primary_stt(settings: Settings) -> deepgram.STT:
    """Return the primary multilingual Deepgram recognizer."""
    return deepgram.STT(
        model="nova-3",
        language="multi",
        api_key=settings.deepgram_api_key,
    )


def create_fallback_stt(settings: Settings) -> assemblyai.STT:
    """Return the fallback multilingual AssemblyAI recognizer."""
    return assemblyai.STT(
        model="universal-streaming-multilingual",
        api_key=settings.assemblyai_api_key,
    )


def create_stt(settings: Settings) -> stt.FallbackAdapter:
    """Return Deepgram with AssemblyAI as the ordered STT fallback."""
    adapter = stt.FallbackAdapter(
        stt=[create_primary_stt(settings), create_fallback_stt(settings)],
        attempt_timeout=10.0,
        max_retry_per_stt=1,
        retry_interval=5.0,
    )
    adapter.on("stt_availability_changed")(
        lambda event: _log_availability("stt", event)
    )
    return adapter


def create_primary_llm(settings: Settings) -> google.LLM:
    """Return the primary Gemini text language model."""
    return google.LLM(
        model="gemini-3.5-flash-lite",
        thinking_config={"thinking_level": "minimal"},
        max_output_tokens=150,
        temperature=0.2,
        api_key=settings.google_api_key,
    )


def create_fallback_llm(settings: Settings) -> groq.LLM:
    """Return the fallback Groq model with function-calling support."""
    return groq.LLM(
        model="openai/gpt-oss-120b",
        temperature=0.2,
        max_completion_tokens=150,
        api_key=settings.groq_api_key,
    )


def create_llm(settings: Settings) -> llm.FallbackAdapter:
    """Return Gemini with Groq as the ordered LLM fallback."""
    adapter = llm.FallbackAdapter(
        llm=[create_primary_llm(settings), create_fallback_llm(settings)],
        attempt_timeout=10.0,
        max_retry_per_llm=0,
        retry_interval=0.5,
        retry_on_chunk_sent=False,
    )
    adapter.on("llm_availability_changed")(
        lambda event: _log_availability("llm", event)
    )
    return adapter


def create_primary_tts(settings: Settings) -> cartesia.TTS:
    """Return the primary multilingual Cartesia synthesizer."""
    return cartesia.TTS(
        model="sonic-3",
        language=None,
        api_key=settings.cartesia_api_key,
    )


def create_fallback_tts(settings: Settings) -> elevenlabs.TTS:
    """Return multilingual ElevenLabs fallback speech synthesis."""
    return elevenlabs.TTS(
        model="eleven_flash_v2_5",
        voice_id=settings.elevenlabs_voice_id,
        api_key=settings.elevenlabs_api_key,
    )


def create_tts(settings: Settings) -> tts.FallbackAdapter:
    """Return Cartesia with ElevenLabs as the ordered TTS fallback."""
    adapter = tts.FallbackAdapter(
        tts=[
            create_primary_tts(settings),
            create_fallback_tts(settings),
        ],
        max_retry_per_tts=1,
    )
    adapter.on("tts_availability_changed")(
        lambda event: _log_availability("tts", event)
    )
    return adapter


def create_agent_session(settings: Settings | None = None) -> AgentSession:
    """Return a new session configured with all three fallback chains."""
    resolved_settings = settings if settings is not None else get_settings()
    return AgentSession(
        stt=create_stt(resolved_settings),
        llm=create_llm(resolved_settings),
        tts=create_tts(resolved_settings),
    )
