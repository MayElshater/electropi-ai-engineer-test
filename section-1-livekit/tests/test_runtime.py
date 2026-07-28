"""Offline tests for provider fallbacks and the LiveKit runtime lifecycle."""

import asyncio
import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest
from livekit.agents import AgentServer, cli

import config
import runtime as runtime_module
from config import Settings


def make_settings() -> Settings:
    """Return complete deterministic settings for runtime tests."""
    return Settings(
        livekit_url="wss://example.livekit.cloud",
        livekit_api_key="livekit-key",
        livekit_api_secret="livekit-secret",
        deepgram_api_key="deepgram-test-key",
        assemblyai_api_key="assemblyai-test-key",
        google_api_key="google-test-key",
        groq_api_key="groq-test-key",
        cartesia_api_key="cartesia-test-key",
        elevenlabs_api_key="elevenlabs-test-key",
        elevenlabs_voice_id="elevenlabs-voice-id",
    )


def import_app() -> ModuleType:
    """Import a fresh app module for isolated import behavior."""
    sys.modules.pop("app", None)
    return importlib.import_module("app")


@pytest.mark.parametrize(
    ("factory_name", "provider_name", "expected"),
    [
        (
            "create_primary_stt",
            "deepgram",
            {"model": "nova-3", "language": "multi", "api_key": "deepgram-test-key"},
        ),
        (
            "create_fallback_stt",
            "assemblyai",
            {
                "model": "universal-streaming-multilingual",
                "api_key": "assemblyai-test-key",
            },
        ),
        (
            "create_primary_llm",
            "google",
            {
                "model": "gemini-3.5-flash-lite",
                "thinking_config": {"thinking_level": "minimal"},
                "max_output_tokens": 150,
                "temperature": 0.2,
                "api_key": "google-test-key",
            },
        ),
        (
            "create_fallback_llm",
            "groq",
            {
                "model": "openai/gpt-oss-120b",
                "temperature": 0.2,
                "max_completion_tokens": 150,
                "api_key": "groq-test-key",
            },
        ),
        (
            "create_primary_tts",
            "cartesia",
            {"model": "sonic-3", "language": None, "api_key": "cartesia-test-key"},
        ),
        (
            "create_fallback_tts",
            "elevenlabs",
            {
                "model": "eleven_flash_v2_5",
                "voice_id": "elevenlabs-voice-id",
                "api_key": "elevenlabs-test-key",
            },
        ),
    ],
)


def test_provider_factory_configuration(
    monkeypatch: pytest.MonkeyPatch,
    factory_name: str,
    provider_name: str,
    expected: dict[str, object],
) -> None:
    received: dict[str, object] = {}
    created = object()
    provider = getattr(runtime_module, provider_name)
    constructor_name = "STT" if "stt" in factory_name else "LLM" if "llm" in factory_name else "TTS"

    def fake_constructor(**kwargs: object) -> object:
        received.update(kwargs)
        return created

    monkeypatch.setattr(provider, constructor_name, fake_constructor)

    assert getattr(runtime_module, factory_name)(make_settings()) is created
    assert received == expected


@pytest.mark.parametrize(
    ("kind", "factory_name", "primary_name", "fallback_name", "expected_options"),
    [
        (
            "stt",
            "create_stt",
            "create_primary_stt",
            "create_fallback_stt",
            {"attempt_timeout": 10.0, "max_retry_per_stt": 1, "retry_interval": 5.0},
        ),
        (
            "llm",
            "create_llm",
            "create_primary_llm",
            "create_fallback_llm",
            {
                "attempt_timeout": 10.0,
                "max_retry_per_llm": 0,
                "retry_interval": 0.5,
                "retry_on_chunk_sent": False,
            },
        ),
        (
            "tts",
            "create_tts",
            "create_primary_tts",
            "create_fallback_tts",
            {"max_retry_per_tts": 1},
        ),
    ],
)
def test_fallback_chain_order_and_policy(
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    factory_name: str,
    primary_name: str,
    fallback_name: str,
    expected_options: dict[str, object],
) -> None:
    primary = object()
    fallback = object()
    received: dict[str, object] = {}
    events: list[str] = []

    class FakeAdapter:
        def __init__(self, **kwargs: object) -> None:
            received.update(kwargs)

        def on(self, event_name: str):
            events.append(event_name)
            return lambda callback: callback

    monkeypatch.setattr(runtime_module, primary_name, lambda settings: primary)
    monkeypatch.setattr(runtime_module, fallback_name, lambda settings, *args: fallback)
    monkeypatch.setattr(getattr(runtime_module, kind), "FallbackAdapter", FakeAdapter)

    result = getattr(runtime_module, factory_name)(make_settings())

    assert isinstance(result, FakeAdapter)
    assert received.pop(kind) == [primary, fallback]
    assert received == expected_options
    assert events == [f"{kind}_availability_changed"]


def test_create_agent_session_receives_only_fallback_adapters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = make_settings()
    adapters = {"stt": object(), "llm": object(), "tts": object()}
    received: dict[str, object] = {}
    session = object()

    for kind, adapter in adapters.items():
        monkeypatch.setattr(runtime_module, f"create_{kind}", lambda value, item=adapter: item)

    def fake_session(**kwargs: object) -> object:
        received.update(kwargs)
        return session

    monkeypatch.setattr(runtime_module, "AgentSession", fake_session)

    assert runtime_module.create_agent_session(settings) is session
    assert received == adapters


def test_zero_argument_session_loads_settings_once(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = make_settings()
    calls: list[str] = []
    monkeypatch.setattr(runtime_module, "get_settings", lambda: calls.append("settings") or settings)
    monkeypatch.setattr(runtime_module, "create_stt", lambda value: object())
    monkeypatch.setattr(runtime_module, "create_llm", lambda value: object())
    monkeypatch.setattr(runtime_module, "create_tts", lambda value: object())
    monkeypatch.setattr(runtime_module, "AgentSession", lambda **kwargs: object())

    runtime_module.create_agent_session()

    assert calls == ["settings"]


def test_import_runtime_has_no_factory_side_effects(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    sys.modules.pop("runtime", None)

    with monkeypatch.context() as patch:
        patch.setattr(config, "get_settings", lambda: calls.append("settings"))
        importlib.import_module("runtime")

    sys.modules["runtime"] = runtime_module
    assert calls == []


def test_provider_keys_are_not_exposed_by_settings_repr() -> None:
    settings = make_settings()
    representation = repr(settings)

    for key in (
        settings.deepgram_api_key,
        settings.assemblyai_api_key,
        settings.google_api_key,
        settings.groq_api_key,
        settings.cartesia_api_key,
        settings.elevenlabs_api_key,
    ):
        assert key not in representation


def test_import_exposes_server_without_running_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[object] = []
    monkeypatch.setattr(cli, "run_app", lambda server: calls.append(server))

    app = import_app()

    assert isinstance(app.server, AgentServer)
    assert calls == []


def test_initial_greeting_has_required_focus() -> None:
    app = import_app()
    greeting = app.INITIAL_GREETING_INSTRUCTION.lower()

    assert greeting.strip()
    assert "arabic" in greeting
    assert "english" in greeting
    assert "full name" in greeting
    assert "phone_number" not in greeting
    assert "email" not in greeting
    assert "address" not in greeting
    assert "preferred_contact_method" not in greeting


def test_server_has_registered_rtc_entrypoint() -> None:
    app = import_app()

    async def another_entrypoint(ctx) -> None:
        return None

    with pytest.raises(RuntimeError, match="only one rtc_session"):
        app.server.rtc_session(another_entrypoint)


def test_entrypoint_preserves_lifecycle(monkeypatch: pytest.MonkeyPatch) -> None:
    app = import_app()
    events: list[object] = []
    room = object()
    settings = object()
    intake_data = object()
    agent = SimpleNamespace(intake_data=intake_data)

    class FakeSession:
        def on(self, event_name: str):
            def register(callback):
                events.append(("event_handler", event_name))
                return callback

            return register

        async def start(self, *, room, agent) -> None:
            events.append(("start", room, agent))

        async def generate_reply(self, *, instructions: str) -> None:
            events.append(("greeting", instructions))

    class FakeRecorder:
        def __init__(self, session, state) -> None:
            events.append(("recorder", session, state))

        def attach(self) -> None:
            events.append("recorder_attached")

        def start(self) -> None:
            events.append("recorder_started")

    monkeypatch.setattr(app, "get_settings", lambda: events.append(("settings", settings)) or settings)
    monkeypatch.setattr(app, "create_agent_session", lambda value: events.append(("session", value)) or FakeSession())
    monkeypatch.setattr(app, "StructuredIntakeAgent", lambda: events.append("agent") or agent)
    monkeypatch.setattr(app, "ConversationRecorder", FakeRecorder)

    asyncio.run(app.entrypoint(SimpleNamespace(room=room)))

    assert events[0:3] == [("settings", settings), ("session", settings), "agent"]
    assert events[3][0] == "recorder"
    assert events[3][2] is intake_data
    assert events[4:] == [
        "recorder_attached",
        ("event_handler", "metrics_collected"),
        "recorder_started",
        ("start", room, agent),
        ("greeting", app.INITIAL_GREETING_INSTRUCTION),
    ]
