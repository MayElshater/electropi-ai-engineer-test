"""Tests for the Deepgram and Gemini LiveKit runtime skeleton."""

import asyncio
import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest
from livekit.agents import AgentServer, AgentSession, cli
from livekit.plugins import cartesia, deepgram, google

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
        google_api_key="google-test-key",
        cartesia_api_key="cartesia-key",
    )


def import_app() -> ModuleType:
    """Import a fresh app module for isolated import behavior."""
    sys.modules.pop("app", None)
    return importlib.import_module("app")


def test_create_stt_returns_deepgram_stt() -> None:
    assert isinstance(runtime_module.create_stt(make_settings()), deepgram.STT)


def test_create_stt_passes_model_language_and_api_key(monkeypatch) -> None:
    settings = make_settings()
    created = object()
    received: dict[str, object] = {}

    def fake_stt(**kwargs: object) -> object:
        received.update(kwargs)
        return created

    monkeypatch.setattr(runtime_module.deepgram, "STT", fake_stt)

    result = runtime_module.create_stt(settings)

    assert result is created
    assert received["model"] == "nova-3"
    assert received["language"] == "multi"
    assert received["api_key"] == settings.deepgram_api_key


def test_create_llm_returns_google_llm() -> None:
    assert isinstance(runtime_module.create_llm(make_settings()), google.LLM)


def test_create_llm_passes_model_and_api_key(monkeypatch) -> None:
    settings = make_settings()
    created = object()
    received: dict[str, object] = {}

    def fake_llm(**kwargs: object) -> object:
        received.update(kwargs)
        return created

    monkeypatch.setattr(runtime_module.google, "LLM", fake_llm)

    result = runtime_module.create_llm(settings)

    assert result is created
    assert received["model"] == "gemini-3.5-flash"
    assert received["api_key"] == settings.google_api_key


def test_create_tts_returns_cartesia_tts() -> None:
    assert isinstance(runtime_module.create_tts(make_settings()), cartesia.TTS)


def test_create_tts_passes_model_language_and_api_key(monkeypatch) -> None:
    settings = make_settings()
    created = object()
    received: dict[str, object] = {}

    def fake_tts(**kwargs: object) -> object:
        received.update(kwargs)
        return created

    monkeypatch.setattr(runtime_module.cartesia, "TTS", fake_tts)

    result = runtime_module.create_tts(settings)

    assert result is created
    assert received == {
        "model": "sonic-3",
        "language": None,
        "api_key": settings.cartesia_api_key,
    }
    assert "voice" not in received


def test_create_agent_session_returns_agent_session() -> None:
    async def create() -> AgentSession:
        return runtime_module.create_agent_session(make_settings())

    assert isinstance(asyncio.run(create()), AgentSession)


def test_create_agent_session_supplies_stt_llm_and_tts(monkeypatch) -> None:
    settings = make_settings()
    stt = object()
    llm = object()
    tts = object()
    session = object()
    dependencies: list[tuple[str, Settings]] = []
    received: dict[str, object] = {}

    monkeypatch.setattr(
        runtime_module,
        "create_stt",
        lambda value: dependencies.append(("stt", value)) or stt,
    )
    monkeypatch.setattr(
        runtime_module,
        "create_llm",
        lambda value: dependencies.append(("llm", value)) or llm,
    )
    monkeypatch.setattr(
        runtime_module,
        "create_tts",
        lambda value: dependencies.append(("tts", value)) or tts,
    )

    def fake_session(**kwargs: object) -> object:
        received.update(kwargs)
        return session

    monkeypatch.setattr(runtime_module, "AgentSession", fake_session)

    result = runtime_module.create_agent_session(settings)

    assert result is session
    assert dependencies == [
        ("stt", settings),
        ("llm", settings),
        ("tts", settings),
    ]
    assert received == {"stt": stt, "llm": llm, "tts": tts}


def test_explicit_settings_avoid_get_settings(monkeypatch) -> None:
    monkeypatch.setattr(
        runtime_module,
        "get_settings",
        lambda: pytest.fail("get_settings should not be called"),
    )
    monkeypatch.setattr(runtime_module, "create_stt", lambda settings: object())
    monkeypatch.setattr(runtime_module, "create_llm", lambda settings: object())
    monkeypatch.setattr(runtime_module, "create_tts", lambda settings: object())
    monkeypatch.setattr(runtime_module, "AgentSession", lambda **kwargs: object())

    runtime_module.create_agent_session(make_settings())


def test_zero_argument_session_loads_settings_once(monkeypatch) -> None:
    settings = make_settings()
    calls: list[str] = []
    monkeypatch.setattr(
        runtime_module,
        "get_settings",
        lambda: calls.append("settings") or settings,
    )
    monkeypatch.setattr(runtime_module, "create_stt", lambda value: object())
    monkeypatch.setattr(runtime_module, "create_llm", lambda value: object())
    monkeypatch.setattr(runtime_module, "create_tts", lambda value: object())
    monkeypatch.setattr(runtime_module, "AgentSession", lambda **kwargs: object())

    runtime_module.create_agent_session()

    assert calls == ["settings"]


def test_factories_return_fresh_instances() -> None:
    settings = make_settings()
    first_stt = runtime_module.create_stt(settings)
    second_stt = runtime_module.create_stt(settings)
    first_llm = runtime_module.create_llm(settings)
    second_llm = runtime_module.create_llm(settings)
    first_tts = runtime_module.create_tts(settings)
    second_tts = runtime_module.create_tts(settings)

    async def create_pair() -> tuple[AgentSession, AgentSession]:
        return (
            runtime_module.create_agent_session(settings),
            runtime_module.create_agent_session(settings),
        )

    first_session, second_session = asyncio.run(create_pair())

    assert first_stt is not second_stt
    assert first_llm is not second_llm
    assert first_tts is not second_tts
    assert first_session is not second_session


def test_import_runtime_has_no_factory_side_effects(monkeypatch) -> None:
    calls: list[str] = []
    sys.modules.pop("runtime", None)

    with monkeypatch.context() as patch:
        patch.setattr(config, "get_settings", lambda: calls.append("settings"))
        patch.setattr(deepgram, "STT", lambda **kwargs: calls.append("stt"))
        patch.setattr(google, "LLM", lambda **kwargs: calls.append("llm"))
        patch.setattr(cartesia, "TTS", lambda **kwargs: calls.append("tts"))
        importlib.import_module("runtime")

    sys.modules["runtime"] = runtime_module
    assert calls == []


def test_provider_keys_are_not_exposed_by_settings_repr() -> None:
    settings = make_settings()
    representation = repr(settings)

    assert settings.deepgram_api_key not in representation
    assert settings.google_api_key not in representation
    assert settings.cartesia_api_key not in representation


def test_deepgram_key_is_not_added_to_factory_errors(monkeypatch) -> None:
    settings = make_settings()

    def fail_stt(**kwargs: object) -> object:
        raise RuntimeError("STT construction failed")

    monkeypatch.setattr(runtime_module.deepgram, "STT", fail_stt)

    with pytest.raises(RuntimeError) as error:
        runtime_module.create_stt(settings)

    assert settings.deepgram_api_key not in str(error.value)


def test_google_key_is_not_added_to_factory_errors(monkeypatch) -> None:
    settings = make_settings()

    def fail_llm(**kwargs: object) -> object:
        raise RuntimeError("LLM construction failed")

    monkeypatch.setattr(runtime_module.google, "LLM", fail_llm)

    with pytest.raises(RuntimeError) as error:
        runtime_module.create_llm(settings)

    assert settings.google_api_key not in str(error.value)


def test_cartesia_key_is_not_added_to_factory_errors(monkeypatch) -> None:
    settings = make_settings()

    def fail_tts(**kwargs: object) -> object:
        raise RuntimeError("TTS construction failed")

    monkeypatch.setattr(runtime_module.cartesia, "TTS", fail_tts)

    with pytest.raises(RuntimeError) as error:
        runtime_module.create_tts(settings)

    assert settings.cartesia_api_key not in str(error.value)


def test_import_exposes_server_without_running_cli(monkeypatch) -> None:
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


def test_entrypoint_validates_settings_and_starts_agent(monkeypatch) -> None:
    app = import_app()
    events: list[object] = []
    room = object()
    settings = object()
    agent = object()

    class FakeSession:
        async def start(self, *, room, agent) -> None:
            events.append(("start", room, agent))

        async def generate_reply(self, *, instructions: str) -> None:
            events.append(("greeting", instructions))

    monkeypatch.setattr(
        app,
        "get_settings",
        lambda: events.append(("settings", settings)) or settings,
    )
    monkeypatch.setattr(
        app,
        "create_agent_session",
        lambda value: events.append(("session", value)) or FakeSession(),
    )
    monkeypatch.setattr(
        app,
        "StructuredIntakeAgent",
        lambda: events.append("agent") or agent,
    )

    asyncio.run(app.entrypoint(SimpleNamespace(room=room)))

    assert events == [
        ("settings", settings),
        ("session", settings),
        "agent",
        ("start", room, agent),
        ("greeting", app.INITIAL_GREETING_INSTRUCTION),
    ]