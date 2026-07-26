"""Tests for the provider-free LiveKit runtime skeleton."""

import asyncio
import importlib
import sys
from types import SimpleNamespace

import pytest
from livekit.agents import AgentServer, AgentSession, cli

from runtime import create_agent_session


def import_app():
    """Import a fresh app module for isolated import behavior."""
    sys.modules.pop("app", None)
    return importlib.import_module("app")


def test_create_agent_session_returns_agent_session() -> None:
    async def create() -> AgentSession:
        return create_agent_session()

    assert isinstance(asyncio.run(create()), AgentSession)


def test_create_agent_session_returns_fresh_instances() -> None:
    async def create_pair() -> tuple[AgentSession, AgentSession]:
        return create_agent_session(), create_agent_session()

    first, second = asyncio.run(create_pair())
    assert first is not second


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
        lambda: events.append("session") or FakeSession(),
    )
    monkeypatch.setattr(
        app,
        "StructuredIntakeAgent",
        lambda: events.append("agent") or agent,
    )

    asyncio.run(app.entrypoint(SimpleNamespace(room=room)))

    assert events == [
        ("settings", settings),
        "session",
        "agent",
        ("start", room, agent),
        ("greeting", app.INITIAL_GREETING_INSTRUCTION),
    ]