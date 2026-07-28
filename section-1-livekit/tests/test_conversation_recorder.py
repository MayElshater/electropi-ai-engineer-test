"""Tests for native LiveKit event-backed conversation recording."""

import asyncio
from datetime import UTC, datetime
import json
from pathlib import Path

from livekit.agents import (
    AgentSession,
    CloseEvent,
    ConversationItemAddedEvent,
    FunctionToolsExecutedEvent,
    UserInputTranscribedEvent,
    llm,
)

from conversation_recorder import ConversationRecorder
from intake import IntakeData


def make_session() -> AgentSession:
    """Create a real event emitter with an explicit isolated event loop."""
    return AgentSession(loop=asyncio.new_event_loop())


def test_recorder_writes_native_session_events(tmp_path: Path) -> None:
    session = make_session()
    intake = IntakeData()
    recorder = ConversationRecorder(session, intake, examples_dir=tmp_path)
    recorder.attach()
    recorder.start()

    session.emit(
        "user_input_transcribed",
        UserInputTranscribedEvent(transcript="partial", is_final=False),
    )
    session.emit(
        "user_input_transcribed",
        UserInputTranscribedEvent(transcript="My name is Noor.", is_final=True),
    )
    session.emit(
        "conversation_item_added",
        ConversationItemAddedEvent(
            item=llm.ChatMessage(role="assistant", content=["What is your email?"])
        ),
    )

    intake.full_name = "Noor Hassan"
    arguments = {"full_name": "Noor Hassan", "confirmed": False}
    result = {
        "success": False,
        "message": "Explicit confirmation is required before submission.",
        "missing_fields": ["phone_number", "email", "address"],
        "requires_confirmation": True,
    }
    session.emit(
        "function_tools_executed",
        FunctionToolsExecutedEvent(
            function_calls=[
                llm.FunctionCall(
                    call_id="call-1",
                    name="submit_structured_intake",
                    arguments=json.dumps(arguments),
                )
            ],
            function_call_outputs=[
                llm.FunctionCallOutput(
                    call_id="call-1",
                    name="submit_structured_intake",
                    output=json.dumps(result),
                    is_error=False,
                )
            ],
        ),
    )
    session.emit(
        "close",
        CloseEvent(
            reason="participant_disconnected",
            created_at=datetime(2026, 7, 28, 15, 42, 10, tzinfo=UTC).timestamp(),
        ),
    )

    latest = tmp_path / "latest_session.md"
    timestamped = list((tmp_path / "sessions").glob("session_*.md"))
    assert latest.is_file()
    assert len(timestamped) == 1
    assert latest.read_text(encoding="utf-8") == timestamped[0].read_text(
        encoding="utf-8"
    )

    content = latest.read_text(encoding="utf-8")
    assert "partial" not in content
    assert "## User\n\nMy name is Noor." in content
    assert "## Assistant\n\nWhat is your email?" in content
    assert "`submit_structured_intake`" in content
    assert '"full_name": "Noor Hassan"' in content
    assert "## Current State" in content
    assert "Finished:\n2026-07-28T15:42:10Z" in content
    assert "## Final State" in content


def test_attach_is_idempotent(tmp_path: Path) -> None:
    session = make_session()
    recorder = ConversationRecorder(session, IntakeData(), examples_dir=tmp_path)

    recorder.attach()
    recorder.attach()
    recorder.start()
    session.emit(
        "user_input_transcribed",
        UserInputTranscribedEvent(transcript="One final turn.", is_final=True),
    )

    content = (tmp_path / "latest_session.md").read_text(encoding="utf-8")
    assert content.count("One final turn.") == 1
