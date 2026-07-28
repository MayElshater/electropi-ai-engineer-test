"""Tests for the LiveKit structured intake agent integration."""

import asyncio

from livekit.agents import Agent, llm

import agent as agent_module
from agent import LOW_CONFIDENCE_NAME_INSTRUCTION, StructuredIntakeAgent
from intake import IntakeData
from tools import ToolResult


def test_agent_subclasses_livekit_agent() -> None:
    assert issubclass(StructuredIntakeAgent, Agent)


def test_constructor_creates_empty_intake_and_name_state_by_default() -> None:
    agent = StructuredIntakeAgent()

    assert isinstance(agent.intake_data, IntakeData)
    assert agent.intake_data == IntakeData()
    assert agent.name_capture.value is None
    assert agent.latest_transcript_confidence is None


def test_constructor_preserves_supplied_intake() -> None:
    intake = IntakeData(full_name="Ada Lovelace")

    agent = StructuredIntakeAgent(intake)

    assert agent.intake_data is intake


def test_agent_registers_only_submission_tool() -> None:
    agent = StructuredIntakeAgent()

    assert len(agent.tools) == 1
    assert agent.tools[0].info.name == "submit_structured_intake"


def test_tool_wrapper_delegates_name_state_and_confidence(monkeypatch) -> None:
    intake = IntakeData(full_name="Existing Name")
    agent = StructuredIntakeAgent(intake)
    agent.latest_transcript_confidence = 0.84
    expected: ToolResult = {
        "success": False,
        "message": "delegated",
        "missing_fields": ["phone_number"],
        "requires_confirmation": True,
    }
    received: dict[str, object] = {}

    def fake_process(state: IntakeData, **kwargs: object) -> ToolResult:
        received["state"] = state
        received.update(kwargs)
        return expected

    monkeypatch.setattr(agent_module, "process_structured_intake", fake_process)

    result = asyncio.run(
        StructuredIntakeAgent.submit_structured_intake.__wrapped__(
            agent,
            phone_number="  +20 100 123 4567  ",
            full_name_confirmed=True,
            confirmed=True,
        )
    )

    assert result == expected
    assert received["state"] is intake
    assert received["name_state"] is agent.name_capture
    assert received["full_name_confidence"] == 0.84
    assert received["phone_number"] == "  +20 100 123 4567  "
    assert received["full_name_confirmed"] is True
    assert received["confirmed"] is True


def test_low_confidence_turn_adds_name_safety_instruction() -> None:
    agent = StructuredIntakeAgent()
    turn_context = llm.ChatContext()
    message = llm.ChatMessage(
        role="user",
        content=["Es Mimahy Mohammad"],
        transcript_confidence=0.589,
    )

    asyncio.run(agent.on_user_turn_completed(turn_context, message))

    assert agent.latest_transcript_confidence == 0.589
    assert len(turn_context.items) == 1
    assert turn_context.items[0].text_content == LOW_CONFIDENCE_NAME_INSTRUCTION


def test_high_confidence_turn_does_not_add_safety_instruction() -> None:
    agent = StructuredIntakeAgent()
    turn_context = llm.ChatContext()
    message = llm.ChatMessage(
        role="user",
        content=["مي محمد"],
        transcript_confidence=0.9,
    )

    asyncio.run(agent.on_user_turn_completed(turn_context, message))

    assert agent.latest_transcript_confidence == 0.9
    assert turn_context.items == []
