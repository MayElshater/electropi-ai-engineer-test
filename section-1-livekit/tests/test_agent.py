"""Tests for the LiveKit structured intake agent integration."""

import asyncio

from livekit.agents import Agent

import agent as agent_module
from agent import StructuredIntakeAgent
from intake import IntakeData
from tools import ToolResult


def test_agent_subclasses_livekit_agent() -> None:
    assert issubclass(StructuredIntakeAgent, Agent)


def test_constructor_creates_empty_intake_by_default() -> None:
    agent = StructuredIntakeAgent()

    assert isinstance(agent.intake_data, IntakeData)
    assert agent.intake_data == IntakeData()


def test_constructor_preserves_supplied_intake() -> None:
    intake = IntakeData(full_name="Ada Lovelace")

    agent = StructuredIntakeAgent(intake)

    assert agent.intake_data is intake


def test_agent_registers_only_submission_tool() -> None:
    agent = StructuredIntakeAgent()

    assert len(agent.tools) == 1
    assert agent.tools[0].info.name == "submit_structured_intake"


def test_tool_wrapper_delegates_with_agent_state(monkeypatch) -> None:
    intake = IntakeData(full_name="Existing Name")
    agent = StructuredIntakeAgent(intake)
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
            confirmed=True,
        )
    )

    assert result == expected
    assert received["state"] is intake
    assert received["phone_number"] == "  +20 100 123 4567  "
    assert received["confirmed"] is True
