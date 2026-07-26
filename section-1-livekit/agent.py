"""LiveKit agent integration for structured intake collection."""

from livekit.agents import Agent, function_tool

from intake import IntakeData
from prompts import SYSTEM_PROMPT
from tools import ToolResult, process_structured_intake


class StructuredIntakeAgent(Agent):
    """LiveKit agent that owns structured intake state."""

    def __init__(self, intake_data: IntakeData | None = None) -> None:
        """Initialize the agent with supplied or empty intake state."""
        self.intake_data = intake_data if intake_data is not None else IntakeData()
        super().__init__(instructions=SYSTEM_PROMPT)

    @function_tool(name="submit_structured_intake")
    async def submit_structured_intake(
        self,
        full_name: str | None = None,
        phone_number: str | None = None,
        email: str | None = None,
        address: str | None = None,
        preferred_contact_method: str | None = None,
        confirmed: bool = False,
    ) -> ToolResult:
        """Update fields and submit only after explicit user confirmation."""
        return process_structured_intake(
            self.intake_data,
            full_name=full_name,
            phone_number=phone_number,
            email=email,
            address=address,
            preferred_contact_method=preferred_contact_method,
            confirmed=confirmed,
        )
