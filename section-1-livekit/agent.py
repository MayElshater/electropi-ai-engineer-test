"""LiveKit agent integration for structured intake collection."""

from livekit.agents import Agent, function_tool, llm

from intake import IntakeData
from prompts import SYSTEM_PROMPT
from tools import NameCaptureState, ToolResult, process_structured_intake

LOW_CONFIDENCE_NAME_INSTRUCTION = (
    "The latest speech transcript has confidence below 0.75. If it contains a "
    "full name, do not save or confirm that name; ask the user in their current "
    "language to repeat it or spell it. Never translate, normalize, or guess it."
)


class StructuredIntakeAgent(Agent):
    """LiveKit agent that owns structured intake and pending-name state."""

    def __init__(self, intake_data: IntakeData | None = None) -> None:
        """Initialize the agent with supplied or empty intake state."""
        self.intake_data = intake_data if intake_data is not None else IntakeData()
        self.name_capture = NameCaptureState()
        self.latest_transcript_confidence: float | None = None
        super().__init__(instructions=SYSTEM_PROMPT)

    async def on_user_turn_completed(
        self,
        turn_ctx: llm.ChatContext,
        new_message: llm.ChatMessage,
    ) -> None:
        """Expose low transcript confidence to the LLM before its reply."""
        confidence = new_message.transcript_confidence
        self.latest_transcript_confidence = (
            float(confidence) if isinstance(confidence, (int, float)) else None
        )
        if (
            self.latest_transcript_confidence is not None
            and self.latest_transcript_confidence < 0.75
        ):
            turn_ctx.add_message(
                role="system",
                content=LOW_CONFIDENCE_NAME_INSTRUCTION,
            )

    @function_tool(name="submit_structured_intake")
    async def submit_structured_intake(
        self,
        full_name: str | None = None,
        phone_number: str | None = None,
        email: str | None = None,
        address: str | None = None,
        preferred_contact_method: str | None = None,
        full_name_confirmed: bool = False,
        confirmed: bool = False,
    ) -> ToolResult:
        """Update fields and submit only after required confirmations."""
        return process_structured_intake(
            self.intake_data,
            full_name=full_name,
            phone_number=phone_number,
            email=email,
            address=address,
            preferred_contact_method=preferred_contact_method,
            full_name_confirmed=full_name_confirmed,
            full_name_confidence=self.latest_transcript_confidence,
            name_state=self.name_capture,
            confirmed=confirmed,
        )
