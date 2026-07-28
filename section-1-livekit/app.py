"""LiveKit application entrypoint for the structured intake agent."""

from livekit.agents import (
    AgentServer,
    JobContext,
    MetricsCollectedEvent,
    cli,
    metrics,
)

from agent import StructuredIntakeAgent
from config import get_settings
from conversation_recorder import ConversationRecorder
from runtime import create_agent_session


INITIAL_GREETING_INSTRUCTION: str = (
    "Greet the user briefly. Say they may speak Arabic or English, "
    "then ask only for their full name."
)

server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    """Validate configuration and start one structured intake session."""
    settings = get_settings()
    session = create_agent_session(settings)
    agent = StructuredIntakeAgent()
    recorder = ConversationRecorder(session, agent.intake_data)
    recorder.attach()

    @session.on("metrics_collected")
    def on_metrics_collected(event: MetricsCollectedEvent) -> None:
        metrics.log_metrics(event.metrics)

    recorder.start()
    await session.start(room=ctx.room, agent=agent)
    

    await session.generate_reply(
        instructions=INITIAL_GREETING_INSTRUCTION,
    )


if __name__ == "__main__":
    cli.run_app(server)
