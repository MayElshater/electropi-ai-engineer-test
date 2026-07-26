"""Provider-free LiveKit session construction."""

from livekit.agents import AgentSession


def create_agent_session() -> AgentSession:
    """Return a new provider-free LiveKit agent session."""
    return AgentSession()