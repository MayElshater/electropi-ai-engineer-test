"""Record real LiveKit voice-session events as Markdown."""

from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

from livekit.agents import (
    AgentSession,
    CloseEvent,
    ConversationItemAddedEvent,
    FunctionToolsExecutedEvent,
    UserInputTranscribedEvent,
)

from intake import IntakeData


DEFAULT_EXAMPLES_DIR = Path(__file__).with_name("examples")


def _utc_timestamp(value: datetime | None = None) -> str:
    """Return an ISO 8601 UTC timestamp with a ``Z`` suffix."""
    resolved = value or datetime.now(UTC)
    return resolved.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _json_block(value: Any) -> str:
    """Return stable, readable JSON in a fenced Markdown block."""
    return f"```json\n{json.dumps(value, indent=2, ensure_ascii=False)}\n```"


def _parse_json(value: str) -> Any:
    """Parse a JSON string when possible, otherwise preserve its text."""
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return value


class ConversationRecorder:
    """Persist actual AgentSession conversation and tool events."""

    def __init__(
        self,
        session: AgentSession,
        intake_data: IntakeData,
        *,
        examples_dir: Path = DEFAULT_EXAMPLES_DIR,
    ) -> None:
        """Initialize a recorder for one session without changing session behavior."""
        self._session = session
        self._intake_data = intake_data
        self._examples_dir = examples_dir
        self._sessions_dir = examples_dir / "sessions"
        self._latest_path = examples_dir / "latest_session.md"
        self._session_path: Path | None = None
        self._started_at: datetime | None = None
        self._finished_at: datetime | None = None
        self._blocks: list[str] = []
        self._attached = False

    @property
    def session_path(self) -> Path | None:
        """Return the timestamped output path after recording starts."""
        return self._session_path

    def attach(self) -> None:
        """Subscribe to public LiveKit 1.6.7 events used for recording."""
        if self._attached:
            return
        self._session.on("user_input_transcribed", self._on_user_transcribed)
        self._session.on("conversation_item_added", self._on_conversation_item_added)
        self._session.on("function_tools_executed", self._on_tools_executed)
        self._session.on("close", self._on_close)
        self._attached = True

    def start(self) -> None:
        """Start a timestamped recording and initialize both output files."""
        if self._started_at is not None:
            return
        self._started_at = datetime.now(UTC)
        filename_timestamp = self._started_at.strftime("%Y%m%dT%H%M%S%fZ")
        self._session_path = self._sessions_dir / f"session_{filename_timestamp}.md"
        self._write_snapshot()

    def finish(self, *, finished_at: datetime | None = None) -> None:
        """Finalize the recording with its end time and final structured state."""
        if self._finished_at is not None:
            return
        if self._started_at is None:
            self.start()
        self._finished_at = finished_at or datetime.now(UTC)
        self._write_snapshot()

    def _on_user_transcribed(self, event: UserInputTranscribedEvent) -> None:
        if not event.is_final or not event.transcript.strip():
            return
        self._blocks.append(f"## User\n\n{event.transcript.strip()}")
        self._write_snapshot()

    def _on_conversation_item_added(
        self,
        event: ConversationItemAddedEvent,
    ) -> None:
        item = event.item
        if getattr(item, "role", None) != "assistant":
            return
        text = getattr(item, "text_content", None)
        if isinstance(text, str) and text.strip():
            self._blocks.append(f"## Assistant\n\n{text.strip()}")
            self._write_snapshot()

    def _on_tools_executed(self, event: FunctionToolsExecutedEvent) -> None:
        for function_call, output in event.zipped():
            result = None if output is None else _parse_json(output.output)
            self._blocks.append(
                "\n\n".join(
                    [
                        "## Tool",
                        f"`{function_call.name}`",
                        "### Arguments",
                        _json_block(_parse_json(function_call.arguments)),
                        "### Result",
                        _json_block(result),
                        "## Current State",
                        _json_block(self._intake_data.model_dump()),
                    ]
                )
            )
        self._write_snapshot()

    def _on_close(self, event: CloseEvent) -> None:
        self.finish(finished_at=datetime.fromtimestamp(event.created_at, tz=UTC))

    def _render(self) -> str:
        started_at = self._started_at or datetime.now(UTC)
        sections = [
            "# LiveKit Voice Session",
            "",
            "Started:",
            _utc_timestamp(started_at),
            "",
            "---",
        ]
        for block in self._blocks:
            sections.extend(["", block, "", "---"])
        if self._finished_at is not None:
            sections.extend(
                [
                    "",
                    "Finished:",
                    _utc_timestamp(self._finished_at),
                    "",
                    "## Final State",
                    "",
                    _json_block(self._intake_data.model_dump()),
                ]
            )
        return "\n".join(sections).rstrip() + "\n"

    def _write_snapshot(self) -> None:
        if self._started_at is None or self._session_path is None:
            return
        self._sessions_dir.mkdir(parents=True, exist_ok=True)
        content = self._render()
        self._session_path.write_text(content, encoding="utf-8")
        self._latest_path.write_text(content, encoding="utf-8")