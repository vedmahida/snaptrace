"""
Data models for snaptrace events using dataclasses for zero-dependency serialization.
"""
from __future__ import annotations
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Any


class EventType(Enum):
    """All loggable event types in an agent run."""
    RUN_START = "run_start"
    RUN_END = "run_end"
    STEP = "step"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    REASONING = "reasoning"
    RETRY = "retry"
    ERROR = "error"
    CUSTOM = "custom"


class Status(str, Enum):
    """Execution status for events that have a terminal state."""
    SUCCESS = "success"
    FAILURE = "failure"
    SKIPPED = "skipped"
    PENDING = "pending"


def _utcnow() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    """Generate a short unique event ID."""
    return uuid.uuid4().hex[:12]


@dataclass
class AgentEvent:
    """
    Base structure for every event emitted by agentlog.

    Fields:
        `event_id`: Unique identifier for this event.
        `run_id`: Groups all events belonging to one agent run.
        `session_id`: Optional grouping across multiple runs.
        `event_type`: Category of this event (see EventType).
        `timestamp`: UTC ISO 8601 string when the event was created.
        `agent_name`: Human-readable label for the agent emitting the event.
        `step_index`: Sequential position within the run (None for run-level events).
        `status`: Terminal state of the event.
        `latency_ms`: Wall-clock duration in milliseconds (None if not applicable).
        `metadata`: Arbitrary key-value pairs for extension.
        `payload`: Event-specific structured data.
    """
    run_id: str
    agent_name: str
    event_type: EventType
    event_id: str = field(default_factory=_new_id)
    session_id: Optional[str] = None
    timestamp: str = field(default_factory=_utcnow)
    step_index: Optional[int] = None
    status: Status = Status.SUCCESS
    latency_ms: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert event to a plain dictionary, serializing Enums as their values."""
        d = asdict(self)
        d["event_type"] = self.event_type.value
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentEvent:
        """Deserialize an event from a plain dictionary."""
        data = data.copy()
        data["event_type"] = EventType(data["event_type"])
        data["status"] = Status(data["status"])
        return cls(**data)


@dataclass
class ToolCallPayload:
    """
    Structured payload for TOOL_CALL events.

    Fields:
        `tool_name`: Name of the tool being invoked.
        `tool_input`: Arguments passed to the tool.
        `tool_call_id`: Identifier linking call to result.
    """
    tool_name: str
    tool_input: dict[str, Any]
    tool_call_id: str = field(default_factory=_new_id)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return asdict(self)


@dataclass
class ToolResultPayload:
    """
    Structured payload for TOOL_RESULT events.

    Fields:
        `tool_name`: Name of the tool that was invoked.
        `tool_call_id`: Links back to the originating TOOL_CALL event.
        `output`: Raw output returned by the tool.
        `error`: Error message if the tool failed.
    """

    tool_name: str
    tool_call_id: str
    output: Any = None
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return asdict(self)


@dataclass
class RetryPayload:
    """
    Structured payload for RETRY events.

    Fields:
        `attempt`: Current attempt number (1-indexed).
        `max_attempts`: Maximum allowed attempts.
        `reason`: Human-readable reason for the retry.
        `exception_type`: Class name of the exception that triggered retry.
    """

    attempt: int
    max_attempts: int
    reason: str
    exception_type: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return asdict(self)
