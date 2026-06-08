"""
Core AgentLogger class — the primary interface for all agentlog functionality.
Manages run lifecycle, event emission, and writer dispatch.
"""
from __future__ import annotations
import time
import traceback
import uuid
from contextlib import contextmanager
from typing import Any, Generator, Optional

from snaptrace.context import _run_context, next_step_index
from snaptrace.models import (
    AgentEvent,
    EventType,
    RetryPayload,
    Status,
    ToolCallPayload,
    ToolResultPayload,
    _new_id,
    _utcnow,
)
from snaptrace.writers import BaseWriter, JsonlWriter, StdoutWriter


class AgentLogger:
    """
    Primary interface for structured agent run logging.

    Manages the full lifecycle of an agent run: start, steps, tool calls,
    reasoning, retries, errors, and end. Dispatches all events to configured
    writers. Designed for use as a context manager per run.

    Args:
        agent_name: Human-readable identifier for the agent.
        log_file: If provided, write events to this JSONL file path.
        stdout: If True, also print events to stdout.
        pretty: If True and rich is installed, use colored stdout output.
        session_id: Optional session identifier to group related runs.
        writers: Additional custom writer backends.
        metadata: Default metadata applied to every event in this run.

    Example:
        >>> logger = AgentLogger("my-agent", log_file="runs.jsonl")
        >>> with logger.run() as run_id:
        ...     logger.reasoning("Deciding which tool to call")
        ...     logger.tool_call("search", {"query": "weather today"})
    """

    def __init__(
        self,
        agent_name: str,
        log_file: Optional[str] = None,
        stdout: bool = False,
        pretty: bool = False,
        session_id: Optional[str] = None,
        writers: Optional[list[BaseWriter]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initialize AgentLogger with writer backends and default metadata."""
        self.agent_name = agent_name
        self.session_id = session_id
        self._default_metadata = metadata or {}
        self._writers: list[BaseWriter] = list(writers or [])

        if log_file:
            self._writers.append(JsonlWriter(log_file))
        if stdout:
            self._writers.append(StdoutWriter(pretty=pretty))

        self._run_id: Optional[str] = None

    @contextmanager
    def run(
        self,
        run_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Generator[str, None, None]:
        """
        Context manager that wraps a single agent run.

        Emits RUN_START on enter and RUN_END (with success/failure status
        and total latency) on exit. Sets context variables so all nested
        log calls are automatically associated with this run.

        Args:
            run_id: Override the auto-generated run ID.
            metadata: Extra metadata merged into RUN_START and RUN_END events.

        Yields:
            run_id: The active run identifier string.

        Raises:
            Re-raises any exception after logging an ERROR event.
        """
        rid = run_id or uuid.uuid4().hex
        self._run_id = rid
        extra = {**self._default_metadata, **(metadata or {})}
        start_ts = time.perf_counter()

        with _run_context(rid, self.agent_name, self.session_id):
            self._emit(
                event_type=EventType.RUN_START,
                status=Status.PENDING,
                metadata=extra,
            )
            try:
                yield rid
                elapsed = (time.perf_counter() - start_ts) * 1000
                self._emit(
                    event_type=EventType.RUN_END,
                    status=Status.SUCCESS,
                    latency_ms=round(elapsed, 3),
                    metadata=extra,
                )
            except Exception as exc:
                elapsed = (time.perf_counter() - start_ts) * 1000
                self._emit(
                    event_type=EventType.ERROR,
                    status=Status.FAILURE,
                    latency_ms=round(elapsed, 3),
                    payload={
                        "exception_type": type(exc).__name__,
                        "exception_message": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                    metadata=extra,
                )
                self._emit(
                    event_type=EventType.RUN_END,
                    status=Status.FAILURE,
                    latency_ms=round(elapsed, 3),
                    metadata=extra,
                )
                raise
        self._run_id = None

    def step(
        self,
        description: str,
        payload: Optional[dict[str, Any]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Log a discrete reasoning or execution step within a run.

        Args:
            description: Human-readable description of the step.
            payload: Optional structured data associated with this step.
            metadata: Optional extra metadata for this event.
        """
        idx = next_step_index()
        self._emit(
            event_type=EventType.STEP,
            step_index=idx,
            payload={"description": description, **(payload or {})},
            metadata=metadata,
        )

    def reasoning(
        self,
        thought: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Log a reasoning trace / chain-of-thought from the agent.

        Args:
            thought: The raw reasoning text produced by the LLM.
            metadata: Optional extra metadata for this event.
        """
        idx = next_step_index()
        self._emit(
            event_type=EventType.REASONING,
            step_index=idx,
            payload={"thought": thought},
            metadata=metadata,
        )

    def tool_call(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        tool_call_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """
        Log a tool invocation before execution.

        Args:
            tool_name: Name of the tool being called.
            tool_input: Arguments passed to the tool.
            tool_call_id: Optional ID to correlate with tool_result. Auto-generated if None.
            metadata: Optional extra metadata for this event.

        Returns:
            tool_call_id: Use this to log the corresponding tool_result.
        """
        tc_id = tool_call_id or _new_id()
        idx = next_step_index()
        tc = ToolCallPayload(
            tool_name=tool_name,
            tool_input=tool_input,
            tool_call_id=tc_id,
        )
        self._emit(
            event_type=EventType.TOOL_CALL,
            step_index=idx,
            payload=tc.to_dict(),
            metadata=metadata,
        )
        return tc_id

    def tool_result(
        self,
        tool_name: str,
        tool_call_id: str,
        output: Any = None,
        error: Optional[str] = None,
        latency_ms: Optional[float] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Log the result of a tool invocation.

        Args:
            tool_name: Name of the tool that was called.
            tool_call_id: Must match the tool_call_id from the paired tool_call().
            output: Return value from the tool.
            error: Error message if the tool failed.
            latency_ms: Execution duration in milliseconds.
            metadata: Optional extra metadata for this event.
        """
        idx = next_step_index()
        tr = ToolResultPayload(
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            output=output,
            error=error,
        )
        status = Status.FAILURE if error else Status.SUCCESS
        self._emit(
            event_type=EventType.TOOL_RESULT,
            step_index=idx,
            status=status,
            latency_ms=latency_ms,
            payload=tr.to_dict(),
            metadata=metadata,
        )

    def retry(
        self,
        attempt: int,
        max_attempts: int,
        reason: str,
        exception: Optional[Exception] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Log a retry event within a run.

        Args:
            attempt: Current attempt number (1-indexed).
            max_attempts: Total allowed attempts.
            reason: Human-readable reason for the retry.
            exception: The exception that triggered the retry, if any.
            metadata: Optional extra metadata for this event.
        """
        idx = next_step_index()
        rp = RetryPayload(
            attempt=attempt,
            max_attempts=max_attempts,
            reason=reason,
            exception_type=type(exception).__name__ if exception else None,
        )
        self._emit(
            event_type=EventType.RETRY,
            step_index=idx,
            status=Status.PENDING,
            payload=rp.to_dict(),
            metadata=metadata,
        )

    def error(
        self,
        message: str,
        exception: Optional[Exception] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Log a non-fatal error event (does not end the run).

        Args:
            message: Human-readable error description.
            exception: The exception instance, if any.
            metadata: Optional extra metadata for this event.
        """
        idx = next_step_index()
        payload: dict[str, Any] = {"message": message}
        if exception:
            payload["exception_type"] = type(exception).__name__
            payload["exception_message"] = str(exception)
        self._emit(
            event_type=EventType.ERROR,
            step_index=idx,
            status=Status.FAILURE,
            payload=payload,
            metadata=metadata,
        )

    def custom(
        self,
        event_name: str,
        payload: Optional[dict[str, Any]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Log a fully custom event with an arbitrary payload.

        Args:
            event_name: Label for the custom event stored in payload["event_name"].
            payload: Arbitrary structured data.
            metadata: Optional extra metadata for this event.
        """
        idx = next_step_index()
        self._emit(
            event_type=EventType.CUSTOM,
            step_index=idx,
            payload={"event_name": event_name, **(payload or {})},
            metadata=metadata,
        )

    def close(self) -> None:
        """Close all writer backends and flush any buffered events."""
        for writer in self._writers:
            writer.close()

    def _emit(
        self,
        event_type: EventType,
        status: Status = Status.SUCCESS,
        step_index: Optional[int] = None,
        latency_ms: Optional[float] = None,
        payload: Optional[dict[str, Any]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> AgentEvent:
        """
        Internal method to construct and dispatch an AgentEvent.

        Args:
            event_type: The type of event to emit.
            status: Terminal status for this event.
            step_index: Step position within the run.
            latency_ms: Duration in milliseconds.
            payload: Event-specific data dictionary.
            metadata: Extra key-value metadata.

        Returns:
            The constructed AgentEvent that was dispatched.
        """
        merged_meta = {**self._default_metadata, **(metadata or {})}
        event = AgentEvent(
            run_id=self._run_id or "unscoped",
            agent_name=self.agent_name,
            session_id=self.session_id,
            event_type=event_type,
            status=status,
            step_index=step_index,
            latency_ms=latency_ms,
            payload=payload or {},
            metadata=merged_meta,
        )
        for writer in self._writers:
            writer.write(event)
        return event
