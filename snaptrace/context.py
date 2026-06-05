"""
`Thread-local` and `contextvars-based` run context management.
Provides the active run_id and step counter without passing state explicitly.

Background tracking tool that remembers important information about agent's current task without 
making you pass variables everywhere manually.
"""
from __future__ import annotations
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Generator, Optional

_run_id_var: ContextVar[Optional[str]] = ContextVar("run_id", default=None)
_session_id_var: ContextVar[Optional[str]
                            ] = ContextVar("session_id", default=None)
_agent_name_var: ContextVar[Optional[str]
                            ] = ContextVar("agent_name", default=None)
_step_counter_lock = threading.Lock()


class _StepCounter:
    """Thread-safe step counter scoped to the current run context."""

    def __init__(self) -> None:
        """Initialize the step counter at zero."""
        self._counter = 0
        self._lock = threading.Lock()

    def increment(self) -> int:
        """Increment and return the new step index."""
        with self._lock:
            self._value += 1
            return self._value

    def reset(self) -> None:
        """Reset counter to zero."""
        with self._lock:
            self._value = 0


_step_counter_var: ContextVar[Optional[_StepCounter]] = ContextVar(
    "step_counter", default=None
)


def get_run_id() -> Optional[str]:
    """Return the active run ID from context, or None if no run is active."""
    return _run_id_var.get()


def get_session_id() -> Optional[str]:
    """Return the active session ID from context, or None."""
    return _session_id_var.get()


def get_agent_name() -> Optional[str]:
    """Return the active agent name from context, or None."""
    return _agent_name_var.get()


def next_step_index() -> int:
    """Increment and return the next step index for the current run."""
    counter = _step_counter_var.get()
    if counter is None:
        raise RuntimeError(
            "No active agent run context. Use AgentLogger.run() context manager.")
    return counter.increment()


@contextmanager
def _run_context(
    run_id: str,
    agent_name: str,
    session_id: Optional[str] = None,
) -> Generator[None, None, None]:
    """
    Internal context manager that sets and clears contextvars for a run.

    Args:
        run_id: Unique identifier for this run.
        agent_name: Name of the agent.
        session_id: Optional session grouping identifier.
    """
    counter = _StepCounter()
    token_run = _run_id_var.set(run_id)
    token_session = _session_id_var.set(session_id)
    token_agent = _agent_name_var.set(agent_name)
    token_counter = _step_counter_var.set(counter)
    try:
        yield
    finally:
        _run_id_var.reset(token_run)
        _session_id_var.reset(token_session)
        _agent_name_var.reset(token_agent)
        _step_counter_var.reset(token_counter)
