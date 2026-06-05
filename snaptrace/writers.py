"""
Writer backends for persisting AgentEvents.
Supports JSONL file writer and stdout writer with optional rich formatting.
"""
from __future__ import annotations
import json
import sys
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from snaptrace.models import AgentEvent


class BaseWriter(ABC):
    """Abstract base class for all snaptrace writers."""

    @abstractmethod
    def write(self, event: "AgentEvent") -> None:
        """
        Persist a single AgentEvent.

        Args:
            event: The event to write.
        """

    def close(self) -> None:
        """Release any resources held by the writer. Override if needed."""


class JsonlWriter(BaseWriter):
    """
    Writes events as JSONL to a file.

    Args:
        path: File path to write to. Created if it does not exist.
        append: If True, appends to existing file. If False, overwrites.
    """

    def __init__(self, path: str | Path, append: bool = True) -> None:
        """Initialize the JSONL writer and open the file handle."""
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        self._fh = self._path.open(mode, encoding="utf-8")
        self._lock = threading.Lock()

    def write(self, event: "AgentEvent") -> None:
        """
        Serialize and append an event as a JSON line.

        Args:
            event: The AgentEvent to persist.
        """
        line = json.dumps(event.to_dict(), ensure_ascii=False, default=str)
        with self._lock:
            self._fh.write(line + "\n")
            self._fh.flush()

    def close(self) -> None:
        """Flush and close the underlying file handle."""
        with self._lock:
            self._fh.flush()
            self._fh.close()


class StdoutWriter(BaseWriter):
    """
    Writes events as JSON lines to stdout.

    Optionally uses `rich` for pretty-printed colored output if available.

    Args:
        pretty: If True and rich is installed, use colored output.
        level_filter: Only print events of these types. None means print all.
    """

    def __init__(
        self,
        pretty: bool = False,
        level_filter: Optional[list[str]] = None,
    ) -> None:
        """Initialize stdout writer with optional rich formatting."""
        self._pretty = pretty
        self._level_filter = set(level_filter) if level_filter else None
        self._rich_available = False
        if pretty:
            try:
                from rich.console import Console  # type: ignore
                from rich.syntax import Syntax  # type: ignore

                self._console = Console(stderr=False)
                self._Syntax = Syntax
                self._rich_available = True
            except ImportError:
                pass

    def write(self, event: "AgentEvent") -> None:
        """
        Print an event to stdout, filtered by level if configured.

        Args:
            event: The AgentEvent to print.
        """
        if self._level_filter and event.event_type.value not in self._level_filter:
            return

        line = json.dumps(event.to_dict(), ensure_ascii=False,
                          default=str, indent=2)

        if self._rich_available:
            syntax = self._Syntax(line, "json", theme="monokai")
            self._console.print(syntax)
        else:
            print(line, file=sys.stdout)


class MultiWriter(BaseWriter):
    """
    Fan-out writer that dispatches each event to multiple backends.

    Args:
        writers: List of BaseWriter instances to write to.
    """

    def __init__(self, writers: list[BaseWriter]) -> None:
        """Initialize with a list of writer backends."""
        self._writers = writers

    def write(self, event: "AgentEvent") -> None:
        """
        Write event to all configured backends.

        Args:
            event: The AgentEvent to dispatch.
        """
        for writer in self._writers:
            writer.write(event)

    def close(self) -> None:
        """Close all underlying writers."""
        for writer in self._writers:
            writer.close()




