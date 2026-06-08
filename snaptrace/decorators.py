"""
Function decorators for zero-boilerplate agent step and tool call logging.
Works with both sync and async functions.
"""
from __future__ import annotations
import asyncio
import functools
import time
from typing import Any, Callable, Optional, TypeVar
from snaptrace.context import get_agent_name, get_run_id
from snaptrace.core import AgentLogger

F = TypeVar("F", bound=Callable[..., Any])


def log_step(
    logger: AgentLogger,
    description: Optional[str] = None,
) -> Callable[[F], F]:
    """
    Decorator that logs a function call as a STEP event.

    Works with sync and async functions. The step description defaults
    to the function's qualified name if not provided.

    Args:
        logger: The AgentLogger instance to emit events through.
        description: Human-readable step label. Defaults to function name.

    Returns:
        Decorated function with automatic step logging.

    Example:
        >>> @log_step(logger, description="Fetch user context")
        ... def fetch_user(user_id: str) -> dict:
        ...     return db.get(user_id)
    """

    def decorator(fn: F) -> F:
        label = description or fn.__qualname__

        if asyncio.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                """Async wrapper that emits STEP event around the coroutine."""
                logger.step(label, payload={"args_count": len(
                    args), "kwargs": list(kwargs.keys())})
                return await fn(*args, **kwargs)

            return async_wrapper  # type: ignore

        @functools.wraps(fn)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            """Sync wrapper that emits STEP event around the function call."""
            logger.step(label, payload={"args_count": len(
                args), "kwargs": list(kwargs.keys())})
            return fn(*args, **kwargs)

        return sync_wrapper  # type: ignore

    return decorator


def log_tool(
    logger: AgentLogger,
    tool_name: Optional[str] = None,
) -> Callable[[F], F]:
    """
    Decorator that logs a function call as a TOOL_CALL + TOOL_RESULT event pair.

    Captures tool input (kwargs), output, latency, and any exceptions.
    Works with sync and async functions.

    Args:
        logger: The AgentLogger instance to emit events through.
        tool_name: Override for the tool name. Defaults to function name.

    Returns:
        Decorated function with automatic tool call/result logging.

    Example:
        >>> @log_tool(logger, tool_name="web_search")
        ... def search(query: str, max_results: int = 5) -> list[dict]:
        ...     return api.search(query, max_results)
    """

    def decorator(fn: F) -> F:
        name = tool_name or fn.__name__

        if asyncio.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                """Async wrapper emitting TOOL_CALL and TOOL_RESULT events."""
                tc_id = logger.tool_call(name, tool_input=kwargs)
                start = time.perf_counter()
                try:
                    result = await fn(*args, **kwargs)
                    elapsed = (time.perf_counter() - start) * 1000
                    logger.tool_result(
                        name, tc_id, output=result, latency_ms=round(elapsed, 3))
                    return result
                except Exception as exc:
                    elapsed = (time.perf_counter() - start) * 1000
                    logger.tool_result(name, tc_id, error=str(
                        exc), latency_ms=round(elapsed, 3))
                    raise

            return async_wrapper  # type: ignore

        @functools.wraps(fn)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            """Sync wrapper emitting TOOL_CALL and TOOL_RESULT events."""
            tc_id = logger.tool_call(name, tool_input=kwargs)
            start = time.perf_counter()
            try:
                result = fn(*args, **kwargs)
                elapsed = (time.perf_counter() - start) * 1000
                logger.tool_result(name, tc_id, output=result,
                                   latency_ms=round(elapsed, 3))
                return result
            except Exception as exc:
                elapsed = (time.perf_counter() - start) * 1000
                logger.tool_result(name, tc_id, error=str(
                    exc), latency_ms=round(elapsed, 3))
                raise

        return sync_wrapper  # type: ignore

    return decorator


def log_retry(
    logger: AgentLogger,
    max_attempts: int = 3,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[F], F]:
    """
    Decorator that automatically retries a function and logs each retry as a RETRY event.

    Args:
        logger: The AgentLogger instance to emit events through.
        max_attempts: Maximum number of total attempts (including first try).
        exceptions: Tuple of exception types that trigger a retry.

    Returns:
        Decorated function with automatic retry and logging.

    Example:
        >>> @log_retry(logger, max_attempts=3, exceptions=(TimeoutError, ConnectionError))
        ... def call_api(endpoint: str) -> dict:
        ...     return requests.get(endpoint).json()
    """

    def decorator(fn: F) -> F:

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            """Wrapper that retries on specified exceptions with RETRY event logging."""
            last_exc: Optional[Exception] = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt < max_attempts:
                        logger.retry(
                            attempt=attempt,
                            max_attempts=max_attempts,
                            reason=str(exc),
                            exception=exc,
                        )
                    else:
                        raise
            raise last_exc  # type: ignore

        return wrapper  # type: ignore

    return decorator
