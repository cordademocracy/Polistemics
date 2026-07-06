"""Adaptive progress reporting — rich bars for TTY, structlog heartbeat for non-TTY.

Modes:
- **TTY (rich):** renders a live progress bar with spinner, bar, %, M/N, elapsed, ETA.
- **Non-TTY (heartbeat):** emits structured log lines at configurable percentage
  thresholds and/or time intervals — suitable for subagents, pipes, and CI.

Both modes support sync (`with`) and async (`async with`) context managers.
"""
from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, Any, Literal

import structlog

if TYPE_CHECKING:
    from types import TracebackType

    from rich.progress import Progress

logger = structlog.get_logger(__name__)


def _make_rich_progress() -> Progress:
    """Lazily import rich and build a Progress instance."""
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )

    return Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )


def _is_terminal() -> bool:
    """Lazily check if stdout is a TTY via rich."""
    from rich.console import Console

    return Console().is_terminal


class ProgressTracker:
    """Adaptive progress reporting — rich bars for TTY, structlog heartbeat for non-TTY.

    Supports both async (``async with``) and sync (``with``) context managers.
    Thread-safe: ``advance()`` may be called from multiple async tasks concurrently.

    Args:
        total: Total number of steps expected.
        description: Human-readable label shown in the progress bar or log lines.
        heartbeat_pct: Emit a heartbeat log every this fraction of total in non-TTY
            mode (default 0.1 = every 10%).
        heartbeat_interval_s: Also emit a heartbeat if this many seconds have passed
            since the last one, regardless of percentage thresholds (default 30 s).
        force_mode: Override TTY auto-detection. ``"rich"`` forces rich bars;
            ``"heartbeat"`` forces structlog lines.

    Example::

        async with ProgressTracker(total=100, description="Processing") as pt:
            for item in items:
                await process(item)
                pt.advance(1, model="gpt")
    """

    def __init__(
        self,
        total: int,
        description: str,
        *,
        heartbeat_pct: float = 0.1,
        heartbeat_interval_s: float = 30.0,
        force_mode: Literal["rich", "heartbeat"] | None = None,
    ) -> None:
        self.total = total
        self.description = description
        self._heartbeat_pct = heartbeat_pct
        self._heartbeat_interval_s = heartbeat_interval_s

        # Detect TTY once, allow override
        if force_mode is not None:
            self._use_rich = force_mode == "rich"
        else:
            self._use_rich = _is_terminal()

        # Shared timing state
        self._start_time: float = 0.0

        # Rich-mode state
        self._progress: Progress | None = None
        self._task_id: Any = None  # rich.progress.TaskID

        # Heartbeat-mode state
        self._lock = threading.Lock()
        self._completed: int = 0
        self._last_heartbeat_time: float = 0.0
        # Track which percentage thresholds have already been emitted
        self._emitted_thresholds: set[int] = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def advance(self, n: int = 1, **fields: Any) -> None:
        """Advance progress by n steps.

        Args:
            n: Number of steps to advance.
            **fields: Extra key-value pairs included in heartbeat log events.
        """
        if self._use_rich:
            self._rich_advance(n)
        else:
            self._heartbeat_advance(n, **fields)

    def update_description(self, description: str) -> None:
        """Change the progress description mid-run.

        Args:
            description: New human-readable label.
        """
        self.description = description
        if self._use_rich and self._progress is not None and self._task_id is not None:
            self._progress.update(self._task_id, description=description)

    # ------------------------------------------------------------------
    # Sync context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "ProgressTracker":
        self._start_time = time.monotonic()
        if self._use_rich:
            self._progress = _make_rich_progress()
            self._progress.start()
            self._task_id = self._progress.add_task(self.description, total=self.total)
        else:
            self._last_heartbeat_time = self._start_time
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if self._use_rich and self._progress is not None:
            self._progress.stop()
            self._progress = None
            self._task_id = None

    # ------------------------------------------------------------------
    # Async context manager — delegates to sync (no I/O awaiting needed)
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "ProgressTracker":
        return self.__enter__()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.__exit__(exc_type, exc_val, exc_tb)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rich_advance(self, n: int) -> None:
        """Advance rich progress bar by n steps."""
        if self._progress is not None and self._task_id is not None:
            self._progress.advance(self._task_id, n)

    def _heartbeat_advance(self, n: int, **fields: Any) -> None:
        """Advance internal counter and emit heartbeat log lines when thresholds fire.

        The entire check-and-emit sequence is under a single lock to prevent
        duplicate heartbeat emissions from concurrent ``advance()`` calls.
        """
        now = time.monotonic()

        with self._lock:
            self._completed += n
            completed = self._completed
            elapsed_s = now - self._start_time

            # Avoid division by zero for total=0
            pct = (completed / self.total) if self.total > 0 else 1.0
            pct_int = int(pct * 100)  # e.g. 10, 20, ..., 100

            time_threshold_hit = (
                (now - self._last_heartbeat_time) >= self._heartbeat_interval_s
            )

            # Determine which percentage thresholds are newly crossed
            step = max(1, int(self._heartbeat_pct * 100))
            newly_crossed = [
                t
                for t in range(step, 101, step)
                if t <= pct_int and t not in self._emitted_thresholds
            ]

            should_emit = bool(newly_crossed) or time_threshold_hit
            # Always emit at 100% completion
            if completed == self.total and 100 not in self._emitted_thresholds:
                should_emit = True
                if 100 not in newly_crossed:
                    newly_crossed.append(100)

            if not should_emit:
                return

            self._emitted_thresholds.update(newly_crossed)
            self._last_heartbeat_time = now

        # Log outside the lock (I/O should not block other advance() calls)
        logger.info(
            "progress_heartbeat",
            description=self.description,
            completed=completed,
            total=self.total,
            pct=round(pct, 4),
            elapsed_s=round(elapsed_s, 2),
            **fields,
        )
