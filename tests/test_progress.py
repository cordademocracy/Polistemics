"""Tests for src/common/progress.py — ProgressTracker."""
from __future__ import annotations

import pytest
from structlog.testing import capture_logs

from src.common.progress import ProgressTracker


# ---------------------------------------------------------------------------
# Heartbeat mode (non-TTY)
# ---------------------------------------------------------------------------


def test_heartbeat_emits_at_thresholds() -> None:
    """Heartbeat fires exactly at 10%, 20%, ..., 100% when advancing one step at a time."""
    with capture_logs() as cap:
        with ProgressTracker(total=100, description="test", force_mode="heartbeat") as pt:
            for _ in range(100):
                pt.advance(1)

    heartbeats = [e for e in cap if e.get("event") == "progress_heartbeat"]
    # Expect exactly 10 heartbeats: 10, 20, ..., 100
    assert len(heartbeats) == 10
    pct_values = sorted(round(e["pct"] * 100) for e in heartbeats)
    assert pct_values == list(range(10, 101, 10))


def test_heartbeat_extra_fields() -> None:
    """Extra kwargs passed to advance() appear in the heartbeat log event."""
    with capture_logs() as cap:
        with ProgressTracker(total=10, description="extra", force_mode="heartbeat") as pt:
            for _ in range(10):
                pt.advance(1, model="gpt")

    heartbeats = [e for e in cap if e.get("event") == "progress_heartbeat"]
    assert len(heartbeats) > 0
    # All heartbeats should carry the extra field
    assert all(e.get("model") == "gpt" for e in heartbeats)


def test_heartbeat_always_emits_at_completion() -> None:
    """Final heartbeat fires at 100% even when total doesn't divide evenly into 10% steps."""
    with capture_logs() as cap:
        with ProgressTracker(total=7, description="odd", force_mode="heartbeat") as pt:
            for _ in range(7):
                pt.advance(1)

    heartbeats = [e for e in cap if e.get("event") == "progress_heartbeat"]
    # Last heartbeat must be at completed == total
    final = heartbeats[-1]
    assert final["completed"] == 7
    assert final["total"] == 7
    assert final["pct"] == pytest.approx(1.0, rel=1e-3)


# ---------------------------------------------------------------------------
# Rich mode (TTY)
# ---------------------------------------------------------------------------


def test_rich_mode_no_crash() -> None:
    """Rich mode advances without raising exceptions (visual output not asserted)."""
    with ProgressTracker(total=10, description="rich_test", force_mode="rich") as pt:
        for _ in range(10):
            pt.advance(1)


# ---------------------------------------------------------------------------
# Context manager forms
# ---------------------------------------------------------------------------


def test_context_manager_sync() -> None:
    """Sync ``with`` context manager enters and exits cleanly."""
    with ProgressTracker(total=5, description="sync", force_mode="heartbeat") as pt:
        pt.advance(5)
    # No exception = pass


def test_context_manager_async() -> None:
    """Async ``async with`` context manager enters and exits cleanly."""

    async def _run() -> None:
        async with ProgressTracker(total=5, description="async", force_mode="heartbeat") as pt:
            pt.advance(5)

    import asyncio

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_zero_total() -> None:
    """total=0 enters and exits without raising."""
    with ProgressTracker(total=0, description="zero", force_mode="heartbeat"):
        pass  # nothing to advance


def test_update_description() -> None:
    """update_description() can be called mid-run without crashing."""
    with ProgressTracker(total=10, description="before", force_mode="heartbeat") as pt:
        pt.advance(5)
        pt.update_description("after")
        pt.advance(5)
