"""Lightweight MLFlow tracking wrapper.

All functions gracefully no-op when MLFlow is unavailable or not configured.
Never import mlflow at module level — always import inside try/except ImportError.
"""
from __future__ import annotations

import contextlib
import os
from collections.abc import Generator
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

_tracking_available: bool = False  # set by check_tracking()
_banner_shown: bool = False  # printed at most once per process

# ANSI yellow for the banner
_YELLOW = "\033[93m"
_RESET = "\033[0m"

_BANNER = f"""\
{_YELLOW}┌─────────────────────────────────────────────────────────────┐
│  ⚠️  OBSERVABILITY NOT CONFIGURED — this run will not be   │
│     tracked in MLFlow.                                       │
│                                                              │
│  To enable:                                                  │
│    export MLFLOW_TRACKING_URI=http://127.0.0.1:5000          │
│    mlflow server --host 127.0.0.1 --port 5000                │
│                                                              │
│  Continuing without observability...                         │
└─────────────────────────────────────────────────────────────┘{_RESET}"""


def check_tracking() -> bool:
    """Check whether MLFlow tracking is available and configured.

    Sets module-level _tracking_available flag. Prints a yellow banner
    if tracking is unavailable — at most once per process regardless of
    how many times this is called.

    Returns:
        True if MLFlow is available and a tracking URI is configured.
    """
    global _tracking_available, _banner_shown

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "")
    if not tracking_uri:
        if not _banner_shown:
            print(_BANNER)
            _banner_shown = True
        logger.warning("mlflow_tracking_not_configured")
        _tracking_available = False
        return False

    try:
        import mlflow  # noqa: F401
        _tracking_available = True
        logger.info("mlflow_tracking_configured", uri=tracking_uri)
        return True
    except ImportError:
        print(_BANNER)
        logger.warning("mlflow_not_installed", hint="uv pip install -e '.[observability]'")
        _tracking_available = False
        return False


@contextlib.contextmanager
def start_run(
    run_name: str,
    tags: dict[str, str] | None = None,
) -> Generator[None, None, None]:
    """Context manager wrapping mlflow.start_run().

    No-ops gracefully when tracking is unavailable. Tags are string key-value pairs
    used to filter runs in the MLFlow UI.

    Args:
        run_name: Human-readable name for this run.
        tags: String key-value pairs for filtering and grouping runs.

    Yields:
        Nothing — use tracking.log_* functions inside the context.
    """
    if not _tracking_available:
        yield
        return

    try:
        import mlflow
        with mlflow.start_run(run_name=run_name, tags=tags or {}):
            yield
    except Exception as e:
        logger.warning("mlflow_start_run_failed", error=str(e))
        yield


def log_params(params: dict[str, Any]) -> None:
    """Log run parameters. No-ops when tracking unavailable."""
    if not _tracking_available:
        return
    try:
        import mlflow
        mlflow.log_params(params)
    except Exception as e:
        logger.warning("mlflow_log_params_failed", error=str(e))


def log_metrics(metrics: dict[str, float]) -> None:
    """Log run metrics. No-ops when tracking unavailable. Swallows MLFlow errors."""
    if not _tracking_available:
        return
    try:
        import mlflow
        mlflow.log_metrics(metrics)
    except Exception as e:
        logger.warning("mlflow_log_metrics_failed", error=str(e))


def log_artifact(path: Path) -> None:
    """Log a file artifact. No-ops when path doesn't exist or tracking unavailable."""
    if not _tracking_available:
        return
    if not path.exists():
        logger.warning("mlflow_artifact_not_found", path=str(path))
        return
    try:
        import mlflow
        mlflow.log_artifact(str(path))
    except Exception as e:
        logger.warning("mlflow_log_artifact_failed", path=str(path), error=str(e))
