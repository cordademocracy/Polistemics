"""Shared retry utilities for model API calls.

Provides building blocks used by both the collection pipeline and the judge
panel. The retry loops themselves stay in their respective callers because they
differ in structure (semaphore handling, API key pool rotation). Only the
detection, header parsing, and backoff computation are shared here.
"""
from __future__ import annotations

import random
import re


def is_rate_limit_error(exc: Exception) -> bool:
    """Return True if exc represents an HTTP 429 rate-limit response.

    Checks the status_code attribute first (works for pydantic-ai
    ModelHTTPError and httpx exceptions), then falls back to a string
    match for providers that embed the status in the error message.
    """
    if getattr(exc, "status_code", None) == 429:
        return True
    msg = str(exc).lower()
    return "429" in msg or "rate limit" in msg or "rate_limit" in msg


def parse_retry_after(exc: Exception) -> float | None:
    """Extract the Retry-After delay (seconds) from a rate-limit exception.

    Tries in order:
    1. exc.__cause__.response.headers["Retry-After"]  — httpx response via
       pydantic-ai (ModelHTTPError wraps an httpx.HTTPStatusError whose
       .response carries the headers).
    2. Regex match on str(exc) for "retry-after: <number>" — some providers
       embed the header value in the exception message.

    Returns:
        Delay in seconds, or None if the header is absent or unparseable.
        Callers should fall back to their computed backoff when None.
    """
    try:
        cause = exc.__cause__
        if cause is not None:
            response = getattr(cause, "response", None)
            if response is not None:
                raw = response.headers.get("Retry-After")
                if raw is not None:
                    return float(raw)
    except Exception:  # noqa: BLE001
        pass

    match = re.search(r"retry-after[:\s]+(\d+(?:\.\d+)?)", str(exc), re.IGNORECASE)
    if match:
        return float(match.group(1))

    return None


def compute_backoff(
    attempt: int,
    *,
    base: float,
    cap: float,
    multiplier: float = 1.0,
    jitter: bool = True,
) -> float:
    """Compute exponential backoff delay for a given attempt index.

    Formula: min(multiplier * base^attempt, cap) + uniform jitter in [0, 1).

    Args:
        attempt: Zero-indexed retry attempt number.
        base: Backoff base (e.g. 2.0 for doubling each attempt).
        cap: Maximum delay in seconds.
        multiplier: Scales the exponential term. Use > 1 for rate-limit paths
            where a longer initial delay is appropriate.
        jitter: Add random jitter to spread concurrent retries.

    Returns:
        Delay in seconds, capped at `cap` (plus up to 1s jitter).
    """
    delay = min(multiplier * (base ** attempt), cap)
    if jitter:
        delay += random.random()
    return delay
