from __future__ import annotations

import asyncio
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Callable, Generic, TypeVar

import structlog
from pydantic import BaseModel, TypeAdapter, model_validator
from pydantic_ai import Agent, ModelHTTPError, NativeOutput, PromptedOutput
from pydantic_ai.messages import ModelResponse
from pydantic_ai.models import Model

from src.common.models.config import ModelConfig
from src.common.models.factory import create_model
from src.common.models.retry import compute_backoff, parse_retry_after
from src.common.models.settings import build_model_settings

logger = structlog.get_logger(__name__)

T = TypeVar("T", bound=BaseModel)

_MODEL_RESPONSE_TA: TypeAdapter[ModelResponse] = TypeAdapter(ModelResponse)

# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

_RATE_LIMIT_BACKOFF_BASE: float = 10.0
_RATE_LIMIT_BACKOFF_MAX: float = 120.0


class SingleJudgeResponse(BaseModel, Generic[T]):
    """Result from one judge model for a single evaluation call."""

    model_id: str
    result: T | None = None
    error: str | None = None
    latency_ms: float
    prompt: dict[str, str] | None = None
    raw_response: dict | None = None
    native_reasoning: str | None = None
    logprobs: list[dict] | None = None


class PanelResult(BaseModel, Generic[T]):
    """Aggregated result from running all panel judges."""

    responses: list[SingleJudgeResponse[T]]
    n_succeeded: int = 0
    n_failed: int = 0

    @model_validator(mode="after")
    def _compute_counts(self) -> PanelResult[T]:
        self.n_succeeded = sum(1 for r in self.responses if r.result is not None)
        self.n_failed = sum(1 for r in self.responses if r.result is None)
        return self


class PartialPanelResult(Exception):
    """Raised when fewer judges than expected returned a result.

    Carries the partial responses so the pipeline can persist them to
    incomplete.jsonl and reuse the already-succeeded judge results on retry.
    """

    def __init__(self, n_succeeded: int, n_expected: int, responses: list | None = None) -> None:
        self.n_succeeded = n_succeeded
        self.n_expected = n_expected
        self.responses: list = responses or []
        super().__init__(
            f"Partial panel result: {n_succeeded}/{n_expected} judges succeeded"
        )


# ---------------------------------------------------------------------------
# Response extraction helpers
# ---------------------------------------------------------------------------


def _extract_response_details(
    messages: list | tuple,
) -> tuple[dict | None, str | None]:
    """Extract raw response dict and native reasoning from message history.

    Walks messages in reverse to find the last ModelResponse. Returns
    (raw_response, native_reasoning) where native_reasoning is the
    concatenation of all ThinkingPart contents, or None if absent.
    """
    for msg in reversed(messages):
        if not isinstance(msg, ModelResponse):
            continue

        raw_resp = _MODEL_RESPONSE_TA.dump_python(msg, mode="json")

        thinking_parts = [
            p.content
            for p in msg.parts
            if p.part_kind == "thinking" and hasattr(p, "content")
        ]
        reasoning_text = "\n".join(thinking_parts) if thinking_parts else None

        return raw_resp, reasoning_text

    return None, None


# ---------------------------------------------------------------------------
# Cache key type
# ---------------------------------------------------------------------------

type _AgentKey = tuple[str, str, str]


# ---------------------------------------------------------------------------
# JudgePanel
# ---------------------------------------------------------------------------


class JudgePanel:
    """Run evaluation prompts across a panel of LLM judge models.

    Each model is called in parallel with per-model concurrency control.
    Agents are lazily initialised and cached by (model_id, output_type, system_prompt).
    """

    def __init__(
        self,
        model_ids: list[str],
        judge_model_configs: dict[str, ModelConfig] | None = None,
        temperature: float = 0.0,
        max_concurrent_per_model: int = 5,
        output_dir: Path | None = None,
        judge_timeout_seconds: float | None = None,
        max_retries: int = 2,
        retry_backoff_base: float = 2.0,
        retry_backoff_max: float = 60.0,
    ) -> None:
        self._model_ids = model_ids
        self._judge_model_configs = judge_model_configs or {}
        self._temperature = temperature
        self._output_dir = output_dir
        self._judge_timeout_seconds = judge_timeout_seconds
        self._max_retries = max_retries
        self._retry_backoff_base = retry_backoff_base
        self._retry_backoff_max = retry_backoff_max
        self._semaphores: dict[str, asyncio.Semaphore] = {
            mid: asyncio.Semaphore(max_concurrent_per_model) for mid in model_ids
        }
        self._agents: dict[_AgentKey, Agent] = {}

    @property
    def model_ids(self) -> list[str]:
        """Return the list of judge model IDs in this panel."""
        return self._model_ids

    # --- Lazy agent construction ---

    def _get_agent(self, model_id: str, output_type: type[T], system_prompt: str) -> Agent:
        """Return a cached PydanticAI Agent, creating one lazily if needed."""
        key: _AgentKey = (model_id, output_type.__name__, system_prompt)
        if key not in self._agents:
            mc = self._judge_model_configs.get(model_id)
            if mc is not None:
                mode = mc.output_mode or "tool"
                model: Model | str = create_model(mc)
            else:
                # No ModelConfig — fall back to pydantic-ai native string routing.
                # Used in tests and when judge_model_configs is not provided.
                mode = "tool"
                model = model_id
            self._agents[key] = Agent(
                model=model,
                retries=3,
                output_type=self._wrap_output_type(output_type, mode),
                system_prompt=system_prompt,
            )
        return self._agents[key]

    @staticmethod
    def _wrap_output_type(output_type: type[T], mode: str) -> object:
        """Wrap output_type in the appropriate PydanticAI output marker."""
        _DESC = "Answer each boolean field according to the evaluation question in its description."
        if mode == "native":
            return NativeOutput(output_type, description=_DESC)
        if mode == "prompted":
            return PromptedOutput(output_type, description=_DESC)
        return output_type  # tool mode: plain class

    # --- Core evaluation ---

    async def evaluate(
        self,
        user_prompt: str,
        output_type: type[T],
        system_prompt: str = "",
        skip_model_ids: set[str] | None = None,
    ) -> PanelResult[T]:
        """Run panel judges in parallel and collect results.

        Args:
            user_prompt: The judge user prompt.
            output_type: Pydantic model class for structured output.
            system_prompt: The judge system prompt.
            skip_model_ids: If provided, skip these model IDs (already have responses).
        """
        tasks = [
            self._run_single(mid, user_prompt, output_type, system_prompt)
            for mid in self._model_ids
            if skip_model_ids is None or mid not in skip_model_ids
        ]
        responses = await asyncio.gather(*tasks)
        return PanelResult(responses=list(responses))

    def _build_judge_settings(self, model_id: str) -> dict[str, object]:
        """Build model_settings for a judge call from the judge's ModelConfig.

        Delegates all routing logic (reasoning, provider ordering, token limits)
        to build_model_settings() in model_call.py — the same function used by
        the collection pipeline. Temperature is added here at the panel level.
        """
        settings: dict[str, object] = {"temperature": self._temperature}
        mc = self._judge_model_configs.get(model_id)
        if mc is not None:
            settings.update(build_model_settings(mc))
        return settings

    async def _run_single(
        self,
        model_id: str,
        user_prompt: str,
        output_type: type[T],
        system_prompt: str,
    ) -> SingleJudgeResponse[T]:
        """Call a single judge model with semaphore gating and backoff retries.

        Error handling mirrors the collection pipeline:
        - 401/403 auth errors → fail immediately, do not retry
        - 429 rate limit → exponential backoff with rate-limit constants
        - timeout / transient network → exponential backoff with regular constants
        - other HTTP errors → fail immediately, do not retry

        The semaphore slot is released before sleeping so other coroutines can
        proceed while this one waits.
        """


        agent = self._get_agent(model_id, output_type, system_prompt)
        sem = self._semaphores[model_id]
        settings = self._build_judge_settings(model_id)

        last_error: str = ""
        latency_ms: float = 0.0

        for attempt in range(self._max_retries + 1):
            should_retry = False
            is_rate_limit = False
            retry_after_hint: float | None = None

            # Semaphore block ends BEFORE the sleep so the slot is released
            # while waiting — critical for correct concurrency behaviour.
            async with sem:
                t0 = time.perf_counter()
                try:
                    if self._judge_timeout_seconds is not None:
                        run_result = await asyncio.wait_for(
                            agent.run(user_prompt, model_settings=settings),
                            timeout=self._judge_timeout_seconds,
                        )
                    else:
                        run_result = await agent.run(user_prompt, model_settings=settings)

                    latency_ms = (time.perf_counter() - t0) * 1_000
                    logger.debug("judge_succeeded", model_id=model_id, latency_ms=round(latency_ms, 1))

                    raw_resp, reasoning_text = _extract_response_details(run_result.all_messages())
                    return SingleJudgeResponse(
                        model_id=model_id,
                        result=run_result.output,
                        latency_ms=latency_ms,
                        prompt={"system": system_prompt, "user": user_prompt},
                        raw_response=raw_resp,
                        native_reasoning=reasoning_text,
                        logprobs=None,
                    )

                except asyncio.TimeoutError:
                    latency_ms = (time.perf_counter() - t0) * 1_000
                    last_error = f"timeout after {self._judge_timeout_seconds}s"
                    should_retry = True
                    logger.warning("judge_timeout", model_id=model_id, attempt=attempt + 1)

                except ModelHTTPError as exc:
                    latency_ms = (time.perf_counter() - t0) * 1_000
                    last_error = str(exc)
                    if exc.status_code in (401, 403):
                        # Auth errors are not transient — fail immediately.
                        logger.error(
                            "judge_auth_error",
                            model_id=model_id,
                            status_code=exc.status_code,
                            error=last_error[:200],
                        )
                    elif exc.status_code == 429:
                        is_rate_limit = True
                        should_retry = True
                        retry_after_hint = parse_retry_after(exc)
                        logger.warning("judge_rate_limited", model_id=model_id, attempt=attempt + 1, error=last_error[:200])
                    else:
                        # Other HTTP errors (5xx, etc.) — don't retry
                        logger.error("judge_failed", model_id=model_id, attempt=attempt + 1, error=last_error[:200])

                except (TimeoutError, ConnectionError, OSError) as exc:
                    latency_ms = (time.perf_counter() - t0) * 1_000
                    last_error = str(exc)
                    should_retry = True
                    logger.warning("judge_transient_error", model_id=model_id, attempt=attempt + 1, error=last_error[:200])

                except Exception as exc:
                    latency_ms = (time.perf_counter() - t0) * 1_000
                    last_error = str(exc)
                    logger.error("judge_failed", model_id=model_id, attempt=attempt + 1, error=last_error[:200])

            # Semaphore released — sleep outside it before next attempt.
            is_last_attempt = attempt == self._max_retries
            if should_retry and not is_last_attempt:
                if is_rate_limit:
                    if retry_after_hint is not None:
                        delay = min(retry_after_hint, _RATE_LIMIT_BACKOFF_MAX)
                        delay_source = "retry_after_header"
                    else:
                        delay = compute_backoff(
                            attempt,
                            base=self._retry_backoff_base,
                            cap=_RATE_LIMIT_BACKOFF_MAX,
                            multiplier=_RATE_LIMIT_BACKOFF_BASE,
                        )
                        delay_source = "exponential_backoff"
                else:
                    delay = compute_backoff(
                        attempt,
                        base=self._retry_backoff_base,
                        cap=self._retry_backoff_max,
                    )
                    delay_source = "exponential_backoff"
                logger.debug(
                    "judge_retry_sleep",
                    model_id=model_id,
                    attempt=attempt + 1,
                    delay=round(delay, 2),
                    delay_source=delay_source,
                )
                await asyncio.sleep(delay)
            elif not should_retry:
                break

        return SingleJudgeResponse(model_id=model_id, error=last_error, latency_ms=latency_ms)

    # --- Aggregation helpers ---

    def majority_vote(
        self,
        responses: list[SingleJudgeResponse[T]],
        key_fn: Callable[[T], str],
    ) -> str | None:
        """Return the most common key across successful responses.

        Tie-break: first in sorted order (deterministic).
        """
        keys = [key_fn(r.result) for r in responses if r.result is not None]
        if not keys:
            return None

        counts = Counter(keys)
        max_count = counts.most_common(1)[0][1]
        tied = sorted(k for k, c in counts.items() if c == max_count)
        return tied[0]

    def average(
        self,
        responses: list[SingleJudgeResponse[T]],
        key_fn: Callable[[T], float],
    ) -> float | None:
        """Return the mean value across successful responses."""
        values = [key_fn(r.result) for r in responses if r.result is not None]
        if not values:
            return None
        return mean(values)

    # --- Audit logging ---

    async def write_audit(
        self,
        metric_name: str,
        observation_id: str,
        responses: list[SingleJudgeResponse],
    ) -> None:
        """Append judge responses as JSONL to the audit log file.

        Skips silently if no output_dir is configured.
        """
        if self._output_dir is None:
            return

        self._output_dir.mkdir(parents=True, exist_ok=True)
        audit_path = self._output_dir / f"{metric_name}.jsonl"
        ts = datetime.now(timezone.utc).isoformat()

        lines: list[str] = []
        for resp in responses:
            record = {
                "observation_id": observation_id,
                "model_id": resp.model_id,
                "prompt": resp.prompt,
                "result": resp.result.model_dump() if resp.result is not None else None,
                "raw_response": resp.raw_response,
                "native_reasoning": resp.native_reasoning,
                "logprobs": resp.logprobs,
                "error": resp.error,
                "latency_ms": round(resp.latency_ms, 2),
                "timestamp": ts,
            }
            lines.append(json.dumps(record, ensure_ascii=False))

        with audit_path.open("a", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
