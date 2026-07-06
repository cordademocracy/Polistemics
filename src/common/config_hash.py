"""Deterministic 8-char hex fingerprints for experiment configurations.

Pure computation — no I/O except the template file reads required to build
the hash inputs, and the lock-file helpers at the bottom.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.common.config import JudgesConfig, ModelConfig
    from src.common.models.config import ConditionConfig, PromptConfig
    from src.metrics.rubric import BaseRubric


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _md5_hex8(d: dict) -> str:
    """Return the first 8 hex chars of the MD5 of a deterministically serialised dict."""
    raw = json.dumps(d, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.md5(raw).hexdigest()[:8]


# ---------------------------------------------------------------------------
# Public hash functions
# ---------------------------------------------------------------------------


def compute_collection_hash(
    model_id: str,
    model_config: "ModelConfig",
    prompts_config: "PromptConfig",
    templates_dir: Path,
    conditions: "ConditionConfig",
) -> str:
    """Fingerprint the inputs that determine what a generator model will produce.

    Captures model identity, prompt wording, and experimental conditions.
    Explicitly excludes operational concerns (api_key_env, provider, concurrency,
    retries) and other models' settings.

    Args:
        model_id: Logical identifier of the generator model.
        model_config: Full ModelConfig for that model.
        prompts_config: PromptConfig carrying system_prompt filename and templates map.
        templates_dir: Directory that contains the prompt template files.
        conditions: ConditionConfig for this experiment run.

    Returns:
        8-character lowercase hex string.
    """
    system_prompt_text = (templates_dir / prompts_config.system_prompt).read_text(encoding="utf-8")

    user_templates: dict[str, str] = {
        variation_name: (templates_dir / filename).read_text(encoding="utf-8")
        for variation_name, filename in sorted(prompts_config.templates.items())
    }

    d: dict = {
        "model_id": model_id,
        "model": model_config.model,
        "reasoning_effort": model_config.reasoning_effort,
        "reasoning_max_tokens": model_config.reasoning_max_tokens,
        "max_tokens": model_config.max_tokens,
        "system_prompt": system_prompt_text,
        "user_templates": user_templates,
        "response_language": conditions.response_language,
        "party_label": conditions.party_label,
        "year_mention": conditions.year_mention,
    }
    return _md5_hex8(d)


def compute_judge_hash(
    judges_config: "JudgesConfig",
    judge_model_configs: "dict[str, ModelConfig]",
    metrics: "list[BaseRubric]",
) -> str:
    """Fingerprint the judge panel + rubric structure.

    Args:
        judges_config: JudgesConfig with panel list and temperature.
        judge_model_configs: Full map of model_id → ModelConfig (may contain
            non-judge models; filtered to panel members internally).
        templates_dir: Directory that contains ``judge.txt`` (unused; kept for
            API compatibility).
        metrics: Instantiated rubric objects whose structural fingerprint is captured.

    Returns:
        8-character lowercase hex string.
    """
    panel_sorted = sorted(judges_config.panel)

    judges_dict: dict[str, dict] = {
        model_id: {
            "model": cfg.model,
            "reasoning_effort": cfg.reasoning_effort,
            "reasoning_max_tokens": cfg.reasoning_max_tokens,
            "max_tokens": cfg.max_tokens,
            "output_mode": cfg.output_mode,
        }
        for model_id, cfg in sorted(judge_model_configs.items())
        if model_id in judges_config.panel
    }

    rubrics_dict: dict[str, dict] = {
        metric.name: {
            "sub_questions": [
                {
                    "id": q.id,
                    # frozenset is not JSON-serialisable — normalise to sorted list or None
                    "active_ies": sorted(q.active_ies) if q.active_ies is not None else None,
                }
                for q in (
                    list(metric.shared_items)
                    + [q for qs in metric.condition_specific.values() for q in qs]
                )
            ]
        }
        for metric in metrics
    }

    d: dict = {
        "panel": panel_sorted,
        "temperature": judges_config.temperature,
        "judges": judges_dict,
        "rubrics": rubrics_dict,
    }
    return _md5_hex8(d)


# ---------------------------------------------------------------------------
# Lock-file helpers
# ---------------------------------------------------------------------------


def write_lock_file(path: Path, data: dict) -> None:
    """Write a JSON lock file. Overwrites if exists.

    Args:
        path: Destination file path (parent directory must exist).
        data: Arbitrary JSON-serialisable dict.
    """
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def read_lock_file(path: Path) -> dict | None:
    """Read a JSON lock file.

    Args:
        path: File path to read.

    Returns:
        Parsed dict, or None if the file does not exist.
    """
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
