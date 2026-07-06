"""Tests for src/common/config_hash.py.

Coverage targets:
  - _md5_hex8: output format
  - compute_collection_hash: deterministic, sensitive to confounding fields,
    insensitive to non-confounding fields (api_key_env, concurrency)
  - compute_judge_hash: deterministic, panel member missing from judge_model_configs
    (silent omission — no crash), template read error propagates
  - write_lock_file / read_lock_file: round-trip, missing file returns None
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from src.common.config_hash import (
    _md5_hex8,
    compute_collection_hash,
    compute_judge_hash,
    read_lock_file,
    write_lock_file,
)
from src.common.models.config import (
    ConditionConfig,
    JudgesConfig,
    ModelConfig,
    PromptConfig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_model_config(**overrides) -> ModelConfig:
    defaults = dict(
        id="model_a",
        provider="openrouter",
        model="openai/gpt-4o",
        max_tokens=1024,
        api_key_env="OPENROUTER_API_KEY",
    )
    return ModelConfig(**{**defaults, **overrides})


def _make_prompt_config(
    system: str = "system.txt",
    templates: dict[str, str] | None = None,
) -> PromptConfig:
    from src.common.schemas import PromptVariation
    return PromptConfig(
        levels=[PromptVariation.DEFAULT],
        system_prompt=system,
        templates=templates or {"default": "default_user.txt"},
    )


def _make_condition() -> ConditionConfig:
    return ConditionConfig(response_language="German", party_label="real", year_mention="none")


def _write_templates(tmp_path: Path, system_text: str = "You are an expert.", user_text: str = "Evaluate this.") -> Path:
    """Write template files into tmp_path and return the templates dir."""
    (tmp_path / "system.txt").write_text(system_text, encoding="utf-8")
    (tmp_path / "default_user.txt").write_text(user_text, encoding="utf-8")
    return tmp_path


def _make_judges_config(panel: list[str] | None = None) -> JudgesConfig:
    return JudgesConfig(panel=panel or ["judge_a", "judge_b"], temperature=0.0)


# ---------------------------------------------------------------------------
# _md5_hex8
# ---------------------------------------------------------------------------


class TestMd5Hex8:
    def test_returns_8_char_lowercase_hex(self) -> None:
        result = _md5_hex8({"key": "value"})
        assert len(result) == 8
        assert re.fullmatch(r"[0-9a-f]{8}", result)

    def test_deterministic(self) -> None:
        d = {"a": 1, "b": [1, 2, 3]}
        assert _md5_hex8(d) == _md5_hex8(d)

    def test_different_inputs_differ(self) -> None:
        assert _md5_hex8({"x": 1}) != _md5_hex8({"x": 2})

    def test_key_order_independent(self) -> None:
        # sort_keys=True must make ordering irrelevant
        assert _md5_hex8({"a": 1, "b": 2}) == _md5_hex8({"b": 2, "a": 1})


# ---------------------------------------------------------------------------
# compute_collection_hash
# ---------------------------------------------------------------------------


class TestComputeCollectionHash:
    def test_returns_8_char_hex(self, tmp_path: Path) -> None:
        _write_templates(tmp_path)
        h = compute_collection_hash(
            model_id="model_a",
            model_config=_make_model_config(),
            prompts_config=_make_prompt_config(),
            templates_dir=tmp_path,
            conditions=_make_condition(),
        )
        assert len(h) == 8
        assert re.fullmatch(r"[0-9a-f]{8}", h)

    def test_deterministic_same_inputs(self, tmp_path: Path) -> None:
        _write_templates(tmp_path)
        args = dict(
            model_id="model_a",
            model_config=_make_model_config(),
            prompts_config=_make_prompt_config(),
            templates_dir=tmp_path,
            conditions=_make_condition(),
        )
        assert compute_collection_hash(**args) == compute_collection_hash(**args)

    def test_changes_when_model_string_changes(self, tmp_path: Path) -> None:
        _write_templates(tmp_path)
        base = dict(
            model_id="model_a",
            prompts_config=_make_prompt_config(),
            templates_dir=tmp_path,
            conditions=_make_condition(),
        )
        h1 = compute_collection_hash(model_config=_make_model_config(model="openai/gpt-4o"), **base)
        h2 = compute_collection_hash(model_config=_make_model_config(model="openai/gpt-4o-mini"), **base)
        assert h1 != h2

    def test_changes_when_model_id_changes(self, tmp_path: Path) -> None:
        _write_templates(tmp_path)
        mc = _make_model_config()
        base = dict(model_config=mc, prompts_config=_make_prompt_config(), templates_dir=tmp_path, conditions=_make_condition())
        h1 = compute_collection_hash(model_id="model_a", **base)
        h2 = compute_collection_hash(model_id="model_b", **base)
        assert h1 != h2

    def test_changes_when_system_prompt_text_changes(self, tmp_path: Path) -> None:
        _write_templates(tmp_path, system_text="Original system prompt.")
        mc = _make_model_config()
        args = dict(model_id="model_a", model_config=mc, prompts_config=_make_prompt_config(), templates_dir=tmp_path, conditions=_make_condition())
        h1 = compute_collection_hash(**args)

        # Rewrite the system prompt
        (tmp_path / "system.txt").write_text("Changed system prompt.", encoding="utf-8")
        h2 = compute_collection_hash(**args)
        assert h1 != h2

    def test_changes_when_user_template_text_changes(self, tmp_path: Path) -> None:
        _write_templates(tmp_path, user_text="Original user prompt.")
        mc = _make_model_config()
        args = dict(model_id="model_a", model_config=mc, prompts_config=_make_prompt_config(), templates_dir=tmp_path, conditions=_make_condition())
        h1 = compute_collection_hash(**args)

        (tmp_path / "default_user.txt").write_text("Changed user prompt.", encoding="utf-8")
        h2 = compute_collection_hash(**args)
        assert h1 != h2

    def test_changes_when_response_language_changes(self, tmp_path: Path) -> None:
        _write_templates(tmp_path)
        mc = _make_model_config()
        base = dict(model_id="model_a", model_config=mc, prompts_config=_make_prompt_config(), templates_dir=tmp_path)
        h1 = compute_collection_hash(conditions=ConditionConfig(response_language="German"), **base)
        h2 = compute_collection_hash(conditions=ConditionConfig(response_language="English"), **base)
        assert h1 != h2

    def test_changes_when_max_tokens_changes(self, tmp_path: Path) -> None:
        """max_tokens affects output — it is a confounding field."""
        _write_templates(tmp_path)
        base = dict(model_id="model_a", prompts_config=_make_prompt_config(), templates_dir=tmp_path, conditions=_make_condition())
        h1 = compute_collection_hash(model_config=_make_model_config(max_tokens=512), **base)
        h2 = compute_collection_hash(model_config=_make_model_config(max_tokens=2048), **base)
        assert h1 != h2

    def test_stable_when_api_key_env_changes(self, tmp_path: Path) -> None:
        """api_key_env is operational — must NOT affect the hash."""
        _write_templates(tmp_path)
        base = dict(model_id="model_a", prompts_config=_make_prompt_config(), templates_dir=tmp_path, conditions=_make_condition())
        h1 = compute_collection_hash(model_config=_make_model_config(api_key_env="KEY_A"), **base)
        h2 = compute_collection_hash(model_config=_make_model_config(api_key_env="KEY_B"), **base)
        assert h1 == h2

    def test_raises_on_missing_template_file(self, tmp_path: Path) -> None:
        """If a template file is absent, FileNotFoundError propagates (fail fast)."""
        # Only write system.txt; omit default_user.txt
        (tmp_path / "system.txt").write_text("sys", encoding="utf-8")
        with pytest.raises(FileNotFoundError):
            compute_collection_hash(
                model_id="model_a",
                model_config=_make_model_config(),
                prompts_config=_make_prompt_config(),
                templates_dir=tmp_path,
                conditions=_make_condition(),
            )


# ---------------------------------------------------------------------------
# compute_judge_hash
# ---------------------------------------------------------------------------


def _make_mock_rubric(name: str = "faithfulness") -> object:
    """Build a minimal object that matches the BaseRubric structural interface
    needed by compute_judge_hash (reads .name, .shared_items, .condition_specific)."""
    from src.metrics.rubric import SubQuestion

    class _MockRubric:
        pass

    r = _MockRubric()
    r.name = name  # type: ignore[attr-defined]
    r.shared_items = [  # type: ignore[attr-defined]
        SubQuestion(id="q_shared", text="t", eval="judge", comparability="shared"),
    ]
    r.condition_specific = {}  # type: ignore[attr-defined]
    return r


class TestComputeJudgeHash:
    def test_returns_8_char_hex(self) -> None:
        judges = _make_judges_config()
        model_configs = {
            "judge_a": _make_model_config(id="judge_a"),
            "judge_b": _make_model_config(id="judge_b"),
        }
        h = compute_judge_hash(
            judges_config=judges,
            judge_model_configs=model_configs,
            metrics=[_make_mock_rubric()],  # type: ignore[list-item]
        )
        assert len(h) == 8
        assert re.fullmatch(r"[0-9a-f]{8}", h)

    def test_deterministic(self) -> None:
        judges = _make_judges_config()
        model_configs = {
            "judge_a": _make_model_config(id="judge_a"),
            "judge_b": _make_model_config(id="judge_b"),
        }
        args = dict(
            judges_config=judges,
            judge_model_configs=model_configs,
            metrics=[_make_mock_rubric()],  # type: ignore[list-item]
        )
        assert compute_judge_hash(**args) == compute_judge_hash(**args)

    def test_changes_when_judge_prompt_changes(self) -> None:
        """Judge hash changes when rubric structure changes (no template file dependency)."""
        judges = _make_judges_config()
        model_configs = {"judge_a": _make_model_config(id="judge_a"), "judge_b": _make_model_config(id="judge_b")}

        rubric_a = _make_mock_rubric("rubric_a")
        rubric_b = _make_mock_rubric("rubric_b")

        h1 = compute_judge_hash(judges_config=judges, judge_model_configs=model_configs, metrics=[rubric_a])  # type: ignore[list-item]
        h2 = compute_judge_hash(judges_config=judges, judge_model_configs=model_configs, metrics=[rubric_b])  # type: ignore[list-item]
        assert h1 != h2

    def test_changes_when_panel_composition_changes(self) -> None:
        mc = {"judge_a": _make_model_config(id="judge_a"), "judge_b": _make_model_config(id="judge_b")}
        rubric = _make_mock_rubric()
        h1 = compute_judge_hash(
            judges_config=_make_judges_config(panel=["judge_a"]),
            judge_model_configs=mc, metrics=[rubric],  # type: ignore[list-item]
        )
        h2 = compute_judge_hash(
            judges_config=_make_judges_config(panel=["judge_a", "judge_b"]),
            judge_model_configs=mc, metrics=[rubric],  # type: ignore[list-item]
        )
        assert h1 != h2

    def test_changes_when_temperature_changes(self) -> None:
        mc = {"judge_a": _make_model_config(id="judge_a"), "judge_b": _make_model_config(id="judge_b")}
        rubric = _make_mock_rubric()
        h1 = compute_judge_hash(
            judges_config=JudgesConfig(panel=["judge_a", "judge_b"], temperature=0.0),
            judge_model_configs=mc, metrics=[rubric],  # type: ignore[list-item]
        )
        h2 = compute_judge_hash(
            judges_config=JudgesConfig(panel=["judge_a", "judge_b"], temperature=0.7),
            judge_model_configs=mc, metrics=[rubric],  # type: ignore[list-item]
        )
        assert h1 != h2

    def test_panel_member_missing_from_judge_model_configs_no_crash(self) -> None:
        """Missing panel member silently omitted from hash — must not raise.

        This covers the graceful degradation case: compute_judge_hash filters
        judge_model_configs to panel members, so a missing entry just means
        that model is not in the fingerprint.
        """
        # judge_b is in panel but NOT in judge_model_configs
        mc = {"judge_a": _make_model_config(id="judge_a")}
        judges = _make_judges_config(panel=["judge_a", "judge_b"])
        h = compute_judge_hash(
            judges_config=judges,
            judge_model_configs=mc,
            metrics=[_make_mock_rubric()],  # type: ignore[list-item]
        )
        # Must produce a valid 8-char hex — no crash
        assert len(h) == 8
        assert re.fullmatch(r"[0-9a-f]{8}", h)

    def test_missing_judge_member_changes_hash_vs_full_panel(self) -> None:
        """Hash with a missing panel member must differ from the complete-panel hash."""
        mc_full = {
            "judge_a": _make_model_config(id="judge_a"),
            "judge_b": _make_model_config(id="judge_b"),
        }
        mc_partial = {"judge_a": _make_model_config(id="judge_a")}  # judge_b missing
        judges = _make_judges_config(panel=["judge_a", "judge_b"])
        rubric = _make_mock_rubric()

        h_full = compute_judge_hash(judges_config=judges, judge_model_configs=mc_full, metrics=[rubric])  # type: ignore[list-item]
        h_partial = compute_judge_hash(judges_config=judges, judge_model_configs=mc_partial, metrics=[rubric])  # type: ignore[list-item]
        assert h_full != h_partial

    def test_stable_when_non_panel_model_changes(self) -> None:
        """Models in judge_model_configs but NOT in panel must not affect hash."""
        judges = _make_judges_config(panel=["judge_a"])
        rubric = _make_mock_rubric()

        # judge_b is not in panel — its config changes should not affect the hash
        mc1 = {
            "judge_a": _make_model_config(id="judge_a"),
            "judge_b": _make_model_config(id="judge_b", model="openai/gpt-4o"),
        }
        mc2 = {
            "judge_a": _make_model_config(id="judge_a"),
            "judge_b": _make_model_config(id="judge_b", model="openai/gpt-4o-mini"),
        }
        h1 = compute_judge_hash(judges_config=judges, judge_model_configs=mc1, metrics=[rubric])  # type: ignore[list-item]
        h2 = compute_judge_hash(judges_config=judges, judge_model_configs=mc2, metrics=[rubric])  # type: ignore[list-item]
        assert h1 == h2


# ---------------------------------------------------------------------------
# write_lock_file / read_lock_file
# ---------------------------------------------------------------------------


class TestLockFileRoundTrip:
    def test_write_then_read_returns_original(self, tmp_path: Path) -> None:
        path = tmp_path / "lock.json"
        data = {"key": "value", "number": 42, "nested": {"a": True}}
        write_lock_file(path, data)
        result = read_lock_file(path)
        assert result == data

    def test_read_missing_file_returns_none(self, tmp_path: Path) -> None:
        result = read_lock_file(tmp_path / "nonexistent.json")
        assert result is None

    def test_write_overwrites_existing(self, tmp_path: Path) -> None:
        path = tmp_path / "lock.json"
        write_lock_file(path, {"v": 1})
        write_lock_file(path, {"v": 2})
        result = read_lock_file(path)
        assert result == {"v": 2}

    def test_written_file_is_valid_json(self, tmp_path: Path) -> None:
        path = tmp_path / "lock.json"
        data = {"collection_hash": "ab12cd34", "model_id": "model_a"}
        write_lock_file(path, data)
        raw = path.read_text(encoding="utf-8")
        parsed = json.loads(raw)
        assert parsed == data

    def test_round_trip_preserves_none_values(self, tmp_path: Path) -> None:
        path = tmp_path / "lock.json"
        data = {"judge_hash": None, "metric_name": "faithfulness"}
        write_lock_file(path, data)
        result = read_lock_file(path)
        assert result["judge_hash"] is None
