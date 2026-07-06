from __future__ import annotations

from pathlib import Path

import structlog

from src.common.log import setup_logging


def test_setup_logging_creates_log_dir(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    setup_logging(log_dir=log_dir)
    assert log_dir.exists()


def test_setup_logging_creates_log_file(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    setup_logging(log_dir=log_dir)
    logger = structlog.get_logger("test")
    logger.info("test message", key="value")
    log_file = log_dir / "polistemics.jsonl"
    assert log_file.exists()
    content = log_file.read_text()
    assert "test message" in content
    assert '"key": "value"' in content


def test_json_log_format(tmp_path: Path) -> None:
    import json

    log_dir = tmp_path / "logs"
    setup_logging(log_dir=log_dir)
    logger = structlog.get_logger("test_json")
    logger.info("structured event", metric="faithfulness", items=42)
    log_file = log_dir / "polistemics.jsonl"
    lines = [line for line in log_file.read_text().strip().split("\n") if line]
    # Should be valid JSON
    for line in lines:
        parsed = json.loads(line)
        assert "event" in parsed
