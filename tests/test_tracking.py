"""Tests for src/common/tracking.py."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.common import tracking


# Reset module state between tests
@pytest.fixture(autouse=True)
def reset_tracking():
    """Reset _tracking_available and _banner_shown between tests."""
    orig_available = tracking._tracking_available
    orig_banner = tracking._banner_shown
    yield
    tracking._tracking_available = orig_available
    tracking._banner_shown = orig_banner


class TestCheckTracking:
    def test_01_returns_false_when_uri_not_set(self, monkeypatch):
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
        result = tracking.check_tracking()
        assert result is False
        assert tracking._tracking_available is False

    def test_02_prints_banner_when_unavailable(self, monkeypatch, capsys):
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
        tracking.check_tracking()
        out = capsys.readouterr().out
        assert "OBSERVABILITY NOT CONFIGURED" in out

    def test_03_returns_true_when_mlflow_available(self, monkeypatch):
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
        mock_mlflow = MagicMock()
        with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
            result = tracking.check_tracking()
        assert result is True
        assert tracking._tracking_available is True


class TestStartRun:
    def test_04_noops_when_tracking_unavailable(self):
        tracking._tracking_available = False
        # Should not raise, should not call mlflow
        with tracking.start_run("test_run", tags={"key": "val"}):
            pass  # just runs without error

    def test_05_log_params_noops_when_unavailable(self):
        tracking._tracking_available = False
        tracking.log_params({"a": "1", "b": "2"})  # must not raise

    def test_06_log_metrics_noops_when_unavailable(self):
        tracking._tracking_available = False
        tracking.log_metrics({"n_items": 5.0, "duration_s": 1.2})  # must not raise

    def test_07_log_artifact_skips_nonexistent_path_when_available(self, tmp_path):
        """log_artifact warns and no-ops when path does not exist, even if tracking is on."""
        tracking._tracking_available = True
        mock_mlflow = MagicMock()
        with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
            tracking.log_artifact(tmp_path / "does_not_exist.yaml")  # must not raise
        mock_mlflow.log_artifact.assert_not_called()

    def test_08_log_metrics_swallows_exceptions(self, monkeypatch):
        tracking._tracking_available = True
        mock_mlflow = MagicMock()
        mock_mlflow.log_metrics.side_effect = Exception("bad metric name")
        with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
            tracking.log_metrics({"bad metric!": 1.0})  # must not raise

    def test_09_start_run_exception_still_yields(self):
        """start_run() yields even if mlflow.start_run() raises."""
        tracking._tracking_available = True
        mock_mlflow = MagicMock()
        mock_mlflow.start_run.side_effect = Exception("connection refused")
        executed = False
        ctx = patch.dict("sys.modules", {"mlflow": mock_mlflow})
        with ctx, tracking.start_run("test", tags={}):
            executed = True  # must be reached
        assert executed

    def test_10_banner_shown_only_once(self, monkeypatch, capsys):
        """Banner is printed at most once even if check_tracking() is called multiple times."""
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
        tracking._banner_shown = False
        tracking.check_tracking()
        tracking.check_tracking()
        tracking.check_tracking()
        out = capsys.readouterr().out
        assert out.count("OBSERVABILITY NOT CONFIGURED") == 1
