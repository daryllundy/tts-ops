"""
Unit tests for performance regression detection edge cases.

Tests baseline creation, malformed JSON handling, and missing metric fields.
**Validates: Requirements 6.5**
"""

import json
import sys
import tempfile
from pathlib import Path

import pytest

# Add scripts directory to path
scripts_dir = Path(__file__).parent.parent.parent / "scripts"
sys.path.insert(0, str(scripts_dir))

from check_performance_regression import (
    RegressionResult,
    check_regressions,
    load_metrics,
    save_metrics,
)


def create_valid_metrics():
    """Create a valid metrics dictionary for testing."""
    return {
        "timestamp": "2024-12-04T12:00:00Z",
        "git_sha": "abc123",
        "workflow_run_id": "123456789",
        "metrics": {
            "ttfa_ms": {
                "mean": 450.2,
                "p50": 445.0,
                "p95": 520.0,
                "p99": 580.0,
            },
            "e2e_latency_ms": {
                "mean": 1250.5,
                "p50": 1200.0,
                "p95": 1450.0,
                "p99": 1600.0,
            },
            "throughput_rps": 12.5,
            "error_rate": 0.002,
            "total_requests": 1000,
        }
    }


class TestBaselineCreation:
    """Tests for baseline file creation when none exists."""

    def test_save_metrics_creates_file(self):
        """Test that save_metrics creates a new file with valid JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            baseline_path = Path(tmpdir) / "baseline.json"
            metrics = create_valid_metrics()

            # Save metrics
            save_metrics(baseline_path, metrics)

            # Verify file was created
            assert baseline_path.exists()

            # Verify content is valid JSON
            with open(baseline_path) as f:
                loaded = json.load(f)

            assert loaded == metrics

    def test_save_metrics_creates_parent_directories(self):
        """Test that save_metrics creates parent directories if they don't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            baseline_path = Path(tmpdir) / "nested" / "dir" / "baseline.json"
            metrics = create_valid_metrics()

            # Save metrics (should create nested directories)
            save_metrics(baseline_path, metrics)

            # Verify file was created
            assert baseline_path.exists()
            assert baseline_path.parent.exists()

    def test_baseline_creation_workflow(self):
        """Test the complete workflow of creating a baseline when none exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            baseline_path = Path(tmpdir) / "baseline.json"
            current_metrics = create_valid_metrics()

            # Simulate the workflow: check if baseline exists, create if not
            if not baseline_path.exists():
                save_metrics(baseline_path, current_metrics)

            # Verify baseline was created
            assert baseline_path.exists()

            # Load and verify
            loaded = load_metrics(baseline_path)
            assert loaded == current_metrics


class TestMalformedJSON:
    """Tests for handling malformed JSON files."""

    def test_load_metrics_with_invalid_json(self):
        """Test that load_metrics exits with error on invalid JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            invalid_file = Path(tmpdir) / "invalid.json"

            # Write invalid JSON
            with open(invalid_file, "w") as f:
                f.write("{ invalid json content }")

            # Should exit with error
            with pytest.raises(SystemExit) as exc_info:
                load_metrics(invalid_file)

            assert exc_info.value.code == 1

    def test_load_metrics_with_empty_file(self):
        """Test that load_metrics handles empty files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            empty_file = Path(tmpdir) / "empty.json"

            # Write empty file
            with open(empty_file, "w") as f:
                f.write("")

            # Should exit with error
            with pytest.raises(SystemExit) as exc_info:
                load_metrics(empty_file)

            assert exc_info.value.code == 1

    def test_load_metrics_with_nonexistent_file(self):
        """Test that load_metrics exits with error when file doesn't exist."""
        nonexistent_file = Path("/tmp/nonexistent_file_12345.json")

        # Should exit with error
        with pytest.raises(SystemExit) as exc_info:
            load_metrics(nonexistent_file)

        assert exc_info.value.code == 1


class TestMissingMetricFields:
    """Tests for handling missing metric fields."""

    def test_check_regressions_with_missing_ttfa_fields(self):
        """Test that check_regressions handles missing TTFA fields gracefully."""
        baseline = create_valid_metrics()
        current = create_valid_metrics()

        # Remove some TTFA fields from current
        del current["metrics"]["ttfa_ms"]["p95"]
        del current["metrics"]["ttfa_ms"]["p99"]

        # Should not crash
        result = check_regressions(current, baseline)

        # Should still check other fields
        assert isinstance(result, RegressionResult)

    def test_check_regressions_with_missing_e2e_fields(self):
        """Test that check_regressions handles missing e2e latency fields gracefully."""
        baseline = create_valid_metrics()
        current = create_valid_metrics()

        # Remove some e2e fields from baseline
        del baseline["metrics"]["e2e_latency_ms"]["mean"]
        del baseline["metrics"]["e2e_latency_ms"]["p50"]

        # Should not crash
        result = check_regressions(current, baseline)

        # Should still check other fields
        assert isinstance(result, RegressionResult)

    def test_check_regressions_with_missing_error_rate(self):
        """Test that check_regressions handles missing error_rate field."""
        baseline = create_valid_metrics()
        current = create_valid_metrics()

        # Remove error_rate from current
        del current["metrics"]["error_rate"]

        # Should not crash
        result = check_regressions(current, baseline)

        # Should still check latency fields
        assert isinstance(result, RegressionResult)

    def test_check_regressions_with_missing_entire_metric(self):
        """Test that check_regressions handles missing entire metric sections."""
        baseline = create_valid_metrics()
        current = create_valid_metrics()

        # Remove entire ttfa_ms section from current
        del current["metrics"]["ttfa_ms"]

        # Should not crash
        result = check_regressions(current, baseline)

        # Should still check other metrics
        assert isinstance(result, RegressionResult)

    def test_check_regressions_with_empty_metrics_section(self):
        """Test that check_regressions handles empty metrics section."""
        baseline = create_valid_metrics()
        current = {
            "timestamp": "2024-12-04T12:00:00Z",
            "git_sha": "abc123",
            "workflow_run_id": "123456789",
            "metrics": {}
        }

        # Should not crash
        result = check_regressions(current, baseline)

        # Should have no regressions (nothing to check)
        assert not result.has_regressions()

    def test_check_regressions_with_missing_metrics_section(self):
        """Test that check_regressions handles missing metrics section."""
        baseline = create_valid_metrics()
        current = {
            "timestamp": "2024-12-04T12:00:00Z",
            "git_sha": "abc123",
            "workflow_run_id": "123456789",
        }

        # Should not crash
        result = check_regressions(current, baseline)

        # Should have no regressions (nothing to check)
        assert not result.has_regressions()


class TestZeroBaselineValues:
    """Tests for handling zero baseline values (edge case for division)."""

    def test_check_regressions_with_zero_baseline_latency(self):
        """Test that check_regressions handles zero baseline latency values."""
        baseline = create_valid_metrics()
        current = create_valid_metrics()

        # Set baseline to zero
        baseline["metrics"]["ttfa_ms"]["mean"] = 0.0
        current["metrics"]["ttfa_ms"]["mean"] = 100.0

        # Should detect as regression (any increase from zero is infinite)
        result = check_regressions(current, baseline)

        assert result.has_regressions()

        # Check that the regression was detected
        regression_metrics = {(r["metric"], r["field"]) for r in result.regressions}
        assert ("ttfa_ms", "mean") in regression_metrics

    def test_check_regressions_with_zero_baseline_and_zero_current(self):
        """Test that check_regressions handles both zero baseline and current values."""
        baseline = create_valid_metrics()
        current = create_valid_metrics()

        # Set both to zero
        baseline["metrics"]["ttfa_ms"]["mean"] = 0.0
        current["metrics"]["ttfa_ms"]["mean"] = 0.0

        # Should not detect regression (no change)
        result = check_regressions(current, baseline)

        # Should pass this check
        passed_metrics = {(p["metric"], p["field"]) for p in result.passed}
        assert ("ttfa_ms", "mean") in passed_metrics


class TestRegressionResultClass:
    """Tests for the RegressionResult class."""

    def test_regression_result_initialization(self):
        """Test that RegressionResult initializes correctly."""
        result = RegressionResult()

        assert result.regressions == []
        assert result.warnings == []
        assert result.passed == []
        assert not result.has_regressions()

    def test_add_regression(self):
        """Test adding a regression to the result."""
        result = RegressionResult()

        result.add_regression(
            metric_name="ttfa_ms",
            field="mean",
            baseline_value=100.0,
            current_value=120.0,
            threshold=10.0,
            change_pct=20.0,
        )

        assert result.has_regressions()
        assert len(result.regressions) == 1
        assert result.regressions[0]["metric"] == "ttfa_ms"
        assert result.regressions[0]["field"] == "mean"

    def test_add_passed(self):
        """Test adding a passed check to the result."""
        result = RegressionResult()

        result.add_passed(
            metric_name="ttfa_ms",
            field="mean",
            baseline_value=100.0,
            current_value=105.0,
            change_pct=5.0,
        )

        assert not result.has_regressions()
        assert len(result.passed) == 1
        assert result.passed[0]["metric"] == "ttfa_ms"

    def test_to_dict(self):
        """Test converting RegressionResult to dictionary."""
        result = RegressionResult()

        result.add_regression(
            metric_name="ttfa_ms",
            field="mean",
            baseline_value=100.0,
            current_value=120.0,
            threshold=10.0,
            change_pct=20.0,
        )

        result.add_passed(
            metric_name="e2e_latency_ms",
            field="mean",
            baseline_value=1000.0,
            current_value=1050.0,
            change_pct=5.0,
        )

        result_dict = result.to_dict()

        assert "has_regressions" in result_dict
        assert "regressions" in result_dict
        assert "warnings" in result_dict
        assert "passed" in result_dict
        assert result_dict["has_regressions"] is True
        assert len(result_dict["regressions"]) == 1
        assert len(result_dict["passed"]) == 1
