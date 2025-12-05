"""
Property-based tests for performance regression detection.

**Feature: github-actions-cicd, Property 1: Regression detection correctly identifies threshold violations**
**Validates: Requirements 6.1, 6.2, 6.3**
"""

import sys
from pathlib import Path

import pytest
from hypothesis import given, strategies as st

# Add scripts directory to path
scripts_dir = Path(__file__).parent.parent.parent / "scripts"
sys.path.insert(0, str(scripts_dir))

from check_performance_regression import check_regressions, THRESHOLDS


# Strategy for generating valid metric values (positive floats)
metric_value = st.floats(min_value=0.1, max_value=10000.0, allow_nan=False, allow_infinity=False)

# Strategy for generating error rates (0.0 to 1.0)
error_rate = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)


def create_metrics_dict(ttfa_mean, ttfa_p50, ttfa_p95, ttfa_p99,
                        e2e_mean, e2e_p50, e2e_p95, e2e_p99,
                        error_rate_val):
    """Create a metrics dictionary with the given values."""
    return {
        "timestamp": "2024-12-04T12:00:00Z",
        "git_sha": "abc123",
        "workflow_run_id": "123456789",
        "metrics": {
            "ttfa_ms": {
                "mean": ttfa_mean,
                "p50": ttfa_p50,
                "p95": ttfa_p95,
                "p99": ttfa_p99,
            },
            "e2e_latency_ms": {
                "mean": e2e_mean,
                "p50": e2e_p50,
                "p95": e2e_p95,
                "p99": e2e_p99,
            },
            "throughput_rps": 12.5,
            "error_rate": error_rate_val,
            "total_requests": 1000,
        }
    }


@given(
    baseline_value=metric_value,
    increase_pct=st.floats(min_value=15.1, max_value=100.0),
)
def test_latency_increase_above_threshold_detected_as_regression(baseline_value, increase_pct):
    """
    Property: For any baseline latency metric and any increase percentage above the threshold,
    the regression detection should identify it as a regression.
    
    This tests that when latency increases by more than the threshold (10% for TTFA, 15% for e2e),
    the system correctly identifies it as a regression.
    
    Note: "Exceeds" means strictly greater than (>), so we test with values > 15% for e2e.
    """
    # Create baseline metrics
    baseline = create_metrics_dict(
        ttfa_mean=baseline_value, ttfa_p50=baseline_value, ttfa_p95=baseline_value, ttfa_p99=baseline_value,
        e2e_mean=baseline_value, e2e_p50=baseline_value, e2e_p95=baseline_value, e2e_p99=baseline_value,
        error_rate_val=0.001
    )
    
    # Create current metrics with increased values
    current_value = baseline_value * (1 + increase_pct / 100)
    current = create_metrics_dict(
        ttfa_mean=current_value, ttfa_p50=current_value, ttfa_p95=current_value, ttfa_p99=current_value,
        e2e_mean=current_value, e2e_p50=current_value, e2e_p95=current_value, e2e_p99=current_value,
        error_rate_val=0.001
    )
    
    # Check for regressions
    result = check_regressions(current, baseline)
    
    # Should detect regressions for all fields that exceed their thresholds
    assert result.has_regressions(), (
        f"Failed to detect regression: baseline={baseline_value:.2f}, "
        f"current={current_value:.2f}, increase={increase_pct:.1f}%"
    )
    
    # Verify that regressions were detected for the appropriate metrics
    regression_metrics = {(r["metric"], r["field"]) for r in result.regressions}
    
    # TTFA threshold is 10%, so anything >= 15% should be detected
    assert ("ttfa_ms", "mean") in regression_metrics
    assert ("ttfa_ms", "p50") in regression_metrics
    assert ("ttfa_ms", "p95") in regression_metrics
    assert ("ttfa_ms", "p99") in regression_metrics
    
    # E2E threshold is 15%, so anything >= 15% should be detected
    assert ("e2e_latency_ms", "mean") in regression_metrics
    assert ("e2e_latency_ms", "p50") in regression_metrics
    assert ("e2e_latency_ms", "p95") in regression_metrics
    assert ("e2e_latency_ms", "p99") in regression_metrics


@given(
    baseline_value=metric_value,
    increase_pct=st.floats(min_value=0.0, max_value=8.0),
)
def test_latency_increase_below_threshold_not_detected_as_regression(baseline_value, increase_pct):
    """
    Property: For any baseline latency metric and any increase percentage below the threshold,
    the regression detection should NOT identify it as a regression.
    
    This tests that when latency increases by less than the threshold, the system correctly
    passes the check.
    """
    # Create baseline metrics
    baseline = create_metrics_dict(
        ttfa_mean=baseline_value, ttfa_p50=baseline_value, ttfa_p95=baseline_value, ttfa_p99=baseline_value,
        e2e_mean=baseline_value, e2e_p50=baseline_value, e2e_p95=baseline_value, e2e_p99=baseline_value,
        error_rate_val=0.001
    )
    
    # Create current metrics with slightly increased values (below threshold)
    current_value = baseline_value * (1 + increase_pct / 100)
    current = create_metrics_dict(
        ttfa_mean=current_value, ttfa_p50=current_value, ttfa_p95=current_value, ttfa_p99=current_value,
        e2e_mean=current_value, e2e_p50=current_value, e2e_p95=current_value, e2e_p99=current_value,
        error_rate_val=0.001
    )
    
    # Check for regressions
    result = check_regressions(current, baseline)
    
    # Should NOT detect regressions since increase is below threshold
    assert not result.has_regressions(), (
        f"False positive regression: baseline={baseline_value:.2f}, "
        f"current={current_value:.2f}, increase={increase_pct:.1f}%"
    )


@given(
    baseline_error_rate=st.floats(min_value=0.0, max_value=0.005),
    current_error_rate=st.floats(min_value=0.015, max_value=0.5),
)
def test_error_rate_above_threshold_detected_as_regression(baseline_error_rate, current_error_rate):
    """
    Property: For any error rate above the absolute threshold (1% = 0.01),
    the regression detection should identify it as a regression.
    
    This tests that when error rate exceeds 1%, the system correctly identifies it as a regression.
    """
    # Create baseline metrics with low error rate
    baseline = create_metrics_dict(
        ttfa_mean=500.0, ttfa_p50=500.0, ttfa_p95=500.0, ttfa_p99=500.0,
        e2e_mean=1200.0, e2e_p50=1200.0, e2e_p95=1200.0, e2e_p99=1200.0,
        error_rate_val=baseline_error_rate
    )
    
    # Create current metrics with high error rate (above 1% threshold)
    current = create_metrics_dict(
        ttfa_mean=500.0, ttfa_p50=500.0, ttfa_p95=500.0, ttfa_p99=500.0,
        e2e_mean=1200.0, e2e_p50=1200.0, e2e_p95=1200.0, e2e_p99=1200.0,
        error_rate_val=current_error_rate
    )
    
    # Check for regressions
    result = check_regressions(current, baseline)
    
    # Should detect regression for error rate
    assert result.has_regressions(), (
        f"Failed to detect error rate regression: baseline={baseline_error_rate:.4f}, "
        f"current={current_error_rate:.4f}, threshold=0.01"
    )
    
    # Verify that error rate regression was detected
    regression_metrics = {(r["metric"], r["field"]) for r in result.regressions}
    assert ("error_rate", "value") in regression_metrics


@given(
    baseline_error_rate=st.floats(min_value=0.0, max_value=0.005),
    current_error_rate=st.floats(min_value=0.0, max_value=0.008),
)
def test_error_rate_below_threshold_not_detected_as_regression(baseline_error_rate, current_error_rate):
    """
    Property: For any error rate below the absolute threshold (1% = 0.01),
    the regression detection should NOT identify it as a regression.
    
    This tests that when error rate is below 1%, the system correctly passes the check.
    """
    # Create baseline metrics with low error rate
    baseline = create_metrics_dict(
        ttfa_mean=500.0, ttfa_p50=500.0, ttfa_p95=500.0, ttfa_p99=500.0,
        e2e_mean=1200.0, e2e_p50=1200.0, e2e_p95=1200.0, e2e_p99=1200.0,
        error_rate_val=baseline_error_rate
    )
    
    # Create current metrics with low error rate (below 1% threshold)
    current = create_metrics_dict(
        ttfa_mean=500.0, ttfa_p50=500.0, ttfa_p95=500.0, ttfa_p99=500.0,
        e2e_mean=1200.0, e2e_p50=1200.0, e2e_p95=1200.0, e2e_p99=1200.0,
        error_rate_val=current_error_rate
    )
    
    # Check for regressions
    result = check_regressions(current, baseline)
    
    # Should NOT detect regressions since error rate is below threshold
    assert not result.has_regressions(), (
        f"False positive error rate regression: baseline={baseline_error_rate:.4f}, "
        f"current={current_error_rate:.4f}, threshold=0.01"
    )


@given(
    ttfa_baseline=metric_value,
    ttfa_increase=st.floats(min_value=12.0, max_value=50.0),
    e2e_baseline=metric_value,
    e2e_increase=st.floats(min_value=5.0, max_value=12.0),
)
def test_mixed_regressions_correctly_identified(ttfa_baseline, ttfa_increase, e2e_baseline, e2e_increase):
    """
    Property: When some metrics exceed thresholds and others don't,
    the regression detection should correctly identify only the violating metrics.
    
    This tests that the system can handle mixed scenarios where some metrics regress
    and others don't.
    """
    # Create baseline metrics
    baseline = create_metrics_dict(
        ttfa_mean=ttfa_baseline, ttfa_p50=ttfa_baseline, ttfa_p95=ttfa_baseline, ttfa_p99=ttfa_baseline,
        e2e_mean=e2e_baseline, e2e_p50=e2e_baseline, e2e_p95=e2e_baseline, e2e_p99=e2e_baseline,
        error_rate_val=0.001
    )
    
    # TTFA increases above threshold (10%), e2e increases below threshold (15%)
    ttfa_current = ttfa_baseline * (1 + ttfa_increase / 100)
    e2e_current = e2e_baseline * (1 + e2e_increase / 100)
    
    current = create_metrics_dict(
        ttfa_mean=ttfa_current, ttfa_p50=ttfa_current, ttfa_p95=ttfa_current, ttfa_p99=ttfa_current,
        e2e_mean=e2e_current, e2e_p50=e2e_current, e2e_p95=e2e_current, e2e_p99=e2e_current,
        error_rate_val=0.001
    )
    
    # Check for regressions
    result = check_regressions(current, baseline)
    
    # Should detect regressions for TTFA but not e2e
    assert result.has_regressions(), "Failed to detect TTFA regressions"
    
    regression_metrics = {(r["metric"], r["field"]) for r in result.regressions}
    
    # TTFA should be flagged (increase > 10%)
    assert ("ttfa_ms", "mean") in regression_metrics
    assert ("ttfa_ms", "p50") in regression_metrics
    assert ("ttfa_ms", "p95") in regression_metrics
    assert ("ttfa_ms", "p99") in regression_metrics
    
    # E2E should NOT be flagged (increase < 15%)
    assert ("e2e_latency_ms", "mean") not in regression_metrics
    assert ("e2e_latency_ms", "p50") not in regression_metrics
    assert ("e2e_latency_ms", "p95") not in regression_metrics
    assert ("e2e_latency_ms", "p99") not in regression_metrics
