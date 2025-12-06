import sys
from unittest.mock import MagicMock

# Mock prometheus_client globally to avoid Duplicated timeseries errors
# and side effects from metrics collection during tests.
mock_prometheus = MagicMock()
mock_prometheus.CONTENT_TYPE_LATEST = "text/plain; charset=utf-8"
mock_prometheus.generate_latest.return_value = b"# HELP test_metric Test metric\n# TYPE test_metric gauge\ntest_metric 1.0\n"
sys.modules["prometheus_client"] = mock_prometheus

# Also mock src.common.metrics to avoid side effects if needed,
# but mocking prometheus_client should be enough to stop errors.
# However, if we want to test metrics collection, we need to be careful.
# test_metrics.py mocks prometheus_client too.
# If we mock it here, it persists.
