import sys
from unittest.mock import MagicMock

# Mock prometheus_client globally to avoid Duplicated timeseries errors
# and side effects from metrics collection during tests.
sys.modules["prometheus_client"] = MagicMock()

# Also mock src.common.metrics to avoid side effects if needed,
# but mocking prometheus_client should be enough to stop errors.
# However, if we want to test metrics collection, we need to be careful.
# test_metrics.py mocks prometheus_client too.
# If we mock it here, it persists.
