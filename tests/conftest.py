import sys
from unittest.mock import MagicMock


# Mock prometheus_client globally to avoid Duplicated timeseries errors
# and side effects from metrics collection during tests.
# We use a class to strictly define attributes and avoid MagicMock auto-creation
# for constants like CONTENT_TYPE_LATEST.
class MockPrometheus:
    CONTENT_TYPE_LATEST = "text/plain; charset=utf-8"

    @staticmethod
    def generate_latest():
        return b"# HELP test_metric Test metric\n# TYPE test_metric gauge\ntest_metric 1.0\n"

    @staticmethod
    def Counter(*_, **__):
        return MagicMock()

    @staticmethod
    def Gauge(*_, **__):
        return MagicMock()

    @staticmethod
    def Histogram(*_, **__):
        return MagicMock()

    @staticmethod
    def Info(*_, **__):
        return MagicMock()

    @staticmethod
    def Summary(*_, **__):
        return MagicMock()

    @staticmethod
    def Enum(*_, **__):
        return MagicMock()


# Replace the module in sys.modules
sys.modules["prometheus_client"] = MockPrometheus()  # type: ignore
