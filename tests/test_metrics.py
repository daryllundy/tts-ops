"""Unit tests for metrics and logging."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import logging

from common.metrics import (
    TTS_REQUEST_DURATION,
    TTS_REQUESTS_TOTAL,
    TTS_INPUT_LENGTH,
    TTS_OUTPUT_DURATION,
    TTS_GPU_UTILIZATION,
    TTS_GPU_MEMORY_USED,
    TTS_MODEL_LOADED,
    TTS_MODEL_INFO,
    AGENT_CHAT_DURATION,
    AGENT_LLM_DURATION,
    AGENT_REQUESTS_TOTAL,
    AGENT_ACTIVE_CONNECTIONS,
    record_gpu_metrics,
)
from common.logging import setup_logging, get_logger, add_service_context


class TestPrometheusMetrics:
    """Test Prometheus metrics definitions."""

    def test_tts_request_duration_metric(self):
        """Test TTS request duration histogram."""
        TTS_REQUEST_DURATION.labels(endpoint="synthesize", status="success").observe(0.5)

        # Should not raise
        assert TTS_REQUEST_DURATION is not None

    def test_tts_requests_total_counter(self):
        """Test TTS requests total counter."""
        TTS_REQUESTS_TOTAL.labels(endpoint="synthesize", status="success").inc()

        # Should not raise
        assert TTS_REQUESTS_TOTAL is not None

    def test_tts_input_length_histogram(self):
        """Test TTS input length histogram."""
        TTS_INPUT_LENGTH.observe(100)

        assert TTS_INPUT_LENGTH is not None

    def test_tts_output_duration_histogram(self):
        """Test TTS output duration histogram."""
        TTS_OUTPUT_DURATION.observe(5.0)

        assert TTS_OUTPUT_DURATION is not None

    def test_tts_gpu_metrics(self):
        """Test GPU metrics gauges."""
        TTS_GPU_UTILIZATION.labels(device="cuda:0").set(75.0)
        TTS_GPU_MEMORY_USED.labels(device="cuda:0").set(1024 * 1024 * 1024)

        assert TTS_GPU_UTILIZATION is not None
        assert TTS_GPU_MEMORY_USED is not None

    def test_tts_model_loaded_gauge(self):
        """Test model loaded gauge."""
        TTS_MODEL_LOADED.set(1)
        TTS_MODEL_LOADED.set(0)

        assert TTS_MODEL_LOADED is not None

    def test_tts_model_info(self):
        """Test model info metric."""
        TTS_MODEL_INFO.info({
            "model_name": "test-model",
            "device": "cuda:0",
            "dtype": "float16",
        })

        assert TTS_MODEL_INFO is not None

    def test_agent_chat_duration(self):
        """Test agent chat duration histogram."""
        AGENT_CHAT_DURATION.labels(status="success").observe(2.5)

        assert AGENT_CHAT_DURATION is not None

    def test_agent_llm_duration(self):
        """Test agent LLM duration histogram."""
        AGENT_LLM_DURATION.labels(provider="anthropic", model="claude-3").observe(1.2)

        assert AGENT_LLM_DURATION is not None

    def test_agent_requests_total(self):
        """Test agent requests counter."""
        AGENT_REQUESTS_TOTAL.labels(endpoint="chat", status="success").inc()

        assert AGENT_REQUESTS_TOTAL is not None

    def test_agent_active_connections(self):
        """Test active connections gauge."""
        AGENT_ACTIVE_CONNECTIONS.inc()
        AGENT_ACTIVE_CONNECTIONS.dec()
        AGENT_ACTIVE_CONNECTIONS.set(5)

        assert AGENT_ACTIVE_CONNECTIONS is not None

    def test_metric_labels(self):
        """Test metrics with different label combinations."""
        # TTS metrics
        TTS_REQUEST_DURATION.labels(endpoint="synthesize", status="success").observe(0.1)
        TTS_REQUEST_DURATION.labels(endpoint="synthesize", status="error").observe(0.2)

        TTS_REQUESTS_TOTAL.labels(endpoint="health", status="success").inc()

        # Agent metrics
        AGENT_LLM_DURATION.labels(provider="openai", model="gpt-4").observe(1.0)
        AGENT_REQUESTS_TOTAL.labels(endpoint="synthesize", status="error").inc()


class TestGPUMetrics:
    """Test GPU metrics recording."""

    @patch("common.metrics.torch")
    def test_record_gpu_metrics_available(self, mock_torch):
        """Test recording GPU metrics when CUDA is available."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_allocated.return_value = 2 * 1024 * 1024 * 1024  # 2GB

        record_gpu_metrics("cuda:0")

        mock_torch.cuda.is_available.assert_called_once()
        mock_torch.cuda.memory_allocated.assert_called_once_with(0)

    @patch("common.metrics.torch")
    def test_record_gpu_metrics_with_pynvml(self, mock_torch):
        """Test recording GPU metrics with pynvml."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_allocated.return_value = 1024

        with patch("common.metrics.pynvml") as mock_pynvml:
            mock_handle = Mock()
            mock_util = Mock()
            mock_util.gpu = 85.0
            mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle
            mock_pynvml.nvmlDeviceGetUtilizationRates.return_value = mock_util

            record_gpu_metrics("cuda:0")

            mock_pynvml.nvmlInit.assert_called_once()
            mock_pynvml.nvmlDeviceGetHandleByIndex.assert_called_once_with(0)

    @patch("common.metrics.torch")
    def test_record_gpu_metrics_cuda_unavailable(self, mock_torch):
        """Test GPU metrics when CUDA is not available."""
        mock_torch.cuda.is_available.return_value = False

        # Should not raise
        record_gpu_metrics("cpu")

    @patch("common.metrics.torch")
    def test_record_gpu_metrics_different_devices(self, mock_torch):
        """Test GPU metrics with different device indices."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_allocated.return_value = 1024

        record_gpu_metrics("cuda:0")
        mock_torch.cuda.memory_allocated.assert_called_with(0)

        record_gpu_metrics("cuda:1")
        mock_torch.cuda.memory_allocated.assert_called_with(1)

    @patch("common.metrics.torch")
    def test_record_gpu_metrics_error_handling(self, mock_torch):
        """Test GPU metrics error handling."""
        mock_torch.cuda.is_available.side_effect = Exception("CUDA error")

        # Should not raise, just pass silently
        record_gpu_metrics("cuda:0")

    @patch("common.metrics.torch")
    def test_record_gpu_metrics_pynvml_missing(self, mock_torch):
        """Test GPU metrics when pynvml is not available."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_allocated.return_value = 1024

        with patch("common.metrics.pynvml", None):
            # Should not raise even without pynvml
            record_gpu_metrics("cuda:0")


class TestLogging:
    """Test logging configuration."""

    @patch("common.logging.get_observability_settings")
    @patch("common.logging.structlog")
    def test_setup_logging_json_format(self, mock_structlog, mock_settings):
        """Test logging setup with JSON format."""
        mock_settings.return_value = Mock(
            log_level="INFO",
            log_format="json",
            service_name="test-service",
        )

        setup_logging()

        # Should configure structlog
        mock_structlog.configure.assert_called_once()

    @patch("common.logging.get_observability_settings")
    @patch("common.logging.structlog")
    def test_setup_logging_console_format(self, mock_structlog, mock_settings):
        """Test logging setup with console format."""
        mock_settings.return_value = Mock(
            log_level="DEBUG",
            log_format="console",
            service_name="test-service",
        )

        setup_logging()

        mock_structlog.configure.assert_called_once()

    @patch("common.logging.get_observability_settings")
    def test_setup_logging_log_levels(self, mock_settings):
        """Test different log levels."""
        for level in ["DEBUG", "INFO", "WARNING", "ERROR"]:
            mock_settings.return_value = Mock(
                log_level=level,
                log_format="json",
                service_name="test-service",
            )

            setup_logging()

            # Should configure root logger with correct level
            root_logger = logging.getLogger()
            assert root_logger.level == getattr(logging, level)

    def test_get_logger(self):
        """Test getting a structured logger."""
        logger = get_logger("test_module")

        assert logger is not None
        # Should be a structlog logger
        assert hasattr(logger, "info")
        assert hasattr(logger, "error")
        assert hasattr(logger, "warning")
        assert hasattr(logger, "debug")

    @patch("common.logging.get_observability_settings")
    def test_add_service_context(self, mock_settings):
        """Test adding service context to logs."""
        mock_settings.return_value = Mock(
            service_name="test-service",
        )

        event_dict = {"message": "test"}
        result = add_service_context(Mock(), "info", event_dict)

        assert "service" in result
        assert result["service"] == "test-service"

    def test_logger_methods(self):
        """Test logger methods work correctly."""
        logger = get_logger("test")

        # Should not raise
        logger.info("Test info message", key="value")
        logger.error("Test error", error="something")
        logger.warning("Test warning")
        logger.debug("Test debug", data={"nested": "value"})

    @patch("common.logging.get_observability_settings")
    @patch("common.logging.structlog")
    def test_logging_processors(self, mock_structlog, mock_settings):
        """Test logging processors are configured."""
        mock_settings.return_value = Mock(
            log_level="INFO",
            log_format="json",
            service_name="test-service",
        )

        setup_logging()

        # Should configure with processors
        call_args = mock_structlog.configure.call_args
        assert "processors" in call_args.kwargs

    @patch("common.logging.get_observability_settings")
    def test_logging_suppresses_noisy_loggers(self, mock_settings):
        """Test that noisy loggers are suppressed."""
        mock_settings.return_value = Mock(
            log_level="INFO",
            log_format="json",
            service_name="test-service",
        )

        setup_logging()

        # Check that noisy loggers are set to WARNING
        uvicorn_logger = logging.getLogger("uvicorn.access")
        httpx_logger = logging.getLogger("httpx")
        httpcore_logger = logging.getLogger("httpcore")

        assert uvicorn_logger.level == logging.WARNING
        assert httpx_logger.level == logging.WARNING
        assert httpcore_logger.level == logging.WARNING


class TestMetricsIntegration:
    """Test metrics integration scenarios."""

    def test_tts_request_lifecycle(self):
        """Test complete TTS request metrics lifecycle."""
        # Simulate a successful request
        TTS_INPUT_LENGTH.observe(150)
        TTS_REQUEST_DURATION.labels(endpoint="synthesize", status="success").observe(0.75)
        TTS_OUTPUT_DURATION.observe(3.5)
        TTS_REQUESTS_TOTAL.labels(endpoint="synthesize", status="success").inc()

        # Should not raise

    def test_tts_error_lifecycle(self):
        """Test TTS error metrics lifecycle."""
        TTS_REQUESTS_TOTAL.labels(endpoint="synthesize", status="error").inc()
        TTS_REQUEST_DURATION.labels(endpoint="synthesize", status="error").observe(0.1)

    def test_agent_request_lifecycle(self):
        """Test complete agent request metrics lifecycle."""
        # Simulate chat request
        AGENT_REQUESTS_TOTAL.labels(endpoint="chat", status="success").inc()
        AGENT_LLM_DURATION.labels(provider="anthropic", model="claude-3").observe(1.5)
        AGENT_CHAT_DURATION.labels(status="success").observe(2.0)

    def test_websocket_lifecycle(self):
        """Test WebSocket connection metrics."""
        # Connection opened
        AGENT_ACTIVE_CONNECTIONS.inc()
        assert AGENT_ACTIVE_CONNECTIONS._value._value >= 0

        # Connection closed
        AGENT_ACTIVE_CONNECTIONS.dec()

    def test_model_loading_metrics(self):
        """Test model loading metrics."""
        # Model loading
        TTS_MODEL_LOADED.set(0)

        # Model loaded successfully
        TTS_MODEL_LOADED.set(1)
        TTS_MODEL_INFO.info({
            "model_name": "test-model",
            "device": "cuda:0",
            "dtype": "float16",
        })

        # Model unloaded
        TTS_MODEL_LOADED.set(0)
