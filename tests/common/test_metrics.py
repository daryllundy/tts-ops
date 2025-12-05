import sys
from unittest.mock import MagicMock, patch

# Mock prometheus_client to avoid registry errors
sys.modules["prometheus_client"] = MagicMock()

from common.metrics import record_gpu_metrics


class TestMetrics:

    @patch("common.metrics.is_mps_available")
    @patch("common.metrics.TTS_GPU_MEMORY_USED")
    def test_record_gpu_metrics_mps(self, mock_gauge, mock_is_mps):
        """
        Property: MPS metrics collection
        """
        mock_is_mps.return_value = True

        # Mock torch
        mock_torch = MagicMock()
        mock_torch.mps.current_allocated_memory.return_value = 1024

        with patch.dict(sys.modules, {"torch": mock_torch}):
            record_gpu_metrics(device="mps")

            mock_gauge.labels.assert_called_with(device="mps")
            mock_gauge.labels.return_value.set.assert_called_with(1024)

    @patch("common.metrics.is_mps_available")
    @patch("common.metrics.TTS_GPU_MEMORY_USED")
    def test_record_gpu_metrics_cuda(self, mock_gauge, mock_is_mps):
        """
        Property: CUDA metrics collection
        """
        mock_is_mps.return_value = False

        # Mock torch
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_allocated.return_value = 2048

        with patch.dict(sys.modules, {"torch": mock_torch}):
            record_gpu_metrics(device="cuda:0")

            mock_gauge.labels.assert_called_with(device="cuda:0")
            mock_gauge.labels.return_value.set.assert_called_with(2048)

    @patch("common.metrics.is_mps_available")
    @patch("common.metrics.TTS_GPU_MEMORY_USED")
    def test_record_gpu_metrics_mps_failure(self, mock_gauge, mock_is_mps):
        """
        Property: MPS metrics collection failure handling
        """
        mock_is_mps.return_value = True

        # Mock torch to raise exception
        mock_torch = MagicMock()
        mock_torch.mps.current_allocated_memory.side_effect = RuntimeError("MPS error")

        with patch.dict(sys.modules, {"torch": mock_torch}):
            # Should not raise exception
            record_gpu_metrics(device="mps")

            # Should not have set metrics
            mock_gauge.labels.assert_not_called()
