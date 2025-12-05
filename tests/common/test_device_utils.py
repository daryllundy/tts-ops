from unittest.mock import patch

import pytest

from common.device_utils import detect_best_device, is_mps_available, resolve_device


class TestDeviceUtils:

    @patch("torch.cuda.is_available")
    @patch("common.device_utils.is_mps_available")
    def test_detect_best_device_priority(self, mock_mps, mock_cuda):
        """
        Property: Automatic device selection priority (CUDA > MPS > CPU)
        """
        # Case 1: CUDA available -> CUDA
        mock_cuda.return_value = True
        mock_mps.return_value = True # Even if MPS is available
        assert detect_best_device() == "cuda:0"

        # Case 2: CUDA unavailable, MPS available -> MPS
        mock_cuda.return_value = False
        mock_mps.return_value = True
        assert detect_best_device() == "mps"

        # Case 3: Both unavailable -> CPU
        mock_cuda.return_value = False
        mock_mps.return_value = False
        assert detect_best_device() == "cpu"

    @patch("torch.backends.mps.is_available")
    def test_is_mps_available(self, mock_torch_mps):
        """
        Property: MPS availability check
        """
        # Case 1: Available
        mock_torch_mps.return_value = True
        assert is_mps_available() is True

        # Case 2: Unavailable
        mock_torch_mps.return_value = False
        assert is_mps_available() is False

        # Case 3: torch.backends.mps doesn't exist (simulating non-Mac or old PyTorch)
        # We need to be careful with mocking attributes that might not exist
        # The safest way is to mock the module where it's used if possible, or use a context manager
        # that handles non-existence.
        # Here we just rely on the fact that we are mocking the function call in the implementation
        # if the attribute exists.
        # But to test the `hasattr` check, we need to mock `torch.backends` to NOT have `mps`.

        with patch("common.device_utils.torch.backends", spec=[]):
            # spec=[] means it has no attributes, so hasattr(..., "mps") will be False
            assert is_mps_available() is False

    @patch("common.device_utils.detect_best_device")
    def test_resolve_device_auto(self, mock_detect):
        """
        Property: 'auto' resolves to best detected device
        """
        mock_detect.return_value = "cuda:0"
        assert resolve_device("auto") == "cuda:0"

        mock_detect.return_value = "mps"
        assert resolve_device("auto") == "mps"

        mock_detect.return_value = "cpu"
        assert resolve_device("auto") == "cpu"

    @patch("torch.cuda.is_available")
    def test_resolve_device_cuda(self, mock_cuda):
        """
        Property: Explicit CUDA request validation
        """
        # Available
        mock_cuda.return_value = True
        assert resolve_device("cuda:0") == "cuda:0"

        # Unavailable
        mock_cuda.return_value = False
        with pytest.raises(ValueError, match="Device 'cuda:0' requested but CUDA is not available"):
            resolve_device("cuda:0")

    @patch("common.device_utils.is_mps_available")
    def test_resolve_device_mps(self, mock_mps):
        """
        Property: Explicit MPS request validation
        """
        # Available
        mock_mps.return_value = True
        assert resolve_device("mps") == "mps"

        # Unavailable
        mock_mps.return_value = False
        with pytest.raises(ValueError, match="Device 'mps' requested but MPS is not available"):
            resolve_device("mps")

    def test_resolve_device_cpu(self):
        """
        Property: Explicit CPU request always valid
        """
        assert resolve_device("cpu") == "cpu"

    def test_resolve_device_invalid(self):
        """
        Property: Invalid device strings raise ValueError
        """
        with pytest.raises(ValueError, match="Invalid device string"):
            resolve_device("invalid_device")
