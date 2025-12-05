import sys
from unittest.mock import MagicMock, patch
import pytest
import torch
from src.tts_service.model_loader import TTSModelManager, ModelInfo
from src.common.config import TTSServiceSettings

class TestTTSModelManager:
    
    @pytest.fixture
    def mock_settings(self):
        return TTSServiceSettings(
            model_name="test-model",
            device="auto",
            dtype="float16"
        )

    @pytest.fixture
    def manager(self, mock_settings):
        return TTSModelManager(settings=mock_settings)

    @patch("src.tts_service.model_loader.resolve_device")
    def test_load_device_resolution_and_logging(self, mock_resolve, manager):
        """
        Property: Device selection logging and resolution
        """
        mock_resolve.return_value = "mps"
        
        # Mock transformers module
        mock_transformers = MagicMock()
        mock_processor = MagicMock()
        mock_model = MagicMock()
        
        mock_transformers.AutoProcessor.from_pretrained.return_value = mock_processor
        mock_transformers.AutoModelForTextToWaveform.from_pretrained.return_value = mock_model
        
        with patch.dict(sys.modules, {"transformers": mock_transformers}):
            with patch("src.tts_service.model_loader.logger") as mock_logger:
                manager.load()
                
                # Verify resolve_device was called
                mock_resolve.assert_called_once()
                
                # Verify logging
                mock_logger.info.assert_any_call(
                    "Resolved device configuration",
                    configured_device="auto",
                    resolved_device="mps",
                    dtype=str(torch.float16) # Default for float16 on MPS if not changed by logic
                )
                
                # Verify model moved to resolved device
                mock_model.to.assert_called_with("mps")

    def test_get_optimal_dtype_mps(self, manager):
        """
        Property: MPS dtype compatibility
        """
        # Case 1: float16 requested on MPS -> float16 (with warning in logs, but returns float16)
        # Wait, my implementation returns float16 but logs debug.
        dtype = manager._get_optimal_dtype("mps")
        assert dtype == torch.float16
        
        # Case 2: float32 requested -> float32
        manager.settings.dtype = "float32"
        dtype = manager._get_optimal_dtype("mps")
        assert dtype == torch.float32

    def test_get_optimal_dtype_cuda(self, manager):
        """
        Property: CUDA dtype preservation
        """
        manager.settings.dtype = "float16"
        dtype = manager._get_optimal_dtype("cuda:0")
        assert dtype == torch.float16
        
        manager.settings.dtype = "bfloat16"
        dtype = manager._get_optimal_dtype("cuda:0")
        assert dtype == torch.bfloat16

    @patch("torch.cuda.empty_cache")
    @patch("torch.cuda.is_available")
    @patch("torch.mps.empty_cache")
    @patch("src.tts_service.model_loader.is_mps_available")
    def test_clear_device_cache(self, mock_is_mps, mock_mps_empty, mock_cuda_avail, mock_cuda_empty, manager):
        """
        Property: Backend-specific cache clearing
        """
        # Case 1: CUDA
        mock_cuda_avail.return_value = True
        mock_is_mps.return_value = False
        
        manager._clear_device_cache()
        mock_cuda_empty.assert_called_once()
        mock_mps_empty.assert_not_called()
            
        # Case 2: MPS
        mock_cuda_empty.reset_mock()
        mock_cuda_avail.return_value = False
        mock_is_mps.return_value = True
        
        manager._clear_device_cache()
        mock_mps_empty.assert_called_once()
        mock_cuda_empty.assert_not_called()

    @patch("torch.cuda.synchronize")
    @patch("torch.cuda.is_available")
    @patch("torch.mps.synchronize")
    @patch("src.tts_service.model_loader.is_mps_available")
    def test_synchronize_device(self, mock_is_mps, mock_mps_sync, mock_cuda_avail, mock_cuda_sync, manager):
        """
        Property: Backend-specific synchronization
        """
        # Case 1: CUDA
        manager.settings.device = "cuda:0"
        mock_cuda_avail.return_value = True
        mock_is_mps.return_value = False
        
        manager._synchronize_device()
        mock_cuda_sync.assert_called_once()
        mock_mps_sync.assert_not_called()
        
        # Case 2: MPS
        mock_cuda_sync.reset_mock()
        manager.settings.device = "mps"
        mock_is_mps.return_value = True
        
        manager._synchronize_device()
        mock_mps_sync.assert_called_once()
        mock_cuda_sync.assert_not_called()

    @patch("src.tts_service.model_loader.TTSModelManager.synthesize")
    def test_warmup_success(self, mock_synthesize, manager):
        """
        Property: Warmup cross-platform success
        """
        manager._info = MagicMock()
        manager._warmup()
        
        mock_synthesize.assert_called_once()
        assert manager._info.warmup_completed is True
