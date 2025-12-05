"""Unit tests for TTS service."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
import torch

from tts_service.server import app
from tts_service.model_loader import TTSModelManager
from tts_service.streaming import tensor_to_pcm_bytes, tensor_to_wav_bytes


class TestTTSEndpoints:
    """Test TTS API endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client with mocked model."""
        with patch("tts_service.server.get_model_manager") as mock_manager:
            manager = Mock()
            manager.is_loaded = True
            # Create a proper mock with attributes as values, not as Mock objects
            info_mock = Mock()
            info_mock.name = "test-model"
            info_mock.device = "cpu"
            info_mock.dtype = "float32"
            info_mock.sample_rate = 24000
            info_mock.warmup_completed = True
            manager.info = info_mock
            mock_manager.return_value = manager
            
            # Mock synthesize to return a tensor
            manager.synthesize.return_value = torch.randn(24000)
            
            yield TestClient(app, raise_server_exceptions=False)

    def test_health_check_healthy(self, client):
        """Test health endpoint when model is loaded."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True

    def test_health_check_unhealthy(self):
        """Test health endpoint when model is not loaded."""
        with patch("tts_service.server.get_model_manager") as mock_manager:
            manager = Mock()
            manager.is_loaded = False
            manager.info = None
            mock_manager.return_value = manager
            
            with TestClient(app, raise_server_exceptions=False) as client:
                response = client.get("/health")
                assert response.status_code == 200
                data = response.json()
                assert data["status"] == "unhealthy"

    def test_synthesize_success(self, client):
        """Test successful synthesis."""
        response = client.post(
            "/synthesize",
            json={"text": "Hello world", "output_format": "wav"},
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/wav"
        assert "X-Processing-Time-Ms" in response.headers

    def test_synthesize_empty_text(self, client):
        """Test synthesis with empty text."""
        response = client.post(
            "/synthesize",
            json={"text": "", "output_format": "wav"},
        )
        assert response.status_code == 422  # Validation error

    def test_synthesize_invalid_format(self, client):
        """Test synthesis with invalid output format."""
        response = client.post(
            "/synthesize",
            json={"text": "Hello", "output_format": "mp3"},
        )
        assert response.status_code == 422


class TestStreamingUtilities:
    """Test streaming utility functions."""

    def test_tensor_to_pcm_bytes(self):
        """Test PCM byte conversion."""
        audio = torch.randn(1000)
        pcm_bytes = tensor_to_pcm_bytes(audio)
        
        # PCM16 = 2 bytes per sample
        assert len(pcm_bytes) == 2000

    def test_tensor_to_wav_bytes(self):
        """Test WAV byte conversion."""
        audio = torch.randn(24000)  # 1 second at 24kHz
        wav_bytes = tensor_to_wav_bytes(audio, sample_rate=24000)
        
        # WAV header is 44 bytes, plus audio data
        assert len(wav_bytes) > 44
        assert wav_bytes[:4] == b"RIFF"

    def test_tensor_normalization(self):
        """Test that out-of-range tensors are normalized."""
        audio = torch.tensor([2.0, -2.0, 1.0, -1.0])
        pcm_bytes = tensor_to_pcm_bytes(audio)
        
        # Should not raise and should produce valid bytes
        assert len(pcm_bytes) == 8


class TestModelManager:
    """Test TTSModelManager."""

    def test_manager_initialization(self):
        """Test manager initializes correctly."""
        with patch("tts_service.model_loader.get_tts_settings") as mock_settings:
            mock_settings.return_value = Mock(
                model_name="test-model",
                device="cpu",
                dtype="float32",
                max_batch_size=4,
                sample_rate=24000,
                warmup_on_start=False,
                cache_dir=None,
                max_text_length=4096,
            )
            
            manager = TTSModelManager()
            assert not manager.is_loaded
            assert manager.info is None

    def test_synthesize_without_load_raises(self):
        """Test that synthesize raises when model not loaded."""
        with patch("tts_service.model_loader.get_tts_settings") as mock_settings:
            mock_settings.return_value = Mock(
                model_name="test-model",
                device="cpu",
                dtype="float32",
                max_batch_size=4,
                sample_rate=24000,
                warmup_on_start=False,
                cache_dir=None,
                max_text_length=4096,
            )
            
            manager = TTSModelManager()
            
            with pytest.raises(RuntimeError, match="Model not loaded"):
                manager.synthesize("Hello")
