"""Unit tests for TTS service."""

from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
from fastapi.testclient import TestClient

from tts_service.model_loader import TTSModelManager
from tts_service.server import app
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

    def test_synthesize_pcm_format(self, client):
        """Test synthesis with PCM output format."""
        response = client.post(
            "/synthesize",
            json={"text": "Hello world", "output_format": "pcm"},
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/pcm"
        assert "X-Processing-Time-Ms" in response.headers
        assert "X-Sample-Rate" in response.headers

    def test_synthesize_with_voice_id(self, client):
        """Test synthesis with voice_id parameter."""
        response = client.post(
            "/synthesize",
            json={"text": "Hello world", "voice_id": "voice-1", "output_format": "wav"},
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/wav"

    def test_synthesize_streaming(self, client):
        """Test streaming synthesis."""
        response = client.post(
            "/synthesize?stream=true",
            json={"text": "Hello world", "output_format": "wav"},
        )
        assert response.status_code == 200
        assert "X-Sample-Rate" in response.headers
        assert "X-Channels" in response.headers

    def test_synthesize_model_not_loaded(self):
        """Test synthesis when model is not loaded."""
        with patch("tts_service.server.get_model_manager") as mock_manager:
            manager = Mock()
            manager.is_loaded = False
            mock_manager.return_value = manager

            with TestClient(app, raise_server_exceptions=False) as client:
                response = client.post(
                    "/synthesize",
                    json={"text": "Hello", "output_format": "wav"},
                )
                assert response.status_code == 503
                assert "Model not loaded" in response.json()["detail"]

    def test_synthesize_value_error(self, client):
        """Test synthesis with ValueError (e.g., text too long)."""
        with patch("tts_service.server.get_model_manager") as mock_manager:
            manager = Mock()
            manager.is_loaded = True
            manager.synthesize.side_effect = ValueError("Text too long")
            mock_manager.return_value = manager

            response = client.post(
                "/synthesize",
                json={"text": "Hello", "output_format": "wav"},
            )
            assert response.status_code == 400
            assert "Text too long" in response.json()["detail"]

    def test_synthesize_generic_error(self, client):
        """Test synthesis with generic exception."""
        with patch("tts_service.server.get_model_manager") as mock_manager:
            manager = Mock()
            manager.is_loaded = True
            manager.synthesize.side_effect = Exception("Unexpected error")
            mock_manager.return_value = manager

            response = client.post(
                "/synthesize",
                json={"text": "Hello", "output_format": "wav"},
            )
            assert response.status_code == 500
            assert "Synthesis failed" in response.json()["detail"]

    def test_ready_endpoint_success(self, client):
        """Test readiness endpoint when model is ready."""
        response = client.get("/ready")
        assert response.status_code == 200
        assert response.json() == {"status": "ready"}

    def test_ready_endpoint_model_not_loaded(self):
        """Test readiness endpoint when model not loaded."""
        with patch("tts_service.server.get_model_manager") as mock_manager:
            manager = Mock()
            manager.is_loaded = False
            manager.info = None
            mock_manager.return_value = manager

            with TestClient(app, raise_server_exceptions=False) as client:
                response = client.get("/ready")
                assert response.status_code == 503
                assert "Model not loaded" in response.json()["detail"]

    def test_ready_endpoint_warmup_not_completed(self):
        """Test readiness endpoint when warmup not completed."""
        with patch("tts_service.server.get_model_manager") as mock_manager:
            manager = Mock()
            manager.is_loaded = True
            manager.info = Mock(warmup_completed=False)
            mock_manager.return_value = manager

            with TestClient(app, raise_server_exceptions=False) as client:
                response = client.get("/ready")
                assert response.status_code == 503
                assert "warmup not completed" in response.json()["detail"]

    def test_metrics_endpoint(self, client):
        """Test Prometheus metrics endpoint."""
        response = client.get("/metrics")
        assert response.status_code == 200
        # Check for Prometheus content type
        assert "text/plain" in response.headers["content-type"]

    def test_info_endpoint_success(self, client):
        """Test model info endpoint."""
        response = client.get("/info")
        assert response.status_code == 200
        data = response.json()
        assert data["model_name"] == "test-model"
        assert data["device"] == "cpu"
        assert data["dtype"] == "float32"
        assert data["sample_rate"] == 24000
        assert data["warmup_completed"] is True

    def test_info_endpoint_model_not_loaded(self):
        """Test info endpoint when model not loaded."""
        with patch("tts_service.server.get_model_manager") as mock_manager:
            manager = Mock()
            manager.is_loaded = False
            manager.info = None
            mock_manager.return_value = manager

            with TestClient(app, raise_server_exceptions=False) as client:
                response = client.get("/info")
                assert response.status_code == 503
                assert "Model not loaded" in response.json()["detail"]


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

    def test_create_wav_header(self):
        """Test WAV header creation."""
        from tts_service.streaming import create_wav_header

        header = create_wav_header(sample_rate=24000, num_channels=1, bits_per_sample=16)

        # Check header length (44 bytes)
        assert len(header) == 44
        # Check RIFF signature
        assert header[:4] == b"RIFF"
        # Check WAVE signature
        assert header[8:12] == b"WAVE"
        # Check fmt signature
        assert header[12:16] == b"fmt "

    def test_tensor_to_pcm_multidim(self):
        """Test PCM conversion with multi-dimensional tensor."""
        # Create 2D tensor (should be squeezed to 1D)
        audio = torch.randn(1, 1000)
        pcm_bytes = tensor_to_pcm_bytes(audio)

        assert len(pcm_bytes) == 2000

    def test_tensor_to_wav_multidim(self):
        """Test WAV conversion with multi-dimensional tensor."""
        audio = torch.randn(1, 24000)
        wav_bytes = tensor_to_wav_bytes(audio, sample_rate=24000)

        assert len(wav_bytes) > 44
        assert wav_bytes[:4] == b"RIFF"

    @pytest.mark.asyncio
    async def test_stream_audio_chunks_with_header(self):
        """Test streaming audio chunks with WAV header."""
        from tts_service.streaming import stream_audio_chunks

        async def audio_gen():
            yield torch.randn(1000)
            yield torch.randn(1000)

        chunks = []
        async for chunk in stream_audio_chunks(audio_gen(), sample_rate=24000, include_header=True):
            chunks.append(chunk)

        # Should have header + 2 audio chunks
        assert len(chunks) == 3
        # First chunk should be WAV header
        assert chunks[0][:4] == b"RIFF"

    @pytest.mark.asyncio
    async def test_stream_audio_chunks_without_header(self):
        """Test streaming audio chunks without header."""
        from tts_service.streaming import stream_audio_chunks

        async def audio_gen():
            yield torch.randn(1000)
            yield torch.randn(1000)

        chunks = []
        async for chunk in stream_audio_chunks(audio_gen(), sample_rate=24000, include_header=False):
            chunks.append(chunk)

        # Should have only audio chunks
        assert len(chunks) == 2
        # First chunk should be PCM data, not WAV header
        assert chunks[0][:4] != b"RIFF"

    @pytest.mark.asyncio
    async def test_audio_buffer_basic(self):
        """Test AudioBuffer basic operations."""
        from tts_service.streaming import AudioBuffer

        buffer = AudioBuffer(sample_rate=24000)

        # Add chunks
        await buffer.add_chunk(torch.randn(1000))
        await buffer.add_chunk(torch.randn(1000))

        # Get complete audio
        audio = await buffer.get_complete_audio()
        assert audio.shape[0] == 2000

    @pytest.mark.asyncio
    async def test_audio_buffer_empty(self):
        """Test AudioBuffer with no chunks."""
        from tts_service.streaming import AudioBuffer

        buffer = AudioBuffer(sample_rate=24000)
        audio = await buffer.get_complete_audio()

        assert audio.shape[0] == 0

    @pytest.mark.asyncio
    async def test_audio_buffer_clear(self):
        """Test AudioBuffer clear operation."""
        from tts_service.streaming import AudioBuffer

        buffer = AudioBuffer(sample_rate=24000)
        await buffer.add_chunk(torch.randn(1000))
        await buffer.clear()

        audio = await buffer.get_complete_audio()
        assert audio.shape[0] == 0


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

    def test_load_model_success(self):
        """Test successful model loading."""
        # Patch transformers in sys.modules to handle local import
        mock_transformers = MagicMock()
        mock_processor_cls = mock_transformers.AutoProcessor
        mock_model_cls = mock_transformers.AutoModelForTextToWaveform

        with patch("tts_service.model_loader.get_tts_settings") as mock_settings, \
             patch.dict("sys.modules", {"transformers": mock_transformers}):

            mock_settings.return_value = Mock(
                model_name="test-model",
                device="cpu",
                dtype="float32",
                sample_rate=24000,
                warmup_on_start=False,
                cache_dir=None,
                max_text_length=4096,
            )

            # Mock the model and processor
            mock_processor_instance = Mock()
            mock_processor_cls.from_pretrained.return_value = mock_processor_instance

            mock_model_instance = Mock()
            mock_model_cls.from_pretrained.return_value = mock_model_instance

            manager = TTSModelManager()
            manager.load()

            assert manager.is_loaded
            assert manager.info is not None
            assert manager.info.name == "test-model"
            assert manager.info.device == "cpu"

    def test_load_model_with_warmup(self):
        """Test model loading with warmup enabled."""
        # Patch transformers in sys.modules
        mock_transformers = MagicMock()
        mock_processor_cls = mock_transformers.AutoProcessor
        mock_model_cls = mock_transformers.AutoModelForTextToWaveform

        with patch("tts_service.model_loader.get_tts_settings") as mock_settings, \
             patch.dict("sys.modules", {"transformers": mock_transformers}):

            mock_settings.return_value = Mock(
                model_name="test-model",
                device="cpu",
                dtype="float32",
                sample_rate=24000,
                warmup_on_start=True,
                cache_dir=None,
                max_text_length=4096,
            )

            mock_processor_instance = Mock()
            mock_processor_cls.from_pretrained.return_value = mock_processor_instance
            mock_processor_instance.return_value.to.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}

            mock_model_instance = Mock()
            mock_model_instance.generate.return_value = torch.randn(24000)
            mock_model_cls.from_pretrained.return_value = mock_model_instance

            manager = TTSModelManager()
            manager.load()

            assert manager.info.warmup_completed is True

    def test_load_model_already_loaded(self):
        """Test loading when model is already loaded."""
        with patch("tts_service.model_loader.get_tts_settings") as mock_settings:
            mock_settings.return_value = Mock(
                model_name="test-model",
                device="cpu",
                dtype="float32",
                sample_rate=24000,
                warmup_on_start=False,
                cache_dir=None,
                max_text_length=4096,
            )

            manager = TTSModelManager()
            manager._model = Mock()  # Simulate already loaded

            manager.load()  # Should not raise, just skip

    def test_unload_model(self):
        """Test model unloading."""
        with patch("tts_service.model_loader.get_tts_settings") as mock_settings:
            mock_settings.return_value = Mock(
                model_name="test-model",
                device="cpu",
                dtype="float32",
                sample_rate=24000,
                warmup_on_start=False,
                cache_dir=None,
                max_text_length=4096,
            )

            manager = TTSModelManager()
            manager._model = Mock()
            manager._processor = Mock()
            manager._info = Mock()

            manager.unload()

            assert manager._model is None
            assert manager._processor is None
            assert manager._info is None
            assert not manager.is_loaded

    def test_synthesize_text_too_long(self):
        """Test synthesis with text exceeding max length."""
        with patch("tts_service.model_loader.get_tts_settings") as mock_settings:
            mock_settings.return_value = Mock(
                model_name="test-model",
                device="cpu",
                dtype="float32",
                sample_rate=24000,
                warmup_on_start=False,
                cache_dir=None,
                max_text_length=100,
            )

            manager = TTSModelManager()
            manager._model = Mock()  # Simulate loaded model

            long_text = "a" * 101

            with pytest.raises(ValueError, match="exceeds maximum length"):
                manager.synthesize(long_text)

    def test_synthesize_streaming(self):
        """Test streaming synthesis."""
        with patch("tts_service.model_loader.get_tts_settings") as mock_settings:
            mock_settings.return_value = Mock(
                model_name="test-model",
                device="cpu",
                dtype="float32",
                sample_rate=24000,
                warmup_on_start=False,
                cache_dir=None,
                max_text_length=4096,
            )

            manager = TTSModelManager()
            manager._model = Mock()
            manager._processor = Mock()
            manager._processor.return_value.to.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
            manager._model.generate.return_value = torch.randn(10000)

            chunks = list(manager.synthesize_streaming("Hello", chunk_size=1000))

            assert len(chunks) == 10
            for chunk in chunks:
                assert len(chunk) <= 1000

    def test_inference_context_not_loaded(self):
        """Test inference context when model not loaded."""
        with patch("tts_service.model_loader.get_tts_settings") as mock_settings:
            mock_settings.return_value = Mock(
                model_name="test-model",
                device="cpu",
                dtype="float32",
                sample_rate=24000,
                warmup_on_start=False,
                cache_dir=None,
                max_text_length=4096,
            )

            manager = TTSModelManager()

            with pytest.raises(RuntimeError, match="Model not loaded"):
                with manager.inference_context():
                    pass
