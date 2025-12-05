"""Integration tests for TTS and Agent services."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, Mock, patch
import torch

from tts_service.server import app as tts_app
from agent_app.api import app as agent_app


class TestTTSIntegration:
    """Integration tests for TTS service."""

    @pytest.fixture
    def tts_client(self):
        """Create TTS test client with mocked model."""
        with patch("tts_service.server.get_model_manager") as mock_manager:
            manager = Mock()
            manager.is_loaded = True
            # Create info mock with attributes set explicitly (not via Mock constructor)
            # Note: Mock(name=...) sets the mock's internal name, not an attribute
            info_mock = Mock()
            info_mock.name = "test-model"
            info_mock.device = "cpu"
            info_mock.dtype = "float32"
            info_mock.sample_rate = 24000
            info_mock.warmup_completed = True
            manager.info = info_mock
            manager.synthesize.return_value = torch.randn(24000)

            def mock_streaming(text, chunk_size=4096):
                audio = torch.randn(24000)
                for i in range(0, len(audio), chunk_size):
                    yield audio[i : i + chunk_size]

            manager.synthesize_streaming = mock_streaming
            mock_manager.return_value = manager

            yield TestClient(tts_app, raise_server_exceptions=False)

    def test_health_to_synthesis_flow(self, tts_client):
        """Test complete flow from health check to synthesis."""
        # Check health
        health_response = tts_client.get("/health")
        assert health_response.status_code == 200
        assert health_response.json()["model_loaded"] is True

        # Check readiness
        ready_response = tts_client.get("/ready")
        assert ready_response.status_code == 200

        # Synthesize
        synth_response = tts_client.post(
            "/synthesize",
            json={"text": "Hello world", "output_format": "wav"},
        )
        assert synth_response.status_code == 200
        assert synth_response.headers["content-type"] == "audio/wav"

    def test_info_endpoint_integration(self, tts_client):
        """Test model info endpoint."""
        # Get model info
        info_response = tts_client.get("/info")
        assert info_response.status_code == 200

        info = info_response.json()
        assert info["model_name"] == "test-model"
        assert info["device"] == "cpu"
        assert info["sample_rate"] == 24000

        # Verify we can synthesize after checking info
        synth_response = tts_client.post(
            "/synthesize",
            json={"text": "Test", "output_format": "wav"},
        )
        assert synth_response.status_code == 200

    def test_multiple_synthesis_requests(self, tts_client):
        """Test multiple synthesis requests in sequence."""
        for i in range(5):
            response = tts_client.post(
                "/synthesize",
                json={"text": f"Test message {i}", "output_format": "wav"},
            )
            assert response.status_code == 200

    def test_different_output_formats(self, tts_client):
        """Test both WAV and PCM output formats."""
        # WAV format
        wav_response = tts_client.post(
            "/synthesize",
            json={"text": "Test", "output_format": "wav"},
        )
        assert wav_response.status_code == 200
        assert wav_response.headers["content-type"] == "audio/wav"

        # PCM format
        pcm_response = tts_client.post(
            "/synthesize",
            json={"text": "Test", "output_format": "pcm"},
        )
        assert pcm_response.status_code == 200
        assert pcm_response.headers["content-type"] == "audio/pcm"


class TestAgentIntegration:
    """Integration tests for Agent service."""

    @pytest.fixture
    def agent_client(self):
        """Create agent test client with mocked dependencies."""
        with patch("agent_app.api.get_llm_client") as mock_llm, \
             patch("agent_app.api.get_tts_client") as mock_tts:

            llm_client = AsyncMock()
            llm_client.generate.return_value = "Hello! How can I help you?"

            async def mock_streaming(messages):
                for word in ["Hello", " ", "World"]:
                    yield word

            llm_client.generate_streaming = mock_streaming
            mock_llm.return_value = llm_client

            tts_client = AsyncMock()
            tts_client.health_check.return_value = True
            tts_client.synthesize.return_value = b"fake-audio"

            async def mock_tts_streaming(text, voice_id=None):
                yield b"audio1"
                yield b"audio2"

            tts_client.synthesize_streaming = mock_tts_streaming
            mock_tts.return_value = tts_client

            yield TestClient(agent_app, raise_server_exceptions=False)

    def test_health_to_chat_flow(self, agent_client):
        """Test complete flow from health check to chat."""
        # Check health
        health_response = agent_client.get("/health")
        assert health_response.status_code == 200
        assert health_response.json()["status"] == "healthy"

        # Chat request
        chat_response = agent_client.post(
            "/chat",
            json={"text": "Hello", "include_audio": False},
        )
        assert chat_response.status_code == 200
        assert "text" in chat_response.json()
        assert "processing_time_ms" in chat_response.json()

    def test_chat_without_audio(self, agent_client):
        """Test chat without audio generation."""
        response = agent_client.post(
            "/chat",
            json={"text": "Tell me a joke", "include_audio": False},
        )
        assert response.status_code == 200

        data = response.json()
        assert "text" in data
        assert len(data["text"]) > 0

    def test_direct_synthesis_integration(self, agent_client):
        """Test direct synthesis endpoint."""
        response = agent_client.post(
            "/synthesize",
            params={"text": "Hello world"},
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/wav"

    def test_multiple_chat_requests(self, agent_client):
        """Test multiple chat requests."""
        messages = ["Hello", "How are you?", "Tell me about AI", "Goodbye"]

        for msg in messages:
            response = agent_client.post(
                "/chat",
                json={"text": msg, "include_audio": False},
            )
            assert response.status_code == 200
            assert "text" in response.json()

    def test_conversation_flow(self, agent_client):
        """Test conversation with conversation ID."""
        conv_id = "test-conv-123"

        # First message
        response1 = agent_client.post(
            "/chat",
            json={
                "text": "Hello",
                "conversation_id": conv_id,
                "include_audio": False,
            },
        )
        assert response1.status_code == 200
        assert response1.json()["conversation_id"] == conv_id

        # Follow-up message
        response2 = agent_client.post(
            "/chat",
            json={
                "text": "Follow up question",
                "conversation_id": conv_id,
                "include_audio": False,
            },
        )
        assert response2.status_code == 200
        assert response2.json()["conversation_id"] == conv_id


class TestEndToEnd:
    """End-to-end integration tests."""

    @pytest.fixture
    def both_services(self):
        """Setup both TTS and Agent services."""
        with patch("tts_service.server.get_model_manager") as mock_tts_manager, \
             patch("agent_app.api.get_llm_client") as mock_llm, \
             patch("agent_app.api.get_tts_client") as mock_tts_client:

            # Mock TTS service
            tts_manager = Mock()
            tts_manager.is_loaded = True
            # Create info mock with attributes set explicitly
            info_mock = Mock()
            info_mock.name = "test-model"
            info_mock.device = "cpu"
            info_mock.dtype = "float32"
            info_mock.sample_rate = 24000
            info_mock.warmup_completed = True
            tts_manager.info = info_mock
            tts_manager.synthesize.return_value = torch.randn(24000)
            mock_tts_manager.return_value = tts_manager

            # Mock LLM client
            llm = AsyncMock()
            llm.generate.return_value = "Test response"
            mock_llm.return_value = llm

            # Mock TTS client (used by agent)
            tts_client = AsyncMock()
            tts_client.health_check.return_value = True
            tts_client.synthesize.return_value = b"audio-data"
            mock_tts_client.return_value = tts_client

            yield {
                "tts": TestClient(tts_app, raise_server_exceptions=False),
                "agent": TestClient(agent_app, raise_server_exceptions=False),
            }

    def test_full_pipeline(self, both_services):
        """Test complete pipeline from agent to TTS."""
        tts_client = both_services["tts"]
        agent_client = both_services["agent"]

        # Verify TTS is healthy
        tts_health = tts_client.get("/health")
        assert tts_health.status_code == 200

        # Verify agent is healthy
        agent_health = agent_client.get("/health")
        assert agent_health.status_code == 200

        # Agent chat request (which internally calls TTS)
        chat_response = agent_client.post(
            "/chat",
            json={"text": "Hello", "include_audio": False},
        )
        assert chat_response.status_code == 200

    def test_metrics_endpoints(self, both_services):
        """Test metrics endpoints on both services."""
        tts_client = both_services["tts"]
        agent_client = both_services["agent"]

        # TTS metrics
        tts_metrics = tts_client.get("/metrics")
        assert tts_metrics.status_code == 200
        assert "text/plain" in tts_metrics.headers["content-type"]

        # Agent metrics
        agent_metrics = agent_client.get("/metrics")
        assert agent_metrics.status_code == 200
        assert "text/plain" in agent_metrics.headers["content-type"]

    def test_error_propagation(self, both_services):
        """Test error handling across services."""
        agent_client = both_services["agent"]

        # Invalid request should return 422
        response = agent_client.post(
            "/chat",
            json={"text": ""},  # Empty text
        )
        assert response.status_code == 422


class TestServiceResilience:
    """Test service resilience and error handling."""

    def test_tts_graceful_degradation(self):
        """Test TTS service when model fails to load."""
        with patch("tts_service.server.get_model_manager") as mock_manager:
            manager = Mock()
            manager.is_loaded = False
            manager.info = None
            mock_manager.return_value = manager

            client = TestClient(tts_app, raise_server_exceptions=False)

            # Health should still respond
            health = client.get("/health")
            assert health.status_code == 200
            assert health.json()["status"] == "unhealthy"

            # Synthesis should fail gracefully
            synth = client.post(
                "/synthesize",
                json={"text": "Test", "output_format": "wav"},
            )
            assert synth.status_code == 503

    def test_agent_tts_unavailable(self):
        """Test agent service when TTS is unavailable."""
        from agent_app.tts_client import TTSServiceUnavailable

        with patch("agent_app.api.get_llm_client") as mock_llm, \
             patch("agent_app.api.get_tts_client") as mock_tts:

            llm_client = AsyncMock()
            llm_client.generate.return_value = "Test response"
            mock_llm.return_value = llm_client

            tts_client = AsyncMock()
            tts_client.health_check.return_value = False
            tts_client.synthesize.side_effect = TTSServiceUnavailable("TTS down")
            mock_tts.return_value = tts_client

            client = TestClient(agent_app, raise_server_exceptions=False)

            # Chat without audio should still work
            response = client.post(
                "/chat",
                json={"text": "Hello", "include_audio": False},
            )
            assert response.status_code == 200

    def test_concurrent_requests(self):
        """Test handling concurrent requests."""
        with patch("tts_service.server.get_model_manager") as mock_manager:
            manager = Mock()
            manager.is_loaded = True
            # Create info mock with attributes set explicitly
            info_mock = Mock()
            info_mock.name = "test-model"
            info_mock.device = "cpu"
            info_mock.dtype = "float32"
            info_mock.sample_rate = 24000
            info_mock.warmup_completed = True
            manager.info = info_mock
            manager.synthesize.return_value = torch.randn(24000)
            mock_manager.return_value = manager

            client = TestClient(tts_app)

            # Simulate concurrent requests (sequential in test, but tests thread safety)
            responses = []
            for i in range(10):
                response = client.post(
                    "/synthesize",
                    json={"text": f"Message {i}", "output_format": "wav"},
                )
                responses.append(response)

            # All should succeed
            assert all(r.status_code == 200 for r in responses)
