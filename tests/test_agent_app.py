"""Unit tests for agent service."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, Mock, patch, MagicMock

from agent_app.api import app
from agent_app.tts_client import TTSClient, TTSClientError, TTSServiceUnavailable


class TestAgentEndpoints:
    """Test agent API endpoints."""

    @pytest.fixture
    def mock_clients(self):
        """Create mocked LLM and TTS clients."""
        with patch("agent_app.api.get_llm_client") as mock_llm, \
             patch("agent_app.api.get_tts_client") as mock_tts:
            
            llm_client = AsyncMock()
            llm_client.generate.return_value = "Hello! How can I help you?"
            mock_llm.return_value = llm_client
            
            tts_client = AsyncMock()
            tts_client.health_check.return_value = True
            tts_client.synthesize.return_value = b"fake-audio-data"
            mock_tts.return_value = tts_client
            
            yield {"llm": llm_client, "tts": tts_client}

    @pytest.fixture
    def client(self, mock_clients):
        """Create test client."""
        return TestClient(app, raise_server_exceptions=False)

    def test_health_check(self, client, mock_clients):
        """Test health endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "llm_provider" in data

    def test_chat_success(self, client, mock_clients):
        """Test successful chat request."""
        response = client.post(
            "/chat",
            json={"text": "Hello", "include_audio": False},
        )
        assert response.status_code == 200
        data = response.json()
        assert "text" in data
        assert "processing_time_ms" in data

    def test_chat_empty_text(self, client, mock_clients):
        """Test chat with empty text."""
        response = client.post(
            "/chat",
            json={"text": "", "include_audio": False},
        )
        assert response.status_code == 422

    def test_chat_llm_error(self, client, mock_clients):
        """Test chat when LLM fails."""
        mock_clients["llm"].generate.side_effect = Exception("LLM error")
        
        response = client.post(
            "/chat",
            json={"text": "Hello", "include_audio": False},
        )
        assert response.status_code == 500

    def test_synthesize_endpoint(self, client, mock_clients):
        """Test direct synthesis endpoint."""
        response = client.post(
            "/synthesize",
            params={"text": "Hello world"},
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/wav"

    def test_chat_with_audio(self, client, mock_clients):
        """Test chat with audio generation."""
        response = client.post(
            "/chat",
            json={"text": "Hello", "include_audio": True},
        )
        assert response.status_code == 200
        data = response.json()
        assert "text" in data
        assert "processing_time_ms" in data

    def test_chat_with_voice_id(self, client, mock_clients):
        """Test chat with specific voice ID."""
        response = client.post(
            "/chat",
            json={"text": "Hello", "voice_id": "voice-1", "include_audio": False},
        )
        assert response.status_code == 200

    def test_chat_with_conversation_id(self, client, mock_clients):
        """Test chat with conversation ID."""
        response = client.post(
            "/chat",
            json={"text": "Hello", "conversation_id": "conv-123", "include_audio": False},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["conversation_id"] == "conv-123"

    def test_chat_streaming_audio(self, client, mock_clients):
        """Test streaming audio response."""
        async def mock_streaming():
            yield b"chunk1"
            yield b"chunk2"

        mock_clients["tts"].synthesize_streaming.return_value = mock_streaming()

        response = client.post(
            "/chat?stream=true",
            json={"text": "Hello", "include_audio": True},
        )
        assert response.status_code == 200
        assert "X-Response-Text" in response.headers
        assert "X-Processing-Time-Ms" in response.headers

    def test_chat_tts_unavailable(self, client, mock_clients):
        """Test chat when TTS service is unavailable in streaming mode."""
        from agent_app.tts_client import TTSServiceUnavailable

        # TTS is only called when stream=true AND include_audio=true
        async def mock_streaming_error(text, voice_id=None):
            raise TTSServiceUnavailable("TTS down")
            yield  # Make it a generator

        mock_clients["tts"].synthesize_streaming = mock_streaming_error

        response = client.post(
            "/chat?stream=true",
            json={"text": "Hello", "include_audio": True},
        )
        # Should return 200 because StreamingResponse sends headers before generator error
        assert response.status_code == 200

    def test_synthesize_with_voice_id(self, client, mock_clients):
        """Test synthesis with voice ID."""
        response = client.post(
            "/synthesize",
            params={"text": "Hello", "voice_id": "voice-1"},
        )
        assert response.status_code == 200

    def test_synthesize_tts_error(self, client, mock_clients):
        """Test synthesis when TTS client fails."""
        from agent_app.tts_client import TTSClientError

        mock_clients["tts"].synthesize.side_effect = TTSClientError("TTS failed")

        response = client.post(
            "/synthesize",
            params={"text": "Hello"},
        )
        assert response.status_code == 503

    def test_metrics_endpoint(self, client):
        """Test Prometheus metrics endpoint."""
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]


class TestTTSClient:
    """Test TTS client."""

    @pytest.fixture
    def client(self):
        """Create TTS client with mocked settings."""
        with patch("agent_app.tts_client.get_agent_settings") as mock_settings:
            mock_settings.return_value = Mock(
                tts_base_url="http://localhost:8000",
                tts_timeout=30.0,
            )
            return TTSClient()

    @pytest.mark.asyncio
    async def test_connect(self, client):
        """Test client connection."""
        await client.connect()
        assert client._client is not None

    @pytest.mark.asyncio
    async def test_close(self, client):
        """Test client close."""
        await client.connect()
        await client.close()
        assert client._client is None

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager."""
        with patch("agent_app.tts_client.get_agent_settings") as mock_settings:
            mock_settings.return_value = Mock(
                tts_base_url="http://localhost:8000",
                tts_timeout=30.0,
            )

            async with TTSClient() as client:
                assert client._client is not None

            assert client._client is None

    @pytest.mark.asyncio
    async def test_client_property_not_connected(self, client):
        """Test accessing client property when not connected."""
        with pytest.raises(RuntimeError, match="not connected"):
            _ = client.client

    @pytest.mark.asyncio
    async def test_health_check_success(self, client):
        """Test successful health check."""
        with patch.object(client, "_client") as mock_http:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"model_loaded": True}
            mock_http.get = AsyncMock(return_value=mock_response)

            result = await client.health_check()
            assert result is True

    @pytest.mark.asyncio
    async def test_health_check_failure(self, client):
        """Test health check when service is down."""
        with patch.object(client, "_client") as mock_http:
            import httpx
            mock_http.get = AsyncMock(side_effect=httpx.ConnectError("Connection failed"))

            result = await client.health_check()
            assert result is False

    @pytest.mark.asyncio
    async def test_health_check_unhealthy_status(self, client):
        """Test health check with non-200 status."""
        with patch.object(client, "_client") as mock_http:
            mock_response = Mock()
            mock_response.status_code = 503
            mock_http.get = AsyncMock(return_value=mock_response)

            result = await client.health_check()
            assert result is False

    @pytest.mark.asyncio
    async def test_wait_for_ready_success(self, client):
        """Test waiting for service to be ready."""
        with patch.object(client, "_client") as mock_http:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_http.get = AsyncMock(return_value=mock_response)

            result = await client.wait_for_ready(timeout=5.0, interval=1.0)
            assert result is True

    @pytest.mark.asyncio
    async def test_wait_for_ready_timeout(self, client):
        """Test wait for ready timeout."""
        with patch.object(client, "_client") as mock_http:
            import httpx
            mock_http.get = AsyncMock(side_effect=httpx.HTTPError("Error"))

            result = await client.wait_for_ready(timeout=2.0, interval=1.0)
            assert result is False

    @pytest.mark.asyncio
    async def test_synthesize_success(self, client):
        """Test successful synthesis."""
        with patch.object(client, "_client") as mock_http:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.content = b"audio-data"
            mock_http.post = AsyncMock(return_value=mock_response)

            audio = await client.synthesize("Hello world")
            assert audio == b"audio-data"

    @pytest.mark.asyncio
    async def test_synthesize_with_voice_id(self, client):
        """Test synthesis with voice ID."""
        with patch.object(client, "_client") as mock_http:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.content = b"audio-data"
            mock_http.post = AsyncMock(return_value=mock_response)

            audio = await client.synthesize("Hello", voice_id="voice-1", output_format="pcm")
            assert audio == b"audio-data"

    @pytest.mark.asyncio
    async def test_synthesize_service_unavailable(self, client):
        """Test synthesis when service returns 503."""
        from agent_app.tts_client import TTSServiceUnavailable

        with patch.object(client, "_client") as mock_http:
            mock_response = Mock()
            mock_response.status_code = 503
            mock_http.post = AsyncMock(return_value=mock_response)

            with pytest.raises(TTSServiceUnavailable):
                await client.synthesize("Hello")

    @pytest.mark.asyncio
    async def test_synthesize_error_response(self, client):
        """Test synthesis with error response."""
        from agent_app.tts_client import TTSSynthesisError

        with patch.object(client, "_client") as mock_http:
            mock_response = Mock()
            mock_response.status_code = 400
            mock_response.json.return_value = {"detail": "Invalid text"}
            mock_http.post = AsyncMock(return_value=mock_response)

            with pytest.raises(TTSSynthesisError, match="Invalid text"):
                await client.synthesize("Hello")

    @pytest.mark.asyncio
    async def test_synthesize_connection_error(self, client):
        """Test synthesis with connection error."""
        from agent_app.tts_client import TTSServiceUnavailable
        import httpx

        with patch.object(client, "_client") as mock_http:
            mock_http.post = AsyncMock(side_effect=httpx.ConnectError("Failed"))

            with pytest.raises(TTSServiceUnavailable, match="Cannot connect"):
                await client.synthesize("Hello")

    @pytest.mark.asyncio
    async def test_synthesize_streaming(self, client):
        """Test streaming synthesis."""
        with patch.object(client, "_client") as mock_http:
            mock_response = Mock()
            mock_response.status_code = 200

            async def mock_aiter_bytes(chunk_size):
                yield b"chunk1"
                yield b"chunk2"

            mock_response.aiter_bytes = mock_aiter_bytes

            class MockStream:
                async def __aenter__(self):
                    return mock_response

                async def __aexit__(self, *args):
                    pass

            mock_http.stream = Mock(return_value=MockStream())

            chunks = []
            async for chunk in client.synthesize_streaming("Hello"):
                chunks.append(chunk)

            assert len(chunks) == 2
            assert chunks[0] == b"chunk1"
            assert chunks[1] == b"chunk2"

    @pytest.mark.asyncio
    async def test_synthesize_streaming_error(self, client):
        """Test streaming synthesis with error."""
        from agent_app.tts_client import TTSServiceUnavailable
        import httpx

        with patch.object(client, "_client") as mock_http:
            mock_http.stream = Mock(side_effect=httpx.ConnectError("Failed"))

            with pytest.raises(TTSServiceUnavailable):
                async for _ in client.synthesize_streaming("Hello"):
                    pass


class TestLLMClients:
    """Test LLM client implementations."""

    def test_create_anthropic_client(self):
        """Test creating Anthropic client."""
        from agent_app.llm_client import create_llm_client

        with patch("agent_app.llm_client.get_agent_settings") as mock_settings:
            mock_settings.return_value = Mock(
                llm_provider="anthropic",
                llm_model="claude-sonnet-4-20250514",
                llm_max_tokens=1024,
                llm_temperature=0.7,
                system_prompt="Test prompt",
            )

            client = create_llm_client()
            assert client is not None

    def test_create_openai_client(self):
        """Test creating OpenAI client."""
        from agent_app.llm_client import create_llm_client

        with patch("agent_app.llm_client.get_agent_settings") as mock_settings:
            mock_settings.return_value = Mock(
                llm_provider="openai",
                llm_model="gpt-4",
                llm_max_tokens=1024,
                llm_temperature=0.7,
                system_prompt="Test prompt",
            )

            client = create_llm_client()
            assert client is not None

    def test_create_invalid_provider(self):
        """Test creating client with invalid provider."""
        from agent_app.llm_client import create_llm_client

        with patch("agent_app.llm_client.get_agent_settings") as mock_settings:
            mock_settings.return_value = Mock(llm_provider="invalid")

            with pytest.raises(ValueError, match="Unknown LLM provider"):
                create_llm_client()

    @pytest.mark.asyncio
    async def test_anthropic_generate(self):
        """Test Anthropic client generate method."""
        from agent_app.llm_client import AnthropicClient

        settings = Mock(
            llm_model="claude-sonnet-4-20250514",
            llm_max_tokens=1024,
            system_prompt="Test prompt",
        )

        client = AnthropicClient(settings)

        # Mock anthropic module since it might not be installed
        mock_anthropic_module = MagicMock()
        mock_instance = AsyncMock()
        mock_message = Mock()
        mock_message.content = [Mock(text="Hello!")]
        mock_instance.messages.create = AsyncMock(return_value=mock_message)
        mock_anthropic_module.AsyncAnthropic.return_value = mock_instance

        with patch.dict("sys.modules", {"anthropic": mock_anthropic_module}):
            from agent_app.llm_client import AnthropicClient
            
            client = AnthropicClient(settings)
            response = await client.generate([{"role": "user", "content": "Hi"}])
            assert response == "Hello!"

    @pytest.mark.asyncio
    async def test_openai_generate(self):
        """Test OpenAI client generate method."""
        from agent_app.llm_client import OpenAIClient

        settings = Mock(
            llm_model="gpt-4",
            llm_max_tokens=1024,
            llm_temperature=0.7,
            system_prompt="Test prompt",
        )

        client = OpenAIClient(settings)

        # Mock openai module
        mock_openai_module = MagicMock()
        mock_instance = AsyncMock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Hello!"))]
        mock_instance.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_openai_module.AsyncOpenAI.return_value = mock_instance

        with patch.dict("sys.modules", {"openai": mock_openai_module}):
            from agent_app.llm_client import OpenAIClient
            
            client = OpenAIClient(settings)
            response = await client.generate([{"role": "user", "content": "Hi"}])
            assert response == "Hello!"

    @pytest.mark.asyncio
    async def test_local_client_not_implemented(self):
        """Test local LLM client raises NotImplementedError."""
        from agent_app.llm_client import LocalLLMClient

        settings = Mock()
        client = LocalLLMClient(settings)

        with pytest.raises(NotImplementedError):
            await client.generate([{"role": "user", "content": "Hi"}])
