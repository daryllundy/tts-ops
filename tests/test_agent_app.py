"""Unit tests for agent service."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, Mock, patch

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

    def test_create_invalid_provider(self):
        """Test creating client with invalid provider."""
        from agent_app.llm_client import create_llm_client
        
        with patch("agent_app.llm_client.get_agent_settings") as mock_settings:
            mock_settings.return_value = Mock(llm_provider="invalid")
            
            with pytest.raises(ValueError, match="Unknown LLM provider"):
                create_llm_client()
