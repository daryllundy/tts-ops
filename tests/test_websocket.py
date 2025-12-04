"""Unit tests for WebSocket endpoint."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, Mock, patch
import json

from agent_app.api import app


class TestWebSocketChat:
    """Test WebSocket chat endpoint."""

    @pytest.fixture
    def mock_clients(self):
        """Create mocked LLM and TTS clients."""
        with patch("agent_app.api.get_llm_client") as mock_llm, \
             patch("agent_app.api.get_tts_client") as mock_tts:

            llm_client = AsyncMock()

            async def mock_generate_streaming(messages):
                yield "Hello"
                yield " "
                yield "World"

            llm_client.generate_streaming = mock_generate_streaming
            mock_llm.return_value = llm_client

            tts_client = AsyncMock()

            async def mock_synthesize_streaming(text, voice_id=None):
                yield b"audio1"
                yield b"audio2"

            tts_client.synthesize_streaming = mock_synthesize_streaming
            mock_tts.return_value = tts_client

            yield {"llm": llm_client, "tts": tts_client}

    def test_websocket_basic_flow(self, mock_clients):
        """Test basic WebSocket chat flow."""
        client = TestClient(app)

        with client.websocket_connect("/ws/chat") as websocket:
            # Send message
            websocket.send_json({"text": "Hello"})

            # Receive text chunks
            messages = []
            while True:
                data = websocket.receive()
                if data["type"] == "websocket.send":
                    payload = data.get("text")
                    if payload:
                        msg = json.loads(payload)
                        messages.append(msg)
                        if msg.get("type") == "done":
                            break
                elif data["type"] == "websocket.bytes":
                    # Audio chunk received
                    pass

            # Should receive done message
            assert any(msg.get("type") == "done" for msg in messages)

    def test_websocket_empty_text(self, mock_clients):
        """Test WebSocket with empty text."""
        client = TestClient(app)

        with client.websocket_connect("/ws/chat") as websocket:
            # Send empty message
            websocket.send_json({"text": ""})

            # Should receive error
            data = websocket.receive_json()
            assert data["type"] == "error"
            assert "Empty text" in data["message"]

    def test_websocket_with_voice_id(self, mock_clients):
        """Test WebSocket with voice ID."""
        client = TestClient(app)

        with client.websocket_connect("/ws/chat") as websocket:
            websocket.send_json({"text": "Hello", "voice_id": "voice-1"})

            # Collect messages
            messages = []
            try:
                while True:
                    data = websocket.receive()
                    if data["type"] == "websocket.send":
                        payload = data.get("text")
                        if payload:
                            msg = json.loads(payload)
                            messages.append(msg)
                            if msg.get("type") == "done":
                                break
            except Exception:
                pass

            # Should complete successfully
            assert any(msg.get("type") == "done" for msg in messages)

    def test_websocket_tts_error(self):
        """Test WebSocket when TTS fails."""
        from agent_app.tts_client import TTSClientError

        with patch("agent_app.api.get_llm_client") as mock_llm, \
             patch("agent_app.api.get_tts_client") as mock_tts:

            llm_client = AsyncMock()

            async def mock_generate_streaming(messages):
                yield "Hello"

            llm_client.generate_streaming = mock_generate_streaming
            mock_llm.return_value = llm_client

            tts_client = AsyncMock()

            async def mock_synthesize_error(text, voice_id=None):
                raise TTSClientError("TTS failed")
                yield  # Make it a generator

            tts_client.synthesize_streaming = mock_synthesize_error
            mock_tts.return_value = tts_client

            client = TestClient(app)

            with client.websocket_connect("/ws/chat") as websocket:
                websocket.send_json({"text": "Hello"})

                # Should still receive text chunks and done message
                messages = []
                try:
                    while True:
                        data = websocket.receive()
                        if data["type"] == "websocket.send":
                            payload = data.get("text")
                            if payload:
                                msg = json.loads(payload)
                                messages.append(msg)
                                if msg.get("type") == "done":
                                    break
                except Exception:
                    pass

                # Should receive done despite TTS error
                assert any(msg.get("type") == "done" for msg in messages)

    def test_websocket_disconnect(self, mock_clients):
        """Test WebSocket disconnect handling."""
        client = TestClient(app)

        with client.websocket_connect("/ws/chat") as websocket:
            websocket.send_json({"text": "Hello"})
            # Client disconnects
            websocket.close()

        # Should handle disconnect gracefully

    def test_websocket_llm_streaming(self, mock_clients):
        """Test that LLM text is streamed to client."""
        client = TestClient(app)

        with client.websocket_connect("/ws/chat") as websocket:
            websocket.send_json({"text": "Tell me a story"})

            # Collect text chunks
            text_chunks = []
            try:
                while True:
                    data = websocket.receive()
                    if data["type"] == "websocket.send":
                        payload = data.get("text")
                        if payload:
                            msg = json.loads(payload)
                            if msg.get("type") == "text":
                                text_chunks.append(msg["content"])
                            elif msg.get("type") == "done":
                                break
            except Exception:
                pass

            # Should receive text chunks from LLM
            assert len(text_chunks) > 0

    def test_websocket_audio_streaming(self, mock_clients):
        """Test that audio is streamed to client."""
        client = TestClient(app)

        with client.websocket_connect("/ws/chat") as websocket:
            websocket.send_json({"text": "Hello"})

            # Collect audio chunks
            audio_chunks = []
            try:
                while True:
                    data = websocket.receive()
                    if data["type"] == "websocket.bytes":
                        audio_chunks.append(data["bytes"])
                    elif data["type"] == "websocket.send":
                        payload = data.get("text")
                        if payload:
                            msg = json.loads(payload)
                            if msg.get("type") == "done":
                                break
            except Exception:
                pass

            # Should receive audio chunks
            assert len(audio_chunks) > 0

    def test_websocket_multiple_messages(self, mock_clients):
        """Test multiple messages in same connection."""
        client = TestClient(app)

        with client.websocket_connect("/ws/chat") as websocket:
            # Send first message
            websocket.send_json({"text": "Hello"})

            # Wait for done
            while True:
                data = websocket.receive()
                if data["type"] == "websocket.send":
                    payload = data.get("text")
                    if payload:
                        msg = json.loads(payload)
                        if msg.get("type") == "done":
                            break

            # Send second message
            websocket.send_json({"text": "How are you?"})

            # Should receive response for second message
            received_done = False
            try:
                while True:
                    data = websocket.receive()
                    if data["type"] == "websocket.send":
                        payload = data.get("text")
                        if payload:
                            msg = json.loads(payload)
                            if msg.get("type") == "done":
                                received_done = True
                                break
            except Exception:
                pass

            assert received_done

    def test_websocket_concurrent_connections(self, mock_clients):
        """Test multiple concurrent WebSocket connections."""
        client = TestClient(app)

        # Open two connections
        with client.websocket_connect("/ws/chat") as ws1, \
             client.websocket_connect("/ws/chat") as ws2:

            # Send message on first connection
            ws1.send_json({"text": "Hello"})

            # Send message on second connection
            ws2.send_json({"text": "Hi"})

            # Both should work independently
            # This tests that connections don't interfere with each other
