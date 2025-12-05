"""
Unit tests for the smoke test helper script.

Tests the health check and TTS synthesis validation logic, including retry handling
and failure scenarios.
"""

import io
import json
import sys
import time
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add scripts directory to path
scripts_dir = Path(__file__).parent.parent.parent / "scripts"
sys.path.insert(0, str(scripts_dir))

from smoke_test import check_health, test_tts_synthesis as verify_tts_synthesis, main


class TestCheckHealth:
    """Tests for the check_health function."""

    @patch("urllib.request.urlopen")
    def test_health_check_success(self, mock_urlopen):
        """Test successful health check."""
        # Mock response
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.read.return_value = json.dumps({"status": "healthy"}).encode("utf-8")
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response

        assert check_health("http://localhost:8000") is True
        mock_urlopen.assert_called_once()

    @patch("urllib.request.urlopen")
    def test_health_check_failure_status(self, mock_urlopen):
        """Test health check with non-200 status."""
        mock_response = MagicMock()
        mock_response.status = 500
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response

        assert check_health("http://localhost:8000") is False

    @patch("urllib.request.urlopen")
    def test_health_check_unhealthy_status(self, mock_urlopen):
        """Test health check with unhealthy status in body."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.read.return_value = json.dumps({"status": "unhealthy"}).encode("utf-8")
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response

        assert check_health("http://localhost:8000") is False

    @patch("urllib.request.urlopen")
    def test_health_check_connection_error(self, mock_urlopen):
        """Test health check with connection error."""
        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")
        assert check_health("http://localhost:8000") is False

    @patch("urllib.request.urlopen")
    def test_health_check_timeout(self, mock_urlopen):
        """Test health check with timeout."""
        import socket
        mock_urlopen.side_effect = socket.timeout("Request timed out")
        assert check_health("http://localhost:8000", timeout=1) is False

    @patch("urllib.request.urlopen")
    def test_health_check_malformed_json(self, mock_urlopen):
        """Test health check with malformed JSON response."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.read.return_value = b"not valid json"
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response

        assert check_health("http://localhost:8000") is False

    @patch("urllib.request.urlopen")
    def test_health_check_missing_status_field(self, mock_urlopen):
        """Test health check with missing status field in response."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.read.return_value = json.dumps({"message": "ok"}).encode("utf-8")
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response

        assert check_health("http://localhost:8000") is False

    @patch("urllib.request.urlopen")
    def test_health_check_custom_timeout(self, mock_urlopen):
        """Test health check respects custom timeout parameter."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.read.return_value = json.dumps({"status": "healthy"}).encode("utf-8")
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response

        check_health("http://localhost:8000", timeout=10)
        
        # Verify timeout was passed to urlopen
        call_args = mock_urlopen.call_args
        assert call_args[1]["timeout"] == 10


class TestTTSValidation:
    """Tests for the test_tts_synthesis function."""

    @patch("urllib.request.urlopen")
    def test_synthesis_success(self, mock_urlopen):
        """Test successful TTS synthesis."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.read.return_value = b"audio_data"
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response

        assert verify_tts_synthesis("http://localhost:8000") is True

    @patch("urllib.request.urlopen")
    def test_synthesis_failure_status(self, mock_urlopen):
        """Test synthesis with non-200 status."""
        mock_response = MagicMock()
        mock_response.status = 400
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response

        assert verify_tts_synthesis("http://localhost:8000") is False

    @patch("urllib.request.urlopen")
    def test_synthesis_empty_response(self, mock_urlopen):
        """Test synthesis with empty response body."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.read.return_value = b""
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response

        assert verify_tts_synthesis("http://localhost:8000") is False

    @patch("urllib.request.urlopen")
    def test_synthesis_connection_error(self, mock_urlopen):
        """Test synthesis with connection error."""
        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")
        assert verify_tts_synthesis("http://localhost:8000") is False

    @patch("urllib.request.urlopen")
    def test_synthesis_timeout(self, mock_urlopen):
        """Test synthesis with timeout."""
        import socket
        mock_urlopen.side_effect = socket.timeout("Request timed out")
        assert verify_tts_synthesis("http://localhost:8000", timeout=5) is False

    @patch("urllib.request.urlopen")
    def test_synthesis_http_error(self, mock_urlopen):
        """Test synthesis with HTTP error (e.g., 500)."""
        mock_urlopen.side_effect = urllib.error.HTTPError(
            "http://localhost:8000/v1/audio/speech",
            500,
            "Internal Server Error",
            {},
            None
        )
        assert verify_tts_synthesis("http://localhost:8000") is False

    @patch("urllib.request.urlopen")
    def test_synthesis_custom_text(self, mock_urlopen):
        """Test synthesis with custom text input."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.read.return_value = b"audio_data_for_custom_text"
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response

        assert verify_tts_synthesis("http://localhost:8000", text="Custom test text") is True

    @patch("urllib.request.urlopen")
    def test_synthesis_custom_timeout(self, mock_urlopen):
        """Test synthesis respects custom timeout parameter."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.read.return_value = b"audio_data"
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response

        verify_tts_synthesis("http://localhost:8000", timeout=15)
        
        # Verify timeout was passed to urlopen
        call_args = mock_urlopen.call_args
        assert call_args[1]["timeout"] == 15

    @patch("urllib.request.urlopen")
    def test_synthesis_request_payload(self, mock_urlopen):
        """Test synthesis sends correct request payload."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.read.return_value = b"audio_data"
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response

        verify_tts_synthesis("http://localhost:8000", text="Test message")
        
        # Verify request was made with correct payload
        call_args = mock_urlopen.call_args
        request = call_args[0][0]
        
        # Check the request data
        payload = json.loads(request.data.decode("utf-8"))
        assert payload["input"] == "Test message"
        assert payload["voice"] == "alloy"
        assert payload["model"] == "tts-1"
        assert payload["response_format"] == "mp3"


class TestMain:
    """Tests for the main function."""

    @patch("smoke_test.check_health")
    @patch("smoke_test.test_tts_synthesis")
    @patch("sys.exit")
    def test_main_success(self, mock_exit, mock_synthesis, mock_health):
        """Test main execution flow - success."""
        mock_health.return_value = True
        mock_synthesis.return_value = True
        
        # Mock args
        with patch("sys.argv", ["smoke_test.py", "--url", "http://localhost:8000"]):
            main()
            
        mock_exit.assert_called_with(0)

    @patch("smoke_test.check_health")
    @patch("sys.exit")
    def test_main_health_failure(self, mock_exit, mock_health):
        """Test main execution flow - health check failure."""
        mock_health.return_value = False
        
        with patch("sys.argv", ["smoke_test.py", "--url", "http://localhost:8000", "--retries", "1", "--delay", "0"]):
            main()
            
        # Should exit with 1. Note: since sys.exit is mocked, execution continues,
        # so we check if it was called with 1 at any point.
        mock_exit.assert_any_call(1)

    @patch("smoke_test.check_health")
    @patch("smoke_test.test_tts_synthesis")
    @patch("sys.exit")
    def test_main_synthesis_failure(self, mock_exit, mock_synthesis, mock_health):
        """Test main execution flow - synthesis failure."""
        mock_health.return_value = True
        mock_synthesis.return_value = False
        
        with patch("sys.argv", ["smoke_test.py", "--url", "http://localhost:8000"]):
            main()
            
        mock_exit.assert_any_call(1)

    @patch("smoke_test.check_health")
    @patch("smoke_test.test_tts_synthesis")
    @patch("time.sleep")
    @patch("sys.exit")
    def test_main_retry_logic(self, mock_exit, mock_sleep, mock_synthesis, mock_health):
        """Test main execution with retry logic."""
        # Health check fails twice, then succeeds
        mock_health.side_effect = [False, False, True]
        mock_synthesis.return_value = True
        
        with patch("sys.argv", ["smoke_test.py", "--url", "http://localhost:8000", "--retries", "3", "--delay", "1"]):
            main()
            
        # Should have called health check 3 times
        assert mock_health.call_count == 3
        # Should have slept twice (between retries)
        assert mock_sleep.call_count == 2
        mock_sleep.assert_called_with(1)
        # Should exit successfully
        mock_exit.assert_called_with(0)

    @patch("smoke_test.check_health")
    @patch("time.sleep")
    @patch("sys.exit")
    def test_main_all_retries_exhausted(self, mock_exit, mock_sleep, mock_health):
        """Test main execution when all retries are exhausted."""
        mock_health.return_value = False
        
        with patch("sys.argv", ["smoke_test.py", "--url", "http://localhost:8000", "--retries", "2", "--delay", "0"]):
            main()
            
        # Should have called health check 2 times
        assert mock_health.call_count == 2
        # Should exit with failure
        mock_exit.assert_any_call(1)

    @patch("smoke_test.check_health")
    @patch("smoke_test.test_tts_synthesis")
    @patch("sys.exit")
    def test_main_custom_timeout(self, mock_exit, mock_synthesis, mock_health):
        """Test main execution with custom timeout."""
        mock_health.return_value = True
        mock_synthesis.return_value = True
        
        with patch("sys.argv", ["smoke_test.py", "--url", "http://localhost:8000", "--timeout", "20"]):
            main()
            
        # Verify timeout was passed to both functions
        mock_health.assert_called_with("http://localhost:8000", 20)
        mock_synthesis.assert_called_with("http://localhost:8000", timeout=20)
        mock_exit.assert_called_with(0)

    @patch("smoke_test.check_health")
    @patch("smoke_test.test_tts_synthesis")
    @patch("sys.exit")
    def test_main_url_trailing_slash(self, mock_exit, mock_synthesis, mock_health):
        """Test main execution handles URLs with trailing slashes."""
        mock_health.return_value = True
        mock_synthesis.return_value = True
        
        with patch("sys.argv", ["smoke_test.py", "--url", "http://localhost:8000/"]):
            main()
            
        # Functions should be called with the URL (they handle trailing slash internally)
        mock_health.assert_called()
        mock_synthesis.assert_called()
        mock_exit.assert_called_with(0)
