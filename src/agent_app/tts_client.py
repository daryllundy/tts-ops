"""HTTP client for TTS service communication."""

import asyncio
from collections.abc import AsyncGenerator

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from common.config import AgentServiceSettings, get_agent_settings
from common.logging import get_logger

logger = get_logger(__name__)


class TTSClientError(Exception):
    """Base exception for TTS client errors."""


class TTSServiceUnavailable(TTSClientError):
    """TTS service is not available."""


class TTSSynthesisError(TTSClientError):
    """Error during synthesis."""


class TTSClient:
    """Async HTTP client for TTS service."""

    def __init__(self, settings: AgentServiceSettings | None = None) -> None:
        self.settings = settings or get_agent_settings()
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "TTSClient":
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, *args: object) -> None:
        """Async context manager exit."""
        await self.close()

    async def connect(self) -> None:
        """Initialize the HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.settings.tts_base_url,
                timeout=httpx.Timeout(self.settings.tts_timeout, connect=5.0),
            )

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> httpx.AsyncClient:
        """Get the HTTP client, raising if not connected."""
        if self._client is None:
            raise RuntimeError("Client not connected. Call connect() first.")
        return self._client

    @retry(
        retry=retry_if_exception_type((httpx.ConnectError, httpx.ConnectTimeout)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=5),
    )
    async def health_check(self) -> bool:
        """Check if TTS service is healthy."""
        try:
            response = await self.client.get("/health")
            if response.status_code == 200:
                data = response.json()
                return bool(data.get("model_loaded", False))
            return False
        except httpx.HTTPError as e:
            logger.warning("TTS health check failed", error=str(e))
            return False

    async def wait_for_ready(self, timeout: float = 60.0, interval: float = 2.0) -> bool:
        """Wait for TTS service to become ready."""
        elapsed = 0.0
        while elapsed < timeout:
            try:
                response = await self.client.get("/ready")
                if response.status_code == 200:
                    return True
            except httpx.HTTPError:
                pass

            await asyncio.sleep(interval)
            elapsed += interval

        return False

    @retry(
        retry=retry_if_exception_type((httpx.ConnectError, httpx.ReadTimeout)),
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=3),
    )
    async def synthesize(
        self,
        text: str,
        voice_id: str | None = None,
        output_format: str = "wav",
    ) -> bytes:
        """
        Synthesize speech from text.

        Args:
            text: Text to synthesize
            voice_id: Optional voice identifier
            output_format: Output format (wav or pcm)

        Returns:
            Audio bytes

        Raises:
            TTSServiceUnavailable: If service is not available
            TTSSynthesisError: If synthesis fails
        """
        try:
            response = await self.client.post(
                "/synthesize",
                json={
                    "text": text,
                    "voice_id": voice_id,
                    "output_format": output_format,
                },
            )

            if response.status_code == 503:
                raise TTSServiceUnavailable("TTS service is not available")

            if response.status_code != 200:
                detail = response.json().get("detail", "Unknown error")
                raise TTSSynthesisError(f"Synthesis failed: {detail}")

            return response.content

        except httpx.ConnectError as e:
            raise TTSServiceUnavailable(f"Cannot connect to TTS service: {e}") from e
        except httpx.HTTPError as e:
            raise TTSSynthesisError(f"HTTP error during synthesis: {e}") from e

    async def synthesize_streaming(
        self,
        text: str,
        voice_id: str | None = None,
    ) -> AsyncGenerator[bytes, None]:
        """
        Synthesize speech with streaming response.

        Args:
            text: Text to synthesize
            voice_id: Optional voice identifier

        Yields:
            Audio byte chunks
        """
        try:
            async with self.client.stream(
                "POST",
                "/synthesize",
                params={"stream": "true"},
                json={"text": text, "voice_id": voice_id},
            ) as response:
                if response.status_code == 503:
                    raise TTSServiceUnavailable("TTS service is not available")

                if response.status_code != 200:
                    raise TTSSynthesisError(f"Synthesis failed: {response.status_code}")

                async for chunk in response.aiter_bytes(chunk_size=4096):
                    yield chunk

        except httpx.ConnectError as e:
            raise TTSServiceUnavailable(f"Cannot connect to TTS service: {e}") from e


# Singleton client instance
_client: TTSClient | None = None


def get_tts_client() -> TTSClient:
    """Get the global TTS client instance."""
    global _client
    if _client is None:
        _client = TTSClient()
    return _client
