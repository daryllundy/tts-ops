"""LLM client abstraction supporting multiple providers."""

from abc import ABC, abstractmethod
from typing import AsyncGenerator

from common.config import AgentServiceSettings, get_agent_settings
from common.logging import get_logger

logger = get_logger(__name__)


class LLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    async def generate(
        self,
        messages: list[dict[str, str]],
        system_prompt: str | None = None,
    ) -> str:
        """Generate a response from the LLM."""

    @abstractmethod
    async def generate_streaming(
        self,
        messages: list[dict[str, str]],
        system_prompt: str | None = None,
    ) -> AsyncGenerator[str, None]:
        """Generate a streaming response from the LLM."""


class AnthropicClient(LLMClient):
    """Anthropic Claude API client."""

    def __init__(self, settings: AgentServiceSettings) -> None:
        self.settings = settings
        self._client = None

    async def _get_client(self):
        """Lazy initialization of Anthropic client."""
        if self._client is None:
            try:
                from anthropic import AsyncAnthropic

                self._client = AsyncAnthropic()
            except ImportError:
                raise RuntimeError("anthropic package not installed. Install with: pip install anthropic")
        return self._client

    async def generate(
        self,
        messages: list[dict[str, str]],
        system_prompt: str | None = None,
    ) -> str:
        client = await self._get_client()

        response = await client.messages.create(
            model=self.settings.llm_model,
            max_tokens=self.settings.llm_max_tokens,
            system=system_prompt or self.settings.system_prompt,
            messages=messages,
        )

        return response.content[0].text

    async def generate_streaming(
        self,
        messages: list[dict[str, str]],
        system_prompt: str | None = None,
    ) -> AsyncGenerator[str, None]:
        client = await self._get_client()

        async with client.messages.stream(
            model=self.settings.llm_model,
            max_tokens=self.settings.llm_max_tokens,
            system=system_prompt or self.settings.system_prompt,
            messages=messages,
        ) as stream:
            async for text in stream.text_stream:
                yield text


class OpenAIClient(LLMClient):
    """OpenAI API client."""

    def __init__(self, settings: AgentServiceSettings) -> None:
        self.settings = settings
        self._client = None

    async def _get_client(self):
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            try:
                from openai import AsyncOpenAI

                self._client = AsyncOpenAI()
            except ImportError:
                raise RuntimeError("openai package not installed. Install with: pip install openai")
        return self._client

    async def generate(
        self,
        messages: list[dict[str, str]],
        system_prompt: str | None = None,
    ) -> str:
        client = await self._get_client()

        full_messages = []
        if system_prompt or self.settings.system_prompt:
            full_messages.append({
                "role": "system",
                "content": system_prompt or self.settings.system_prompt,
            })
        full_messages.extend(messages)

        response = await client.chat.completions.create(
            model=self.settings.llm_model,
            max_tokens=self.settings.llm_max_tokens,
            temperature=self.settings.llm_temperature,
            messages=full_messages,
        )

        return response.choices[0].message.content or ""

    async def generate_streaming(
        self,
        messages: list[dict[str, str]],
        system_prompt: str | None = None,
    ) -> AsyncGenerator[str, None]:
        client = await self._get_client()

        full_messages = []
        if system_prompt or self.settings.system_prompt:
            full_messages.append({
                "role": "system",
                "content": system_prompt or self.settings.system_prompt,
            })
        full_messages.extend(messages)

        stream = await client.chat.completions.create(
            model=self.settings.llm_model,
            max_tokens=self.settings.llm_max_tokens,
            temperature=self.settings.llm_temperature,
            messages=full_messages,
            stream=True,
        )

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class LocalLLMClient(LLMClient):
    """Client for local LLM inference (placeholder for vLLM/Ollama integration)."""

    def __init__(self, settings: AgentServiceSettings) -> None:
        self.settings = settings
        logger.warning("LocalLLMClient is a placeholder - implement for your local LLM setup")

    async def generate(
        self,
        messages: list[dict[str, str]],
        system_prompt: str | None = None,
    ) -> str:
        # Placeholder implementation
        raise NotImplementedError("Local LLM client not implemented")

    async def generate_streaming(
        self,
        messages: list[dict[str, str]],
        system_prompt: str | None = None,
    ) -> AsyncGenerator[str, None]:
        raise NotImplementedError("Local LLM client not implemented")
        yield  # Make this a generator


def create_llm_client(settings: AgentServiceSettings | None = None) -> LLMClient:
    """Factory function to create appropriate LLM client."""
    settings = settings or get_agent_settings()

    clients = {
        "anthropic": AnthropicClient,
        "openai": OpenAIClient,
        "local": LocalLLMClient,
    }

    client_class = clients.get(settings.llm_provider)
    if not client_class:
        raise ValueError(f"Unknown LLM provider: {settings.llm_provider}")

    return client_class(settings)


# Singleton instance
_llm_client: LLMClient | None = None


def get_llm_client() -> LLMClient:
    """Get the global LLM client instance."""
    global _llm_client
    if _llm_client is None:
        _llm_client = create_llm_client()
    return _llm_client
