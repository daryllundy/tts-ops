"""Shared configuration management using Pydantic Settings."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class TTSServiceSettings(BaseSettings):
    """Configuration for the TTS microservice."""

    model_config = SettingsConfigDict(env_prefix="TTS_", env_file=".env", extra="ignore")

    model_name: str = Field(
        default="microsoft/VibeVoice-Realtime-0.5B",
        description="HuggingFace model identifier for VibeVoice",
    )
    device: str = Field(default="auto", description="Inference device (auto, cuda:N, mps, or cpu)")
    max_batch_size: int = Field(default=4, ge=1, le=16, description="Maximum inference batch size")
    max_text_length: int = Field(default=4096, ge=100, description="Maximum input text length")
    sample_rate: int = Field(default=24000, description="Audio output sample rate")
    dtype: Literal["float16", "bfloat16", "float32"] = Field(
        default="float16", description="Model inference dtype"
    )
    warmup_on_start: bool = Field(default=True, description="Run warmup inference on startup")
    cache_dir: str | None = Field(default=None, description="Model cache directory")

    host: str = Field(default="0.0.0.0", description="Server bind host")
    port: int = Field(default=8000, ge=1, le=65535, description="Server bind port")
    workers: int = Field(default=1, ge=1, description="Number of Uvicorn workers")

    @field_validator("device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        if v in ("auto", "mps", "cpu"):
            return v

        # Validate CUDA device format: cuda:N where N is a non-negative integer
        if v.startswith("cuda:"):
            try:
                device_id = v.split(":", 1)[1]
                device_num = int(device_id)
                if device_num >= 0:
                    return v
            except (ValueError, IndexError):
                pass

        raise ValueError("Device must be 'auto', 'cpu', 'mps', or 'cuda:N' where N is a non-negative integer")


class AgentServiceSettings(BaseSettings):
    """Configuration for the voice agent service."""

    model_config = SettingsConfigDict(env_prefix="AGENT_", env_file=".env", extra="ignore")

    host: str = Field(default="0.0.0.0", description="Server bind host")
    port: int = Field(default=8080, ge=1, le=65535, description="Server bind port")
    workers: int = Field(default=2, ge=1, description="Number of Uvicorn workers")

    tts_base_url: str = Field(
        default="http://localhost:8000", description="TTS service base URL"
    )
    tts_timeout: float = Field(default=30.0, ge=1.0, description="TTS request timeout in seconds")

    llm_provider: Literal["anthropic", "openai", "local"] = Field(
        default="anthropic", description="LLM provider"
    )
    llm_model: str = Field(default="claude-sonnet-4-20250514", description="LLM model name")
    llm_max_tokens: int = Field(default=1024, ge=1, description="Maximum LLM response tokens")
    llm_temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="LLM temperature")

    system_prompt: str = Field(
        default="You are a helpful voice assistant. Keep responses concise and conversational.",
        description="System prompt for LLM",
    )


class ObservabilitySettings(BaseSettings):
    """Configuration for logging, metrics, and tracing."""

    model_config = SettingsConfigDict(env_prefix="OBS_", env_file=".env", extra="ignore")

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO", description="Logging level"
    )
    log_format: Literal["json", "console"] = Field(
        default="json", description="Log output format"
    )
    metrics_enabled: bool = Field(default=True, description="Enable Prometheus metrics")
    metrics_port: int = Field(default=9090, description="Prometheus metrics port")
    tracing_enabled: bool = Field(default=False, description="Enable OpenTelemetry tracing")
    tracing_endpoint: str | None = Field(default=None, description="OTLP endpoint for traces")
    service_name: str = Field(default="vibevoice-agent", description="Service name for telemetry")


@lru_cache
def get_tts_settings() -> TTSServiceSettings:
    """Get cached TTS service settings."""
    return TTSServiceSettings()


@lru_cache
def get_agent_settings() -> AgentServiceSettings:
    """Get cached agent service settings."""
    return AgentServiceSettings()


@lru_cache
def get_observability_settings() -> ObservabilitySettings:
    """Get cached observability settings."""
    return ObservabilitySettings()
