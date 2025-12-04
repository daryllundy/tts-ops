"""Unit tests for configuration management."""

import pytest
from unittest.mock import patch
import os

from common.config import (
    TTSServiceSettings,
    AgentServiceSettings,
    ObservabilitySettings,
    get_tts_settings,
    get_agent_settings,
    get_observability_settings,
)


class TestTTSServiceSettings:
    """Test TTS service configuration."""

    def test_default_values(self):
        """Test default configuration values."""
        with patch.dict(os.environ, {}, clear=True):
            settings = TTSServiceSettings()

            assert settings.model_name == "microsoft/VibeVoice-Realtime-0.5B"
            assert settings.device == "cuda:0"
            assert settings.max_batch_size == 4
            assert settings.max_text_length == 4096
            assert settings.sample_rate == 24000
            assert settings.dtype == "float16"
            assert settings.warmup_on_start is True
            assert settings.host == "0.0.0.0"
            assert settings.port == 8000
            assert settings.workers == 1

    def test_env_override(self):
        """Test environment variable override."""
        with patch.dict(os.environ, {
            "TTS_MODEL_NAME": "custom-model",
            "TTS_DEVICE": "cpu",
            "TTS_PORT": "9000",
            "TTS_MAX_BATCH_SIZE": "8",
        }):
            settings = TTSServiceSettings()

            assert settings.model_name == "custom-model"
            assert settings.device == "cpu"
            assert settings.port == 9000
            assert settings.max_batch_size == 8

    def test_device_validation_cuda(self):
        """Test device validation for CUDA."""
        with patch.dict(os.environ, {"TTS_DEVICE": "cuda:0"}):
            settings = TTSServiceSettings()
            assert settings.device == "cuda:0"

        with patch.dict(os.environ, {"TTS_DEVICE": "cuda:1"}):
            settings = TTSServiceSettings()
            assert settings.device == "cuda:1"

    def test_device_validation_cpu(self):
        """Test device validation for CPU."""
        with patch.dict(os.environ, {"TTS_DEVICE": "cpu"}):
            settings = TTSServiceSettings()
            assert settings.device == "cpu"

    def test_device_validation_invalid(self):
        """Test device validation with invalid value."""
        with patch.dict(os.environ, {"TTS_DEVICE": "invalid"}):
            with pytest.raises(ValueError, match="Device must be"):
                TTSServiceSettings()

    def test_dtype_validation(self):
        """Test dtype validation."""
        valid_dtypes = ["float16", "bfloat16", "float32"]

        for dtype in valid_dtypes:
            with patch.dict(os.environ, {"TTS_DTYPE": dtype}):
                settings = TTSServiceSettings()
                assert settings.dtype == dtype

    def test_port_range(self):
        """Test port number validation."""
        with patch.dict(os.environ, {"TTS_PORT": "80"}):
            settings = TTSServiceSettings()
            assert settings.port == 80

        with patch.dict(os.environ, {"TTS_PORT": "65535"}):
            settings = TTSServiceSettings()
            assert settings.port == 65535

    def test_max_batch_size_constraints(self):
        """Test batch size constraints."""
        with patch.dict(os.environ, {"TTS_MAX_BATCH_SIZE": "1"}):
            settings = TTSServiceSettings()
            assert settings.max_batch_size == 1

        with patch.dict(os.environ, {"TTS_MAX_BATCH_SIZE": "16"}):
            settings = TTSServiceSettings()
            assert settings.max_batch_size == 16

    def test_cache_dir_optional(self):
        """Test cache directory is optional."""
        with patch.dict(os.environ, {}, clear=True):
            settings = TTSServiceSettings()
            assert settings.cache_dir is None

        with patch.dict(os.environ, {"TTS_CACHE_DIR": "/tmp/cache"}):
            settings = TTSServiceSettings()
            assert settings.cache_dir == "/tmp/cache"


class TestAgentServiceSettings:
    """Test agent service configuration."""

    def test_default_values(self):
        """Test default configuration values."""
        with patch.dict(os.environ, {}, clear=True):
            settings = AgentServiceSettings()

            assert settings.host == "0.0.0.0"
            assert settings.port == 8080
            assert settings.workers == 2
            assert settings.tts_base_url == "http://localhost:8000"
            assert settings.tts_timeout == 30.0
            assert settings.llm_provider == "anthropic"
            assert settings.llm_model == "claude-sonnet-4-20250514"
            assert settings.llm_max_tokens == 1024
            assert settings.llm_temperature == 0.7
            assert "helpful voice assistant" in settings.system_prompt

    def test_env_override(self):
        """Test environment variable override."""
        with patch.dict(os.environ, {
            "AGENT_PORT": "9090",
            "AGENT_TTS_BASE_URL": "http://tts:8000",
            "AGENT_LLM_PROVIDER": "openai",
            "AGENT_LLM_MODEL": "gpt-4",
            "AGENT_LLM_MAX_TOKENS": "2048",
        }):
            settings = AgentServiceSettings()

            assert settings.port == 9090
            assert settings.tts_base_url == "http://tts:8000"
            assert settings.llm_provider == "openai"
            assert settings.llm_model == "gpt-4"
            assert settings.llm_max_tokens == 2048

    def test_llm_providers(self):
        """Test different LLM providers."""
        providers = ["anthropic", "openai", "local"]

        for provider in providers:
            with patch.dict(os.environ, {"AGENT_LLM_PROVIDER": provider}):
                settings = AgentServiceSettings()
                assert settings.llm_provider == provider

    def test_temperature_range(self):
        """Test temperature validation."""
        with patch.dict(os.environ, {"AGENT_LLM_TEMPERATURE": "0.0"}):
            settings = AgentServiceSettings()
            assert settings.llm_temperature == 0.0

        with patch.dict(os.environ, {"AGENT_LLM_TEMPERATURE": "2.0"}):
            settings = AgentServiceSettings()
            assert settings.llm_temperature == 2.0

        with patch.dict(os.environ, {"AGENT_LLM_TEMPERATURE": "1.0"}):
            settings = AgentServiceSettings()
            assert settings.llm_temperature == 1.0

    def test_custom_system_prompt(self):
        """Test custom system prompt."""
        custom_prompt = "You are a specialized assistant."

        with patch.dict(os.environ, {"AGENT_SYSTEM_PROMPT": custom_prompt}):
            settings = AgentServiceSettings()
            assert settings.system_prompt == custom_prompt

    def test_tts_timeout(self):
        """Test TTS timeout configuration."""
        with patch.dict(os.environ, {"AGENT_TTS_TIMEOUT": "60.0"}):
            settings = AgentServiceSettings()
            assert settings.tts_timeout == 60.0


class TestObservabilitySettings:
    """Test observability configuration."""

    def test_default_values(self):
        """Test default configuration values."""
        with patch.dict(os.environ, {}, clear=True):
            settings = ObservabilitySettings()

            assert settings.log_level == "INFO"
            assert settings.log_format == "json"
            assert settings.metrics_enabled is True
            assert settings.metrics_port == 9090
            assert settings.tracing_enabled is False
            assert settings.tracing_endpoint is None
            assert settings.service_name == "vibevoice-agent"

    def test_env_override(self):
        """Test environment variable override."""
        with patch.dict(os.environ, {
            "OBS_LOG_LEVEL": "DEBUG",
            "OBS_LOG_FORMAT": "console",
            "OBS_METRICS_ENABLED": "false",
            "OBS_TRACING_ENABLED": "true",
            "OBS_TRACING_ENDPOINT": "http://jaeger:4318",
            "OBS_SERVICE_NAME": "custom-service",
        }):
            settings = ObservabilitySettings()

            assert settings.log_level == "DEBUG"
            assert settings.log_format == "console"
            assert settings.metrics_enabled is False
            assert settings.tracing_enabled is True
            assert settings.tracing_endpoint == "http://jaeger:4318"
            assert settings.service_name == "custom-service"

    def test_log_levels(self):
        """Test different log levels."""
        levels = ["DEBUG", "INFO", "WARNING", "ERROR"]

        for level in levels:
            with patch.dict(os.environ, {"OBS_LOG_LEVEL": level}):
                settings = ObservabilitySettings()
                assert settings.log_level == level

    def test_log_formats(self):
        """Test log formats."""
        formats = ["json", "console"]

        for fmt in formats:
            with patch.dict(os.environ, {"OBS_LOG_FORMAT": fmt}):
                settings = ObservabilitySettings()
                assert settings.log_format == fmt

    def test_tracing_optional(self):
        """Test tracing endpoint is optional."""
        with patch.dict(os.environ, {}, clear=True):
            settings = ObservabilitySettings()
            assert settings.tracing_endpoint is None

        with patch.dict(os.environ, {"OBS_TRACING_ENDPOINT": "http://localhost:4318"}):
            settings = ObservabilitySettings()
            assert settings.tracing_endpoint == "http://localhost:4318"


class TestSettingsCache:
    """Test settings caching functions."""

    def test_tts_settings_cached(self):
        """Test that TTS settings are cached."""
        settings1 = get_tts_settings()
        settings2 = get_tts_settings()

        # Should return same instance due to lru_cache
        assert settings1 is settings2

    def test_agent_settings_cached(self):
        """Test that agent settings are cached."""
        settings1 = get_agent_settings()
        settings2 = get_agent_settings()

        assert settings1 is settings2

    def test_observability_settings_cached(self):
        """Test that observability settings are cached."""
        settings1 = get_observability_settings()
        settings2 = get_observability_settings()

        assert settings1 is settings2

    def test_settings_isolation(self):
        """Test that different settings are separate instances."""
        tts = get_tts_settings()
        agent = get_agent_settings()
        obs = get_observability_settings()

        # All should be different instances
        assert tts is not agent
        assert tts is not obs
        assert agent is not obs


class TestSettingsValidation:
    """Test advanced validation scenarios."""

    def test_extra_fields_ignored(self):
        """Test that extra environment variables are ignored."""
        with patch.dict(os.environ, {
            "TTS_DEVICE": "cpu",
            "TTS_UNKNOWN_FIELD": "should-be-ignored",
        }):
            # Should not raise
            settings = TTSServiceSettings()
            assert settings.device == "cpu"

    def test_type_conversion(self):
        """Test automatic type conversion."""
        with patch.dict(os.environ, {
            "TTS_PORT": "9000",  # String in env
            "TTS_WARMUP_ON_START": "false",  # String boolean
            "TTS_MAX_BATCH_SIZE": "8",  # String int
        }):
            settings = TTSServiceSettings()

            assert isinstance(settings.port, int)
            assert settings.port == 9000
            assert isinstance(settings.warmup_on_start, bool)
            assert settings.warmup_on_start is False
            assert isinstance(settings.max_batch_size, int)
            assert settings.max_batch_size == 8
