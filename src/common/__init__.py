"""Common utilities shared across services."""

from .config import (
    get_agent_settings,
    get_observability_settings,
    get_tts_settings,
)
from .logging import get_logger, setup_logging
from .metrics import (
    AGENT_CHAT_DURATION,
    AGENT_LLM_DURATION,
    AGENT_REQUESTS_TOTAL,
    TTS_REQUEST_DURATION,
    TTS_REQUESTS_TOTAL,
    TTS_TIME_TO_FIRST_AUDIO,
)

__all__ = [
    "get_tts_settings",
    "get_agent_settings",
    "get_observability_settings",
    "setup_logging",
    "get_logger",
    "TTS_REQUEST_DURATION",
    "TTS_REQUESTS_TOTAL",
    "TTS_TIME_TO_FIRST_AUDIO",
    "AGENT_CHAT_DURATION",
    "AGENT_LLM_DURATION",
    "AGENT_REQUESTS_TOTAL",
]
