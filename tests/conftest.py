"""Pytest configuration and fixtures."""

import pytest
import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


@pytest.fixture(scope="session")
def anyio_backend():
    """Configure async backend for pytest-asyncio."""
    return "asyncio"


@pytest.fixture
def sample_audio_tensor():
    """Create sample audio tensor for testing."""
    import torch
    return torch.randn(24000)  # 1 second at 24kHz


@pytest.fixture
def sample_text():
    """Sample text for TTS testing."""
    return "Hello, this is a test message."
