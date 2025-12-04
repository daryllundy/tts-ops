"""Streaming utilities for real-time audio delivery."""

import asyncio
import io
import struct
from dataclasses import dataclass
from typing import AsyncGenerator

import numpy as np
import soundfile as sf
import torch

from common.logging import get_logger

logger = get_logger(__name__)


@dataclass
class AudioChunk:
    """Represents a chunk of audio data."""

    data: bytes
    sample_rate: int
    is_final: bool = False
    sequence_number: int = 0


def create_wav_header(sample_rate: int, num_channels: int = 1, bits_per_sample: int = 16) -> bytes:
    """
    Create a WAV header for streaming audio.

    For streaming, we use a placeholder size that will be updated
    or we rely on chunked transfer encoding.
    """
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8

    # Use max int32 as placeholder for streaming
    data_size = 0x7FFFFFFF
    file_size = data_size + 36

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        file_size,
        b"WAVE",
        b"fmt ",
        16,  # PCM format chunk size
        1,  # Audio format (PCM)
        num_channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        data_size,
    )

    return header


def tensor_to_wav_bytes(audio: torch.Tensor, sample_rate: int) -> bytes:
    """Convert audio tensor to WAV bytes."""
    if audio.dim() > 1:
        audio = audio.squeeze()

    audio_np = audio.cpu().numpy()

    # Normalize to [-1, 1] if needed
    if audio_np.max() > 1.0 or audio_np.min() < -1.0:
        audio_np = audio_np / max(abs(audio_np.max()), abs(audio_np.min()))

    buffer = io.BytesIO()
    sf.write(buffer, audio_np, sample_rate, format="WAV", subtype="PCM_16")
    buffer.seek(0)
    return buffer.read()


def tensor_to_pcm_bytes(audio: torch.Tensor) -> bytes:
    """Convert audio tensor to raw PCM bytes (16-bit signed)."""
    if audio.dim() > 1:
        audio = audio.squeeze()

    audio_np = audio.cpu().numpy()

    # Normalize and convert to int16
    if audio_np.max() > 1.0 or audio_np.min() < -1.0:
        audio_np = audio_np / max(abs(audio_np.max()), abs(audio_np.min()))

    audio_int16 = (audio_np * 32767).astype(np.int16)
    return audio_int16.tobytes()


async def stream_audio_chunks(
    audio_generator: AsyncGenerator[torch.Tensor, None],
    sample_rate: int,
    include_header: bool = True,
) -> AsyncGenerator[bytes, None]:
    """
    Stream audio chunks as bytes.

    Args:
        audio_generator: Async generator yielding audio tensors
        sample_rate: Audio sample rate
        include_header: Whether to include WAV header at start

    Yields:
        Audio bytes suitable for streaming response
    """
    if include_header:
        yield create_wav_header(sample_rate)

    async for chunk in audio_generator:
        pcm_bytes = tensor_to_pcm_bytes(chunk)
        yield pcm_bytes
        await asyncio.sleep(0)  # Allow other tasks to run


class AudioBuffer:
    """Thread-safe buffer for accumulating audio chunks."""

    def __init__(self, sample_rate: int) -> None:
        self.sample_rate = sample_rate
        self._chunks: list[torch.Tensor] = []
        self._lock = asyncio.Lock()

    async def add_chunk(self, chunk: torch.Tensor) -> None:
        """Add an audio chunk to the buffer."""
        async with self._lock:
            self._chunks.append(chunk)

    async def get_complete_audio(self) -> torch.Tensor:
        """Get all accumulated audio as a single tensor."""
        async with self._lock:
            if not self._chunks:
                return torch.tensor([])
            return torch.cat(self._chunks)

    async def clear(self) -> None:
        """Clear the buffer."""
        async with self._lock:
            self._chunks.clear()

    @property
    async def duration_seconds(self) -> float:
        """Get total duration of buffered audio in seconds."""
        async with self._lock:
            total_samples = sum(c.shape[0] for c in self._chunks)
            return total_samples / self.sample_rate
