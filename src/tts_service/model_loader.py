"""VibeVoice-Realtime model loading and management."""

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Generator

import torch

from common.config import TTSServiceSettings, get_tts_settings
from common.logging import get_logger
from common.metrics import TTS_MODEL_INFO, TTS_MODEL_LOADED

logger = get_logger(__name__)


@dataclass
class ModelInfo:
    """Information about the loaded model."""

    name: str
    device: str
    dtype: str
    sample_rate: int
    loaded_at: float
    warmup_completed: bool = False


class TTSModelManager:
    """Manages the lifecycle of the TTS model."""

    def __init__(self, settings: TTSServiceSettings | None = None) -> None:
        self.settings = settings or get_tts_settings()
        self._model: Any | None = None
        self._processor: Any | None = None
        self._info: ModelInfo | None = None
        self._lock = False

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded and ready."""
        return self._model is not None

    @property
    def info(self) -> ModelInfo | None:
        """Get model information."""
        return self._info

    def load(self) -> None:
        """Load the TTS model into memory."""
        if self._model is not None:
            logger.warning("Model already loaded, skipping")
            return

        logger.info(
            "Loading TTS model",
            model=self.settings.model_name,
            device=self.settings.device,
            dtype=self.settings.dtype,
        )

        start_time = time.perf_counter()

        try:
            # Import here to avoid loading torch at module import time
            from transformers import AutoProcessor, AutoModelForTextToWaveform

            dtype_map = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }
            torch_dtype = dtype_map[self.settings.dtype]

            self._processor = AutoProcessor.from_pretrained(
                self.settings.model_name,
                cache_dir=self.settings.cache_dir,
                trust_remote_code=True,
            )

            self._model = AutoModelForTextToWaveform.from_pretrained(
                self.settings.model_name,
                torch_dtype=torch_dtype,
                cache_dir=self.settings.cache_dir,
                trust_remote_code=True,
            )
            self._model.to(self.settings.device)
            self._model.eval()

            load_duration = time.perf_counter() - start_time

            self._info = ModelInfo(
                name=self.settings.model_name,
                device=self.settings.device,
                dtype=self.settings.dtype,
                sample_rate=self.settings.sample_rate,
                loaded_at=time.time(),
            )

            TTS_MODEL_LOADED.set(1)
            TTS_MODEL_INFO.info({
                "model_name": self.settings.model_name,
                "device": self.settings.device,
                "dtype": self.settings.dtype,
            })

            logger.info("Model loaded successfully", duration_seconds=round(load_duration, 2))

            if self.settings.warmup_on_start:
                self._warmup()

        except Exception as e:
            TTS_MODEL_LOADED.set(0)
            logger.exception("Failed to load model", error=str(e))
            raise RuntimeError(f"Model loading failed: {e}") from e

    def _warmup(self) -> None:
        """Run warmup inference to initialize CUDA kernels."""
        logger.info("Running warmup inference")
        try:
            warmup_text = "Hello, this is a warmup test."
            _ = self.synthesize(warmup_text)
            if self._info:
                self._info.warmup_completed = True
            logger.info("Warmup completed")
        except Exception as e:
            logger.warning("Warmup failed", error=str(e))

    def unload(self) -> None:
        """Unload the model from memory."""
        if self._model is None:
            return

        logger.info("Unloading model")
        del self._model
        del self._processor
        self._model = None
        self._processor = None
        self._info = None
        TTS_MODEL_LOADED.set(0)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @contextmanager
    def inference_context(self) -> Generator[None, None, None]:
        """Context manager for inference with proper CUDA memory management."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")

        try:
            yield
        finally:
            # Clear CUDA cache after inference to prevent memory fragmentation
            if self.settings.device.startswith("cuda"):
                torch.cuda.synchronize()

    def synthesize(self, text: str, voice_id: str | None = None) -> torch.Tensor:
        """
        Synthesize speech from text.

        Args:
            text: Input text to synthesize
            voice_id: Optional voice identifier (for future multi-voice support)

        Returns:
            Audio tensor with shape (samples,)
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")

        if len(text) > self.settings.max_text_length:
            raise ValueError(f"Text exceeds maximum length of {self.settings.max_text_length}")

        with self.inference_context():
            with torch.inference_mode():
                inputs = self._processor(
                    text=text,
                    return_tensors="pt",
                ).to(self.settings.device)

                outputs = self._model.generate(**inputs)
                audio = outputs.squeeze()

        return audio

    def synthesize_streaming(
        self, text: str, chunk_size: int = 4096
    ) -> Generator[torch.Tensor, None, None]:
        """
        Synthesize speech with streaming output.

        This is a placeholder for true streaming support.
        Current implementation generates full audio then chunks it.

        Args:
            text: Input text to synthesize
            chunk_size: Number of samples per chunk

        Yields:
            Audio tensor chunks
        """
        audio = self.synthesize(text)
        for i in range(0, len(audio), chunk_size):
            yield audio[i : i + chunk_size]


# Global model manager instance
_model_manager: TTSModelManager | None = None


def get_model_manager() -> TTSModelManager:
    """Get the global model manager instance."""
    global _model_manager
    if _model_manager is None:
        _model_manager = TTSModelManager()
    return _model_manager
