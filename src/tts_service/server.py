"""FastAPI server for TTS inference service."""

import time
from contextlib import asynccontextmanager
from typing import Annotated, AsyncGenerator

from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.responses import StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from common.config import get_observability_settings, get_tts_settings
from common.logging import get_logger, setup_logging
from common.metrics import (
    TTS_INPUT_LENGTH,
    TTS_OUTPUT_DURATION,
    TTS_REQUEST_DURATION,
    TTS_REQUESTS_TOTAL,
    TTS_TIME_TO_FIRST_AUDIO,
    record_gpu_metrics,
)
from tts_service.model_loader import get_model_manager
from tts_service.streaming import tensor_to_pcm_bytes, tensor_to_wav_bytes

logger = get_logger(__name__)


class SynthesizeRequest(BaseModel):
    """Request body for TTS synthesis."""

    text: str = Field(..., min_length=1, max_length=4096, description="Text to synthesize")
    voice_id: str | None = Field(default=None, description="Voice identifier")
    output_format: str = Field(default="wav", pattern="^(wav|pcm)$", description="Output format")


class SynthesizeResponse(BaseModel):
    """Response metadata for TTS synthesis."""

    duration_seconds: float
    sample_rate: int
    processing_time_ms: float


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    model_name: str | None
    device: str | None
    warmup_completed: bool


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler for startup/shutdown."""
    setup_logging()
    logger.info("Starting TTS service")

    settings = get_tts_settings()
    manager = get_model_manager()

    try:
        manager.load()
        logger.info("TTS service ready", port=settings.port)
        yield
    finally:
        logger.info("Shutting down TTS service")
        manager.unload()


app = FastAPI(
    title="VibeVoice TTS Service",
    description="Real-time text-to-speech synthesis using VibeVoice-Realtime",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health_check() -> HealthResponse:
    """Health check endpoint for liveness/readiness probes."""
    manager = get_model_manager()
    info = manager.info

    return HealthResponse(
        status="healthy" if manager.is_loaded else "unhealthy",
        model_loaded=manager.is_loaded,
        model_name=info.name if info else None,
        device=info.device if info else None,
        warmup_completed=info.warmup_completed if info else False,
    )


@app.get("/ready", tags=["system"])
async def readiness_check() -> dict[str, str]:
    """Readiness probe - only ready when model is loaded and warmed up."""
    manager = get_model_manager()
    info = manager.info

    if not manager.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if info and not info.warmup_completed:
        raise HTTPException(status_code=503, detail="Model warmup not completed")

    return {"status": "ready"}


@app.get("/metrics", tags=["system"])
async def metrics() -> Response:
    """Prometheus metrics endpoint."""
    settings = get_tts_settings()
    record_gpu_metrics(settings.device)
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/synthesize", tags=["tts"], response_model=None)
async def synthesize(
    request: SynthesizeRequest,
    stream: Annotated[bool, Query(description="Stream audio response")] = False,
) -> Response | StreamingResponse:
    """
    Synthesize speech from text.

    Returns audio as WAV or raw PCM bytes.
    """
    start_time = time.perf_counter()
    manager = get_model_manager()
    settings = get_tts_settings()

    if not manager.is_loaded:
        TTS_REQUESTS_TOTAL.labels(endpoint="synthesize", status="error").inc()
        raise HTTPException(status_code=503, detail="Model not loaded")

    TTS_INPUT_LENGTH.observe(len(request.text))

    try:
        if stream:
            first_chunk_time: float | None = None

            async def audio_stream() -> AsyncGenerator[bytes, None]:
                nonlocal first_chunk_time
                for chunk in manager.synthesize_streaming(request.text):
                    if first_chunk_time is None:
                        first_chunk_time = time.perf_counter()
                        TTS_TIME_TO_FIRST_AUDIO.observe(first_chunk_time - start_time)
                    yield tensor_to_pcm_bytes(chunk)

            return StreamingResponse(
                audio_stream(),
                media_type="audio/wav" if request.output_format == "wav" else "audio/pcm",
                headers={
                    "X-Sample-Rate": str(settings.sample_rate),
                    "X-Channels": "1",
                    "X-Bits-Per-Sample": "16",
                },
            )

        # Non-streaming synthesis
        audio = manager.synthesize(request.text, request.voice_id)
        duration = time.perf_counter() - start_time

        audio_duration = len(audio) / settings.sample_rate
        TTS_OUTPUT_DURATION.observe(audio_duration)

        if request.output_format == "wav":
            audio_bytes = tensor_to_wav_bytes(audio, settings.sample_rate)
            media_type = "audio/wav"
        else:
            audio_bytes = tensor_to_pcm_bytes(audio)
            media_type = "audio/pcm"

        TTS_REQUEST_DURATION.labels(endpoint="synthesize", status="success").observe(duration)
        TTS_REQUESTS_TOTAL.labels(endpoint="synthesize", status="success").inc()

        return Response(
            content=audio_bytes,
            media_type=media_type,
            headers={
                "X-Processing-Time-Ms": str(round(duration * 1000, 2)),
                "X-Audio-Duration-Seconds": str(round(audio_duration, 2)),
                "X-Sample-Rate": str(settings.sample_rate),
            },
        )

    except ValueError as e:
        TTS_REQUESTS_TOTAL.labels(endpoint="synthesize", status="error").inc()
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        duration = time.perf_counter() - start_time
        TTS_REQUEST_DURATION.labels(endpoint="synthesize", status="error").observe(duration)
        TTS_REQUESTS_TOTAL.labels(endpoint="synthesize", status="error").inc()
        logger.exception("Synthesis failed", error=str(e))
        raise HTTPException(status_code=500, detail="Synthesis failed") from e


@app.get("/info", tags=["system"])
async def model_info() -> dict:
    """Get information about the loaded model."""
    manager = get_model_manager()
    info = manager.info

    if not info:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "model_name": info.name,
        "device": info.device,
        "dtype": info.dtype,
        "sample_rate": info.sample_rate,
        "warmup_completed": info.warmup_completed,
    }


def main() -> None:
    """Entry point for the TTS service."""
    import uvicorn

    settings = get_tts_settings()
    obs_settings = get_observability_settings()

    uvicorn.run(
        "tts_service.server:app",
        host=settings.host,
        port=settings.port,
        workers=settings.workers,
        log_level=obs_settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
