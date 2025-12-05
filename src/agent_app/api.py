"""FastAPI application for voice agent service."""

import builtins
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager, suppress
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from agent_app.llm_client import get_llm_client
from agent_app.tts_client import TTSClientError, get_tts_client
from common.config import get_agent_settings, get_observability_settings
from common.logging import get_logger, setup_logging
from common.metrics import (
    AGENT_ACTIVE_CONNECTIONS,
    AGENT_CHAT_DURATION,
    AGENT_LLM_DURATION,
    AGENT_REQUESTS_TOTAL,
)

logger = get_logger(__name__)


class ChatRequest(BaseModel):
    """Request body for chat endpoint."""

    text: str = Field(..., min_length=1, max_length=2000, description="User input text")
    voice_id: str | None = Field(default=None, description="Voice for TTS output")
    conversation_id: str | None = Field(default=None, description="Conversation ID for context")
    include_audio: bool = Field(default=True, description="Include TTS audio in response")


class ChatResponse(BaseModel):
    """Response body for chat endpoint."""

    text: str = Field(..., description="Assistant response text")
    conversation_id: str | None = Field(default=None, description="Conversation ID")
    processing_time_ms: float = Field(..., description="Total processing time in milliseconds")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    tts_available: bool
    llm_provider: str


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler."""
    setup_logging()
    logger.info("Starting agent service")

    settings = get_agent_settings()

    # Initialize clients
    tts_client = get_tts_client()
    await tts_client.connect()

    # Wait for TTS service
    logger.info("Waiting for TTS service", url=settings.tts_base_url)
    if await tts_client.wait_for_ready(timeout=30.0):
        logger.info("TTS service ready")
    else:
        logger.warning("TTS service not ready, continuing anyway")

    logger.info("Agent service ready", port=settings.port, llm_provider=settings.llm_provider)

    try:
        yield
    finally:
        logger.info("Shutting down agent service")
        await tts_client.close()


app = FastAPI(
    title="Voice Agent Service",
    description="AI voice agent with LLM and TTS integration",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    settings = get_agent_settings()
    tts_client = get_tts_client()

    tts_healthy = await tts_client.health_check()

    return HealthResponse(
        status="healthy",
        tts_available=tts_healthy,
        llm_provider=settings.llm_provider,
    )


@app.get("/metrics", tags=["system"])
async def metrics() -> Response:
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/chat", tags=["chat"], response_model=None)
async def chat(
    request: ChatRequest,
    stream: Annotated[bool, Query(description="Stream audio response")] = False,
) -> ChatResponse | StreamingResponse:
    """
    Process a chat message and return response with optional TTS audio.

    In non-streaming mode, returns JSON with text response.
    In streaming mode, returns audio stream with text in headers.
    """
    start_time = time.perf_counter()
    settings = get_agent_settings()

    try:
        # Generate LLM response
        llm_start = time.perf_counter()
        llm_client = get_llm_client()

        messages = [{"role": "user", "content": request.text}]
        response_text = await llm_client.generate(messages)

        llm_duration = time.perf_counter() - llm_start
        AGENT_LLM_DURATION.labels(
            provider=settings.llm_provider, model=settings.llm_model
        ).observe(llm_duration)

        logger.info(
            "LLM response generated",
            input_length=len(request.text),
            output_length=len(response_text),
            duration_ms=round(llm_duration * 1000, 2),
        )

        # Generate TTS if requested
        if request.include_audio and stream:
            tts_client = get_tts_client()

            async def audio_stream() -> AsyncGenerator[bytes, None]:
                async for chunk in tts_client.synthesize_streaming(
                    response_text, request.voice_id
                ):
                    yield chunk

            total_time = time.perf_counter() - start_time
            AGENT_CHAT_DURATION.labels(status="success").observe(total_time)
            AGENT_REQUESTS_TOTAL.labels(endpoint="chat", status="success").inc()

            return StreamingResponse(
                audio_stream(),
                media_type="audio/wav",
                headers={
                    "X-Response-Text": response_text[:500],  # Truncate for header size
                    "X-Processing-Time-Ms": str(round(total_time * 1000, 2)),
                },
            )

        total_time = time.perf_counter() - start_time
        AGENT_CHAT_DURATION.labels(status="success").observe(total_time)
        AGENT_REQUESTS_TOTAL.labels(endpoint="chat", status="success").inc()

        return ChatResponse(
            text=response_text,
            conversation_id=request.conversation_id,
            processing_time_ms=round(total_time * 1000, 2),
        )

    except TTSClientError as e:
        AGENT_REQUESTS_TOTAL.labels(endpoint="chat", status="error").inc()
        logger.warning("TTS error", error=str(e))
        raise HTTPException(status_code=503, detail=f"TTS service error: {e}") from e
    except Exception as e:
        total_time = time.perf_counter() - start_time
        AGENT_CHAT_DURATION.labels(status="error").observe(total_time)
        AGENT_REQUESTS_TOTAL.labels(endpoint="chat", status="error").inc()
        logger.exception("Chat processing failed", error=str(e))
        raise HTTPException(status_code=500, detail="Internal error") from e


@app.post("/synthesize", tags=["tts"])
async def synthesize_audio(
    text: str = Query(..., min_length=1, max_length=4000),
    voice_id: str | None = Query(default=None),
) -> Response:
    """
    Direct TTS synthesis without LLM processing.

    Proxies request to TTS service.
    """
    tts_client = get_tts_client()

    try:
        audio = await tts_client.synthesize(text, voice_id)
        return Response(content=audio, media_type="audio/wav")
    except TTSClientError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket) -> None:
    """
    WebSocket endpoint for real-time chat.

    Protocol:
    - Client sends JSON: {"text": "...", "voice_id": "..."}
    - Server sends JSON: {"type": "text", "content": "..."} for text chunks
    - Server sends binary audio chunks
    - Server sends JSON: {"type": "done"} when complete
    """
    await websocket.accept()
    AGENT_ACTIVE_CONNECTIONS.inc()

    get_agent_settings()
    llm_client = get_llm_client()
    tts_client = get_tts_client()

    try:
        while True:
            data = await websocket.receive_json()
            text = data.get("text", "")
            voice_id = data.get("voice_id")

            if not text:
                await websocket.send_json({"type": "error", "message": "Empty text"})
                continue

            # Stream LLM response
            full_response = ""
            messages = [{"role": "user", "content": text}]

            async for chunk in llm_client.generate_streaming(messages):
                full_response += chunk
                await websocket.send_json({"type": "text", "content": chunk})

            # Generate and stream audio
            try:
                async for audio_chunk in tts_client.synthesize_streaming(
                    full_response, voice_id
                ):
                    await websocket.send_bytes(audio_chunk)
            except TTSClientError as e:
                logger.warning("TTS streaming failed", error=str(e))

            await websocket.send_json({"type": "done"})

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.exception("WebSocket error", error=str(e))
        with suppress(builtins.BaseException):
            await websocket.send_json({"type": "error", "message": str(e)})
    finally:
        AGENT_ACTIVE_CONNECTIONS.dec()


def main() -> None:
    """Entry point for the agent service."""
    import uvicorn

    settings = get_agent_settings()
    obs_settings = get_observability_settings()

    uvicorn.run(
        "agent_app.api:app",
        host=settings.host,
        port=settings.port,
        workers=settings.workers,
        log_level=obs_settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
