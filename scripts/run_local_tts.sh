#!/usr/bin/env bash
# Run TTS service locally for development

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Default environment
export TTS_HOST="${TTS_HOST:-0.0.0.0}"
export TTS_PORT="${TTS_PORT:-8000}"
export TTS_MODEL_NAME="${TTS_MODEL_NAME:-microsoft/VibeVoice-Realtime-0.5B}"
export TTS_DEVICE="${TTS_DEVICE:-cuda:0}"
export TTS_WARMUP_ON_START="${TTS_WARMUP_ON_START:-true}"
export OBS_LOG_LEVEL="${OBS_LOG_LEVEL:-INFO}"
export OBS_LOG_FORMAT="${OBS_LOG_FORMAT:-console}"
export PYTHONPATH="$PROJECT_ROOT/src"

echo "Starting TTS service..."
echo "  Model: $TTS_MODEL_NAME"
echo "  Device: $TTS_DEVICE"
echo "  Port: $TTS_PORT"

python -m uvicorn tts_service.server:app \
    --host "$TTS_HOST" \
    --port "$TTS_PORT" \
    --reload \
    --reload-dir "$PROJECT_ROOT/src"
