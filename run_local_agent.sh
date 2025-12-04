#!/usr/bin/env bash
# Run agent service locally for development

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Default environment
export AGENT_HOST="${AGENT_HOST:-0.0.0.0}"
export AGENT_PORT="${AGENT_PORT:-8080}"
export AGENT_TTS_BASE_URL="${AGENT_TTS_BASE_URL:-http://localhost:8000}"
export AGENT_LLM_PROVIDER="${AGENT_LLM_PROVIDER:-anthropic}"
export AGENT_LLM_MODEL="${AGENT_LLM_MODEL:-claude-sonnet-4-20250514}"
export OBS_LOG_LEVEL="${OBS_LOG_LEVEL:-INFO}"
export OBS_LOG_FORMAT="${OBS_LOG_FORMAT:-console}"
export PYTHONPATH="$PROJECT_ROOT/src"

# Check for API key
if [[ "$AGENT_LLM_PROVIDER" == "anthropic" ]] && [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
    echo "Warning: ANTHROPIC_API_KEY not set"
fi

if [[ "$AGENT_LLM_PROVIDER" == "openai" ]] && [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "Warning: OPENAI_API_KEY not set"
fi

echo "Starting Agent service..."
echo "  TTS URL: $AGENT_TTS_BASE_URL"
echo "  LLM Provider: $AGENT_LLM_PROVIDER"
echo "  LLM Model: $AGENT_LLM_MODEL"
echo "  Port: $AGENT_PORT"

python -m uvicorn agent_app.api:app \
    --host "$AGENT_HOST" \
    --port "$AGENT_PORT" \
    --reload \
    --reload-dir "$PROJECT_ROOT/src"
