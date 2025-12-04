# tts-ops: production-ready realtime TTS infrastructure built on VibeVoice

Production-grade realtime AI voice agent platform using Microsoft's VibeVoice-Realtime-0.5B model.

## Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                           Kubernetes Cluster                           │
├────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────┐  │
│  │   Ingress    │───▶│ Agent Service│───▶│     TTS Service (GPU)    │  │
│  │   (NGINX)    │    │  (FastAPI)   │    │  (VibeVoice-Realtime)    │  │
│  └──────────────┘    └──────────────┘    └──────────────────────────┘  │
│         │                   │                        │                 │
│         ▼                   ▼                        ▼                 │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                    Observability Stack                           │  │
│  │  Prometheus ─────▶ Grafana ─────▶ AlertManager                   │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────┘
```

## Features

- **Low-latency TTS**: Sub-500ms time-to-first-audio using VibeVoice-Realtime-0.5B
- **Streaming support**: WebSocket and HTTP streaming for real-time audio delivery
- **GPU optimization**: CUDA-accelerated inference with automatic batch processing
- **Production observability**: Prometheus metrics, structured logging, distributed tracing
- **Kubernetes-native**: Helm charts with HPA, PDB, and resource management
- **Health monitoring**: Liveness/readiness probes with circuit breaker patterns

## Quick Start

### Prerequisites

- Python 3.11+
- Docker with NVIDIA Container Toolkit
- Kubernetes cluster with GPU nodes (for production)
- CUDA 12.1+ compatible GPU

### Local Development

```bash
# Install dependencies
pip install -e ".[dev]"

# Start TTS service (requires GPU)
./scripts/run_local_tts.sh

# Start agent service (separate terminal)
./scripts/run_local_agent.sh

# Test the endpoint
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, how are you today?"}'
```

### Docker Compose (Development)

```bash
docker-compose up -d
```

### Kubernetes Deployment

```bash
# Add Helm repo and install
helm upgrade --install voice-agent ./infra/k8s/helm/voice-agent \
  --namespace voice-agent \
  --create-namespace \
  -f infra/k8s/helm/voice-agent/values-production.yaml
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `TTS_MODEL_NAME` | HuggingFace model identifier | `microsoft/VibeVoice-Realtime-0.5B` |
| `TTS_DEVICE` | Inference device | `cuda:0` |
| `TTS_MAX_BATCH_SIZE` | Maximum batch size for inference | `4` |
| `LLM_PROVIDER` | LLM backend (`openai`, `anthropic`, `local`) | `anthropic` |
| `LLM_MODEL` | Model name for LLM | `claude-sonnet-4-20250514` |
| `LOG_LEVEL` | Logging verbosity | `INFO` |
| `METRICS_PORT` | Prometheus metrics port | `9090` |

### Helm Values

See `infra/k8s/helm/voice-agent/values.yaml` for full configuration options.

## API Reference

### POST /chat

Generate speech from text input.

**Request:**
```json
{
  "text": "Hello, world!",
  "voice_id": "default",
  "stream": true
}
```

**Response (streaming):** Audio chunks as `audio/wav` stream

### POST /synthesize

Direct TTS synthesis without LLM processing.

**Request:**
```json
{
  "text": "Text to synthesize",
  "voice_id": "default",
  "output_format": "wav"
}
```

### GET /health

Health check endpoint returning service status.

### GET /metrics

Prometheus-formatted metrics endpoint.

## Observability

### Key Metrics

- `tts_request_duration_seconds` - TTS request latency histogram
- `tts_time_to_first_audio_seconds` - Time to first audio byte
- `tts_requests_total` - Total request count by status
- `tts_gpu_utilization_percent` - GPU utilization
- `tts_model_load_time_seconds` - Model initialization time

### Grafana Dashboards

Import dashboards from `infra/monitoring/dashboards/`:
- `voice-agent-overview.json` - High-level service health
- `tts-performance.json` - Detailed TTS metrics

## Testing

```bash
# Unit tests
pytest tests/ -v

# Integration tests (requires running services)
pytest tests/integration/ -v --integration

# Load testing
python scripts/load_test_tts.py --rps 10 --duration 60
```

## Project Structure

```
vibevoice-realtime-agent/
├── src/
│   ├── agent_app/          # Voice agent service
│   ├── tts_service/        # TTS microservice
│   └── common/             # Shared utilities
├── infra/
│   ├── docker/             # Dockerfiles
│   ├── k8s/helm/           # Helm charts
│   ├── terraform/          # Infrastructure as code
│   └── monitoring/         # Prometheus/Grafana configs
├── scripts/                # Development scripts
├── tests/                  # Test suites
└── docs/                   # Additional documentation
```

## License

MIT License - see [LICENSE](LICENSE) for details.
