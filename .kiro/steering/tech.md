# Technology Stack

## Core Framework

- **Python 3.11+** - Primary language
- **FastAPI** - Web framework for both services
- **Uvicorn** - ASGI server
- **Pydantic v2** - Data validation and settings management

## ML/AI Stack

- **PyTorch 2.1+** - Deep learning framework
- **Transformers** - HuggingFace model loading
- **VibeVoice-Realtime-0.5B** - Microsoft's TTS model
- **CUDA 12.1+** - GPU acceleration

## LLM Providers

- Anthropic (Claude)
- OpenAI
- Local models (configurable)

## Observability

- **Prometheus** - Metrics collection
- **Grafana** - Visualization
- **structlog** - Structured logging (JSON format)
- **OpenTelemetry** - Distributed tracing (optional)

## Infrastructure

- **Docker** - Containerization with NVIDIA Container Toolkit
- **Kubernetes** - Orchestration
- **Helm** - Package management
- **Terraform** - Infrastructure as code

## Development Tools

- **pytest** - Testing framework with async support
- **ruff** - Linting and formatting
- **mypy** - Type checking
- **pre-commit** - Git hooks

## Common Commands

### Local Development

```bash
# Install dependencies
pip install -e ".[dev]"

# Start TTS service (requires GPU)
./scripts/run_local_tts.sh

# Start agent service
./scripts/run_local_agent.sh

# Run tests
pytest tests/ -v

# Type checking
mypy src/

# Linting
ruff check src/
```

### Docker

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Kubernetes

```bash
# Deploy with Helm
helm upgrade --install voice-agent ./infra/k8s/helm/voice-agent \
  --namespace voice-agent --create-namespace

# Check status
kubectl get pods -n voice-agent

# View logs
kubectl logs -n voice-agent -l app=tts-service -f
```

## Configuration

All services use Pydantic Settings with environment variable overrides:
- TTS settings: `TTS_*` prefix
- Agent settings: `AGENT_*` prefix  
- Observability: `OBS_*` prefix

Configuration can be provided via `.env` files or environment variables.
