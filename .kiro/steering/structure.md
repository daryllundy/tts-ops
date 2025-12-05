# Project Structure

## Directory Layout

```
vibevoice-realtime-agent/
├── src/                    # Source code
│   ├── agent_app/         # Voice agent service
│   │   ├── api.py         # FastAPI application and endpoints
│   │   ├── llm_client.py  # LLM provider abstraction
│   │   └── tts_client.py  # TTS service client
│   ├── tts_service/       # TTS microservice
│   │   ├── server.py      # FastAPI TTS server
│   │   ├── model_loader.py # Model management and inference
│   │   └── streaming.py   # Audio streaming utilities
│   └── common/            # Shared utilities
│       ├── config.py      # Pydantic settings
│       ├── logging.py     # Structured logging setup
│       └── metrics.py     # Prometheus metrics definitions
├── infra/                 # Infrastructure code
│   ├── docker/           # Dockerfiles
│   ├── k8s/              # Kubernetes manifests
│   │   └── helm/         # Helm charts
│   └── terraform/        # IaC for cloud resources
├── scripts/              # Development and utility scripts
├── tests/                # Test suites
└── mnt/user-data/        # Runtime data directory
```

## Code Organization Patterns

### Service Structure

Each service follows a consistent pattern:
- `api.py` or `server.py` - FastAPI application with lifespan management
- Client modules - External service integrations
- Shared `common/` package for cross-service utilities

### Configuration Management

All configuration uses Pydantic Settings with:
- Environment variable prefixes (`TTS_`, `AGENT_`, `OBS_`)
- Type validation and field constraints
- Cached singleton getters (`@lru_cache`)
- `.env` file support

### API Endpoints

Standard endpoint structure:
- `/health` - Health check (liveness probe)
- `/ready` - Readiness check (k8s readiness probe)
- `/metrics` - Prometheus metrics
- Service-specific endpoints (`/chat`, `/synthesize`, etc.)

### Observability

Metrics, logging, and tracing are centralized in `common/`:
- Prometheus metrics defined as module-level constants
- Structured logging with context fields
- Consistent error handling and status tracking

## Import Conventions

- Absolute imports from package roots: `from common.config import ...`
- Type hints using modern syntax: `str | None` instead of `Optional[str]`
- Pydantic models for all request/response schemas

## Testing Structure

- `tests/conftest.py` - Shared fixtures
- `tests/test_*.py` - Unit tests per module
- Integration tests marked with `@pytest.mark.integration`
